"""reproducibility.py

Reproducibility checks for the double-IHC pipeline: do repeated scans of the
same tissue, repeated staining batches, or a full rerun of the classification
workflow on a control TMA agree with each other.

run_workflow_reproducibility classifies cells with
apply_group_specific_thresholds_5groups (one GMM per marker per staining group)
and expects the same nested thresholds dict that function produces,
{"DAB++": {"GATA6": thr, "CK17": thr}, ...}.

For the scanning and staining checks, Visiopharm exports one TSV file per core
with the core-level mean intensity for the two markers, GATA6 and CK17. The
expected folder layout is:

    <base_folder>/<acquisition_date_or_batch>/<TMA>/*.tsv

where each TSV is a single-row core summary.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

from __future__ import annotations

import os
import glob
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parsing helpers
# -----------------------------


def _normalize_core_id(core_str: str) -> str:
    """Normalize core formatting to "NN, L" (zero-padded)."""
    try:
        number, letter = core_str.replace(" ", "").split(",")
        if number.isdigit():
            number = number.zfill(2)
        return f"{number}, {letter}"
    except Exception:
        return str(core_str)


def _extract_core_id(name_value: str) -> str:
    """Extract the core identifier from the Visiopharm Name column.

    Example:
        "Grid (01, A)" -> "01, A"

    If the pattern cannot be parsed, returns the original string.
    """
    if not isinstance(name_value, str):
        return str(name_value)

    m = re.search(r"\(([^)]+)\)\s*$", name_value)
    if not m:
        return name_value

    return _normalize_core_id(m.group(1))


def _to_float_safe(x) -> float:
    """Convert values to float safely (non-convertible -> NaN)."""
    try:
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------
# Data loading
# -----------------------------


def load_reproducibility_data(
    base_folder: str,
    file_pattern: str = "*_objects.tsv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load per-core TSV exports and return a standardized core-level DataFrame.

    The output DataFrame contains only the 5 requested columns:
        1) TMA                      -> from 'Study level 5'
        2) Core                     -> from 'Name' (content inside parentheses)
        3) Acquisition date         -> from 'Study level 4'
        4) Mean Intensity GATA6/Core
        5) Mean Intensity CK17/Core

    Args:
        base_folder (str): Root folder containing the reproducibility study.
        file_pattern (str): Glob pattern to identify per-core TSV files.
        verbose (bool): If True, prints basic progress info.

    Returns:
        pd.DataFrame: Concatenated, clean DataFrame.
    """
    if not os.path.isdir(base_folder):
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    files = sorted(
        glob.glob(os.path.join(base_folder, "**", file_pattern), recursive=True)
    )
    if not files:
        raise FileNotFoundError(
            f"No files found under '{base_folder}' with pattern '{file_pattern}'."
        )

    col_tma = "Study level 5"
    col_date = "Study level 4"
    col_name = "Name"
    col_ck17 = "Mean Intensity CK17/Core"
    col_gata6 = "Mean Intensity GATA6/Core"

    rows: List[Dict] = []

    for fp in files:
        try:
            df = pd.read_csv(fp, sep="	")
        except Exception as e:
            if verbose:
                print(f"Could not read '{fp}': {e}")
            continue

        if df.empty:
            if verbose:
                print(f"Skipping empty file: {fp}")
            continue

        missing = [
            c
            for c in [col_tma, col_date, col_name, col_ck17, col_gata6]
            if c not in df.columns
        ]
        if missing:
            if verbose:
                print(f"Skipping '{fp}' (missing columns: {missing})")
            continue

        r0 = df.iloc[0]
        rows.append(
            {
                "TMA": str(r0[col_tma]),
                "Core": _extract_core_id(str(r0[col_name])),
                "Acquisition date": str(r0[col_date]),
                "Mean Intensity GATA6/Core": _to_float_safe(r0[col_gata6]),
                "Mean Intensity CK17/Core": _to_float_safe(r0[col_ck17]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No valid rows could be loaded. Check folder structure and TSV columns."
        )

    out["TMA"] = out["TMA"].astype(str)
    out["Core"] = out["Core"].astype(str)
    out["Acquisition date"] = out["Acquisition date"].astype(str)

    out = out[
        [
            "TMA",
            "Core",
            "Acquisition date",
            "Mean Intensity GATA6/Core",
            "Mean Intensity CK17/Core",
        ]
    ].copy()

    out = out.sort_values(["TMA", "Acquisition date", "Core"]).reset_index(drop=True)

    if verbose:
        print(
            f"Loaded {len(out):,} cores from {len(files):,} files "
            f"({out['TMA'].nunique()} TMAs, {out['Acquisition date'].nunique()} dates)."
        )

    return out


# -----------------------------
# Plotting
# -----------------------------


def _compute_stats(vals: pd.Series) -> Tuple[float, float, float]:
    """Return (mean, min, max) for a numeric series, ignoring NaNs."""
    vals = pd.to_numeric(vals, errors="coerce")
    if vals.dropna().empty:
        return (float("nan"), float("nan"), float("nan"))
    return (float(vals.mean()), float(vals.min()), float(vals.max()))


def _add_stats_box(ax, groups, colors, get_stats, g_range, c_range, range_label):
    """Render a unified legend-style stats box to the right of the plot.

    A single clean box (matching the legend visual style) is anchored to the
    bottom-right outside area. Each group gets two coloured rows (GATA6 / CK17)
    showing mean, [min – max] and Δ mean (CK17 − GATA6).

    Args:
        ax: Target axes.
        groups: Ordered list of group labels (dates or batches).
        colors: {group: hex_colour}.
        get_stats: Callable(group) -> (g_mean, g_min, g_max, c_mean, c_min, c_max, delta_mean).
        g_range / c_range: Cross-group range of means.
        range_label: e.g. "Δdays(mean)" or "Δbatches(mean)".
    """
    from matplotlib.patches import FancyBboxPatch

    LINE  = 0.052   # axes-fraction height per text line
    X0    = 1.015   # left edge of the box (axes fraction, just outside axes)
    XPAD  = 0.008   # inner horizontal padding
    YPAD  = 0.012   # inner vertical padding
    BOX_W = 0.52    # box width in axes fraction

    # Total lines: title + range header + separator + col header + 2 lines per group
    n_lines = 4 + len(groups) * 2
    box_h   = n_lines * LINE + 2 * YPAD

    # Single background patch anchored at y=0 (bottom of axes, outside)
    ax.add_patch(FancyBboxPatch(
        (X0 - XPAD, 0.0),
        BOX_W, box_h,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        facecolor="white", edgecolor="#bbbbbb",
        linewidth=0.8, clip_on=False, zorder=5,
    ))

    y = box_h - YPAD  # current y, stepping downward

    # Title
    ax.text(X0, y, "Statistics",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, fontweight="bold", color="black",
            clip_on=False, zorder=6)
    y -= LINE

    # Cross-group range header
    ax.text(X0, y, f"{range_label}:  GATA6 = {g_range:.2f}   CK17 = {c_range:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color="#555555",
            clip_on=False, zorder=6)
    y -= LINE

    # Visual separator
    ax.text(X0, y, "─" * 52,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=5.5, color="#cccccc",
            clip_on=False, zorder=6, family="monospace")
    y -= LINE * 0.65

    # Column header
    ax.text(X0, y, "group / marker        mean     [min – max]      Δ (CK17−GATA6)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color="#888888", fontstyle="italic",
            clip_on=False, zorder=6)
    y -= LINE

    # Two rows per group, coloured by group
    for group in groups:
        col = colors[group]
        gm, gn, gx, cm, cn, cx, delta = get_stats(group)

        ax.text(X0, y,
                f"{group}   GATA6   {gm:.2f}  [{gn:.2f} – {gx:.2f}]",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color=col, clip_on=False, zorder=6)
        y -= LINE

        ax.text(X0, y,
                f"{group}   CK17    {cm:.2f}  [{cn:.2f} – {cx:.2f}]   Δ = {delta:.2f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color=col, clip_on=False, zorder=6)
        y -= LINE



def plot_reproducibility_profiles(
    df: pd.DataFrame,
    outpath: str,
    tma_order: Optional[List[str]] = None,
    date_order: Optional[List[str]] = None,
    dpi: int = 150,
) -> None:
    """Single-axes plot per TMA for scanning reproducibility.

    Layout
    ------
    - One figure per TMA saved as <outpath_base>_<TMA>.svg and .png
    - Color encodes acquisition date (colorblind-safe Wong palette)
    - Line style encodes marker: solid = GATA6, dashed = CK17
    - Summary statistics table rendered below the axes

    Args:
        df (pd.DataFrame): Output of :func:`load_reproducibility_data`.
        outpath (str): Base path without extension, e.g.
                       'output/reproducibility/scanning/reproducibility_profiles'
        tma_order (Optional[List[str]]): Explicit TMA order.
        date_order (Optional[List[str]]): Explicit acquisition date order.
        dpi (int): Output DPI for PNG.
    """
    required = {
        "TMA", "Core", "Acquisition date",
        "Mean Intensity GATA6/Core", "Mean Intensity CK17/Core",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    tmas  = tma_order  if tma_order  is not None else sorted(df["TMA"].unique())
    dates = date_order if date_order is not None else sorted(df["Acquisition date"].unique())

    # Colorblind-safe palette (Wong 2011)
    DATE_PALETTE = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    date_colors  = {d: DATE_PALETTE[i % len(DATE_PALETTE)] for i, d in enumerate(dates)}

    outbase = os.path.splitext(outpath)[0]
    os.makedirs(os.path.dirname(outbase) or ".", exist_ok=True)

    for tma in tmas:
        df_tma = df[df["TMA"] == tma].copy()
        all_cores  = sorted(df_tma["Core"].unique())
        core_index = {c: i for i, c in enumerate(all_cores)}

        n_cores   = len(all_cores)
        fig, ax = plt.subplots(figsize=(max(12, n_cores * 0.35), 7))

        for date in dates:
            sub = df_tma[df_tma["Acquisition date"] == date].sort_values("Core")
            if sub.empty:
                continue
            x   = [core_index[c] for c in sub["Core"]]
            y_g = pd.to_numeric(sub["Mean Intensity GATA6/Core"], errors="coerce")
            y_c = pd.to_numeric(sub["Mean Intensity CK17/Core"],  errors="coerce")
            col = date_colors[date]
            ax.plot(x, y_g, color=col, linestyle="-",  linewidth=1.5,
                    marker="o", markersize=3, label=f"{date} — GATA6")
            ax.plot(x, y_c, color=col, linestyle="--", linewidth=1.5,
                    marker="o", markersize=3, label=f"{date} — CK17")

        ax.set_xticks(np.arange(len(all_cores)))
        ax.set_xticklabels(all_cores, rotation=90, fontsize=7)
        ax.set_xlabel("Core", fontsize=10)
        ax.set_ylabel("Mean intensity", fontsize=10)
        ax.set_title(f"Scanning reproducibility — TMA: {tma}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

        # Legend: one unified box — colour per date + linestyle section
        from matplotlib.lines import Line2D
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color="none", label=""),                         # spacer
            Line2D([0], [0], color="gray", linestyle="-",  linewidth=0.5,
                   label="── line style ──"),                                 # section label
            Line2D([0], [0], color="black", linestyle="-",  linewidth=1.5, label="GATA6"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="CK17"),
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=8, framealpha=0.9)

        # ── Stats box ────────────────────────────────────────────────────────
        g_day_means = df_tma.groupby("Acquisition date")["Mean Intensity GATA6/Core"].mean()
        c_day_means = df_tma.groupby("Acquisition date")["Mean Intensity CK17/Core"].mean()
        g_day_range = float(g_day_means.max() - g_day_means.min())
        c_day_range = float(c_day_means.max() - c_day_means.min())

        def _scan_get_stats(date):
            sub = df_tma[df_tma["Acquisition date"] == date]
            gm, gn, gx = _compute_stats(pd.to_numeric(sub["Mean Intensity GATA6/Core"], errors="coerce"))
            cm, cn, cx = _compute_stats(pd.to_numeric(sub["Mean Intensity CK17/Core"],  errors="coerce"))
            return gm, gn, gx, cm, cn, cx, cm - gm

        _add_stats_box(ax, dates, date_colors, _scan_get_stats,
                       g_day_range, c_day_range, "Δdays(mean)")

        plt.tight_layout()
        safe_tma = str(tma).replace(" ", "_").replace("/", "-")
        plt.savefig(f"{outbase}_{safe_tma}.svg", bbox_inches="tight")
        plt.savefig(f"{outbase}_{safe_tma}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Scanning reproducibility saved: {outbase}_{safe_tma}.svg | .png")


def run_reproducibility(
    base_folder: str,
    outdir: str,
    outname: str = "reproducibility_profiles",
    tma_order: Optional[List[str]] = None,
    date_order: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: load scanning reproducibility data and generate plots.

    Produces one SVG + one PNG per TMA, named <outname>_<TMA>.svg / .png

    Args:
        base_folder (str): Root folder containing the reproducibility study.
        outdir (str): Output directory for plots.
        outname (str): Base filename (no extension).
        tma_order (Optional[List[str]]): Explicit TMA order.
        date_order (Optional[List[str]]): Explicit acquisition date order.
        verbose (bool): Verbosity for the loader.

    Returns:
        pd.DataFrame: The loaded standardized DataFrame.
    """
    df = load_reproducibility_data(base_folder=base_folder, verbose=verbose)
    outpath = os.path.join(outdir, outname)
    plot_reproducibility_profiles(
        df=df, outpath=outpath, tma_order=tma_order, date_order=date_order
    )
    return df


# -----------------------------
# Staining reproducibility
# -----------------------------


def load_staining_reproducibility_data(
    base_folder: str,
    file_pattern: str = "*_objects.tsv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load per-core TSV exports for the staining reproducibility study.

    Folder structure expected:
        data/raw/Reproducibility/Staining/<Batch>/*.tsv

    The output DataFrame mirrors the scanning study but uses 'Batch' instead of
    'Acquisition date', since 'Study level 5' holds the batch label and there is
    a single TMA (same TMA re-stained across batches).

    Columns returned:
        Batch | Core | Mean Intensity GATA6/Core | Mean Intensity CK17/Core

    Args:
        base_folder (str): Root folder for the staining study
                           (e.g. 'data/raw/Reproducibility/Staining').
        file_pattern (str): Glob pattern to identify per-core TSV files.
        verbose (bool): If True, prints basic progress info.

    Returns:
        pd.DataFrame: Concatenated, clean DataFrame.
    """
    if not os.path.isdir(base_folder):
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    files = sorted(
        glob.glob(os.path.join(base_folder, "**", file_pattern), recursive=True)
    )
    if not files:
        raise FileNotFoundError(
            f"No files found under '{base_folder}' with pattern '{file_pattern}'."
        )

    col_batch = "Study level 5"   # e.g. "Batch1"
    col_name  = "Name"
    col_ck17  = "Mean Intensity CK17/Core"
    col_gata6 = "Mean Intensity GATA6/Core"

    rows: List[Dict] = []

    for fp in files:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            if verbose:
                print(f"Could not read '{fp}': {e}")
            continue

        if df.empty:
            if verbose:
                print(f"Skipping empty file: {fp}")
            continue

        missing = [c for c in [col_batch, col_name, col_ck17, col_gata6] if c not in df.columns]
        if missing:
            if verbose:
                print(f"Skipping '{fp}' (missing columns: {missing})")
            continue

        r0 = df.iloc[0]
        rows.append(
            {
                "Batch": str(r0[col_batch]),
                "Core": _extract_core_id(str(r0[col_name])),
                "Mean Intensity GATA6/Core": _to_float_safe(r0[col_gata6]),
                "Mean Intensity CK17/Core":  _to_float_safe(r0[col_ck17]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid rows could be loaded. Check folder structure and TSV columns.")

    out = out[["Batch", "Core", "Mean Intensity GATA6/Core", "Mean Intensity CK17/Core"]].copy()
    out = out.sort_values(["Batch", "Core"]).reset_index(drop=True)

    if verbose:
        print(
            f"Loaded {len(out):,} cores from {len(files):,} files "
            f"({out['Batch'].nunique()} batches)."
        )

    return out


def plot_staining_reproducibility_profiles(
    df: pd.DataFrame,
    outpath: str,
    batch_order: Optional[List[str]] = None,
    dpi: int = 150,
) -> None:
    """Single-axes plot for staining reproducibility across batches.

    Layout
    ------
    - One figure (SVG + PNG) with all batches overlaid.
    - Color encodes batch (colorblind-safe Wong palette).
    - Line style encodes marker: solid = GATA6, dashed = CK17.
    - Summary statistics table rendered below the axes.

    Args:
        df (pd.DataFrame): Output of :func:`load_staining_reproducibility_data`.
        outpath (str): Base path without extension,
                       e.g. 'output/reproducibility/staining/staining_profiles'
        batch_order (Optional[List[str]]): Explicit batch ordering.
        dpi (int): Output DPI for PNG.
    """
    required = {"Batch", "Core", "Mean Intensity GATA6/Core", "Mean Intensity CK17/Core"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {sorted(missing)}")

    batches = batch_order if batch_order is not None else sorted(df["Batch"].unique())
    if not batches:
        raise ValueError("No batches found in the data.")

    # Colorblind-safe palette (Wong 2011)
    BATCH_PALETTE = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    batch_colors  = {b: BATCH_PALETTE[i % len(BATCH_PALETTE)] for i, b in enumerate(batches)}

    all_cores  = sorted(df["Core"].unique())
    core_index = {c: i for i, c in enumerate(all_cores)}
    n_cores    = len(all_cores)

    fig, ax = plt.subplots(figsize=(max(12, n_cores * 0.35), 9))

    for batch in batches:
        sub = df[df["Batch"] == batch].sort_values("Core")
        if sub.empty:
            continue
        x   = [core_index[c] for c in sub["Core"]]
        y_g = pd.to_numeric(sub["Mean Intensity GATA6/Core"], errors="coerce")
        y_c = pd.to_numeric(sub["Mean Intensity CK17/Core"],  errors="coerce")
        col = batch_colors[batch]
        ax.plot(x, y_g, color=col, linestyle="-",  linewidth=1.5,
                marker="o", markersize=3, label=f"{batch} — GATA6")
        ax.plot(x, y_c, color=col, linestyle="--", linewidth=1.5,
                marker="o", markersize=3, label=f"{batch} — CK17")

    ax.set_xticks(np.arange(n_cores))
    ax.set_xticklabels(all_cores, rotation=90, fontsize=7)
    ax.set_xlabel("Core", fontsize=10)
    ax.set_ylabel("Mean intensity", fontsize=10)
    ax.set_title("Staining reproducibility — per-core intensity profiles",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Legend: one unified box — colour per batch + linestyle section
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="none", label=""),                         # spacer
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=0.5,
               label="── line style ──"),                                 # section label
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.5, label="GATA6"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="CK17"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.9)

    # ── Stats box ────────────────────────────────────────────────────────────
    g_batch_means = df.groupby("Batch")["Mean Intensity GATA6/Core"].mean()
    c_batch_means = df.groupby("Batch")["Mean Intensity CK17/Core"].mean()
    g_batch_range = float(g_batch_means.max() - g_batch_means.min())
    c_batch_range = float(c_batch_means.max() - c_batch_means.min())

    def _st_get_stats(batch):
        sub = df[df["Batch"] == batch]
        gm, gn, gx = _compute_stats(pd.to_numeric(sub["Mean Intensity GATA6/Core"], errors="coerce"))
        cm, cn, cx = _compute_stats(pd.to_numeric(sub["Mean Intensity CK17/Core"],  errors="coerce"))
        return gm, gn, gx, cm, cn, cx, cm - gm

    _add_stats_box(ax, batches, batch_colors, _st_get_stats,
                   g_batch_range, c_batch_range, "Δbatches(mean)")

    plt.tight_layout()
    outbase = os.path.splitext(outpath)[0]
    os.makedirs(os.path.dirname(outbase) or ".", exist_ok=True)
    plt.savefig(f"{outbase}.svg", bbox_inches="tight")
    plt.savefig(f"{outbase}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Staining reproducibility saved: {outbase}.svg | .png")


def run_staining_reproducibility(
    base_folder: str,
    outdir: str,
    outname: str = "staining_reproducibility_profiles",
    batch_order: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: load staining data and generate the batch reproducibility plot.

    Produces one SVG + one PNG named <outname>.svg / .png

    Usage in main:
        df_staining = run_staining_reproducibility(
            base_folder=STAINING_REPRO_FOLDER,
            outdir=f"{OUTPUT_ROOT}/reproducibility/staining",
            batch_order=STAINING_BATCH_ORDER,
        )

    Args:
        base_folder (str): Root folder with one subfolder per batch.
        outdir (str): Output directory for the plot.
        outname (str): Base filename (no extension).
        batch_order (Optional[List[str]]): Explicit batch order.
                                           If None, batches are sorted alphabetically.
        verbose (bool): Verbosity for the loader.

    Returns:
        pd.DataFrame: The loaded standardized DataFrame.
    """
    df = load_staining_reproducibility_data(base_folder=base_folder, verbose=verbose)
    outpath = os.path.join(outdir, outname)
    plot_staining_reproducibility_profiles(df=df, outpath=outpath, batch_order=batch_order)
    return df


# -----------------------------
# Workflow reproducibility
# -----------------------------


def _profile_stability_metrics(mat: np.ndarray) -> Tuple[float, float]:
    """Magnitude-aware complement to the rank-only Kendall's W.

    Args:
        mat: (n_batches, n_populations) array — one row per batch, one column
            per population (% cells), no NaNs.

    Returns:
        (profile_corr, mean_abs_drift):
          - profile_corr: mean Pearson r between every pair of batch rows
            (all C(n_batches, 2) pairs). Unlike Kendall's W (ranks only), this
            is sensitive to the relative size of the 4 populations, not just
            their order — two batches with an identical ranking but very
            different proportions score lower here than in W.
          - mean_abs_drift: for each population, the SD of its % across
            batches; averaged over the 4 populations. This is the literal
            "± X percentage points" figure: how much, on average, each
            population's share moves from batch to batch for this core/patient.
    """
    n_batches = mat.shape[0]
    corrs = []
    for i in range(n_batches):
        for j in range(i + 1, n_batches):
            if np.std(mat[i]) == 0 or np.std(mat[j]) == 0:
                continue  # constant row (e.g. one population at 100%) -> r undefined
            corrs.append(float(np.corrcoef(mat[i], mat[j])[0, 1]))
    profile_corr = float(np.mean(corrs)) if corrs else float("nan")
    mean_abs_drift = float(np.mean(np.std(mat, axis=0)))
    return profile_corr, mean_abs_drift


def run_workflow_reproducibility(
    base_folder: str,
    outdir: str,
    thresholds: Dict[str, Dict[str, float]],
    processed_folder: Optional[str] = None,
    batch_order: Optional[List[str]] = None,
    dpi: int = 150,
    verbose: bool = True,
    group_col: str = "Staining",
    tma_to_batch: Optional[Dict[str, str]] = None,
    clean_patient_ids: Optional[set] = None,
    batch_factors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Workflow reproducibility: classify H12O-OPT1 across 4 batches with fixed thresholds.

    Loads raw data from base_folder (same folder structure as Low_dilatation_5groups_trial),
    normalises each TMA/batch independently with its own q01-q99, applies fixed
    thresholds from the main analysis (grouped by `group_col`), computes core-level
    population endpoints, and produces two reproducibility plots plus a numeric metrics CSV.

    Args:
        base_folder: Root folder (one subfolder per TMA/batch).
        outdir: Output directory for plots and metrics.
        thresholds: Fixed thresholds from the main analysis, keyed by `group_col` values:
            - group_col="Staining"       → {"DAB++": {...}, "Low_dab": {...}, ...}
            - group_col="Batch_str"      → {"1": {...}, "2": {...}, "4": {...}, ...}
            - group_col="Batch_Staining" → {"1 | DAB++": {...}, ...}
        processed_folder: Cache folder for pre-processed data. Defaults to
            <parent of base_folder>/processed/<base_folder_name>.
        batch_order: Explicit TMA/batch label order (e.g. ["H12O-OPT11", ...]).
        dpi: PNG output DPI.
        verbose: Print progress messages.
        group_col: Column used to assign thresholds, matching however the main
            analysis grouped cells for GMM fitting ("Staining", "Batch_str" or
            "Batch_Staining").
        tma_to_batch: Maps each control TMA/batch folder name to its batch label
            (e.g. {"H12O-OPT11": "1", "H12O-OPT12": "2", ...}). Required for
            group_col in ("Batch_str", "Batch_Staining"), since in the control the
            "batch" is the TMA itself.
        clean_patient_ids: If given, a set of whitespace-stripped Patient IDs (same
            normalization as main.py's cohort filter and bias_assessment.load_and_split).
            The control TMA's cores are restricted to these patients before anything
            downstream (normalization, classification, concordance). Pass None (default)
            to use every patient on the control TMA (the "complete" cohort — no filter).
        batch_factors: Optional staining-batch correction factors from
            batch_correction.compute_control_factors(). When given, the control
            TMA's raw intensities are corrected with them right after loading
            (before normalization), exactly as the main cohort is in main.py.
            None (default) leaves the control TMA uncorrected.

    Returns:
        pd.DataFrame: Core-level endpoints for all batches combined.
    """
    from matplotlib.lines import Line2D
    from data_preprocessing import load_data_if_exists, drop_cores_with_all_nan_patients, validation_calculations, export_cells_per_core_in_raw_order
    from normalization import normalize_intensities_by_tma_quantiles
    from cell_classification import apply_group_specific_thresholds_5groups, add_core_hscores_to_cells

    if processed_folder is None:
        processed_folder = os.path.join(
            os.path.dirname(os.path.abspath(base_folder)),
            "processed",
            os.path.basename(base_folder),
        )

    os.makedirs(outdir, exist_ok=True)

    # 1. Load
    if verbose:
        print("  Loading workflow reproducibility data...")
    data = load_data_if_exists(base_folder, processed_folder)
    data = drop_cores_with_all_nan_patients(data)
    data = data.dropna(subset=["DAB Intensity - Nuclei", "Mag Intensity - Cytoplasm"])

    # 1b. Clean-cohort filter (same ID-matching pattern as main.py / bias_assessment):
    # restrict the control TMA to patients also in the publication roster. No count is
    # hardcoded here, the resulting numbers are printed so they can be checked by hand.
    if clean_patient_ids is not None:
        n_patients_before = data["ID"].nunique()
        n_cores_before = data[["TMA", "Core"]].drop_duplicates().shape[0]
        ids_norm = data["ID"].astype(str).str.replace(r"\s+", "", regex=True)
        data = data[ids_norm.isin(clean_patient_ids)].copy()
        if verbose:
            print(
                f"  [clean filter] Control TMA: {n_patients_before} patients / "
                f"{n_cores_before} cores before -> "
                f"{data['ID'].nunique()} patients / "
                f"{data[['TMA', 'Core']].drop_duplicates().shape[0]} cores after."
            )

    # 1c. Optional staining-batch correction (batch_correction.py), same factors
    # and same position in the pipeline as for the main cohort: on raw
    # intensities, before the per-TMA q01-q99 normalization.
    if batch_factors is not None:
        from batch_correction import apply_batch_correction
        if verbose:
            print("  Applying staining-batch correction to the control TMA...")
        data = apply_batch_correction(data, batch_factors)

    # 2. Normalise per TMA/batch independently
    if verbose:
        print("  Normalising per TMA/batch (independent q01–q99)...")
    data_norm, q_table = normalize_intensities_by_tma_quantiles(
        df=data,
        tma_col="TMA",
        intensity_cols={
            "GATA6": "DAB Intensity - Nuclei",
            "CK17": "Mag Intensity - Cytoplasm",
        },
        q_low=0.01,
        q_high=0.99,
        suffix="_norm",
        clip=True,
        outdir=os.path.join(outdir, "normalization"),
    )

    # Project Phenotyping columns onto the same per-TMA normalised scale
    data_norm = data_norm.merge(
        q_table[["TMA", "GATA6_q01", "GATA6_denom", "CK17_q01", "CK17_denom"]],
        on="TMA", how="left",
    )
    data_norm["DAB Intensity - Nuclei Phenotyping_norm"] = (
        (data_norm["DAB Intensity - Nuclei Phenotyping"] - data_norm["GATA6_q01"])
        / data_norm["GATA6_denom"]
    ).clip(0, 1)
    data_norm["Mag Intensity - Cytoplasm Phenotyping_norm"] = (
        (data_norm["Mag Intensity - Cytoplasm Phenotyping"] - data_norm["CK17_q01"])
        / data_norm["CK17_denom"]
    ).clip(0, 1)
    data_norm.drop(
        columns=["GATA6_q01", "GATA6_denom", "CK17_q01", "CK17_denom"], inplace=True
    )

    # 3. Classify cells with fixed thresholds from main analysis.
    # Build the mode's grouping column on the control cells. In the control the
    # "batch" is the TMA itself, so Batch_str is derived from TMA via tma_to_batch.
    if group_col in ("Batch_str", "Batch_Staining"):
        if tma_to_batch is None:
            raise ValueError(
                f"group_col='{group_col}' requires tma_to_batch (control TMA → batch label)."
            )
        data_norm["Batch_str"] = data_norm["TMA"].map(tma_to_batch).astype("string")
        if data_norm["Batch_str"].isna().any():
            unmapped = sorted(data_norm.loc[data_norm["Batch_str"].isna(), "TMA"].unique())
            raise ValueError(f"tma_to_batch has no entry for control TMAs: {unmapped}")
        if group_col == "Batch_Staining":
            data_norm["Batch_Staining"] = (
                data_norm["Batch_str"] + " | " + data_norm["Staining"].astype("string")
            )

    # Exclude control groups with no threshold for this group_col (out-of-model
    # combinations) — e.g. a batch/staining-group pairing the main cohort never had,
    # so there's no fitted threshold for it. Those cells are dropped rather than
    # silently classified with the wrong group's threshold.
    valid_groups = set(thresholds.keys())
    present_groups = set(data_norm[group_col].dropna().unique())
    missing_groups = sorted(present_groups - valid_groups)
    if missing_groups:
        drop_mask = data_norm[group_col].isin(missing_groups)
        n_cells = int(drop_mask.sum())
        n_cores = data_norm.loc[drop_mask].groupby(["TMA", "Core"]).ngroups
        print(
            f"  [WARN] {len(missing_groups)} '{group_col}' group(s) have no threshold and are "
            f"excluded from the control workflow (out-of-model): {missing_groups} "
            f"→ dropped {n_cells} cells / {n_cores} cores."
        )
        data_norm = data_norm[~drop_mask].copy()

    if verbose:
        print(f"  Classifying cells with fixed thresholds (group_col='{group_col}')...")
    data_norm = apply_group_specific_thresholds_5groups(
        df=data_norm,
        staining_col=group_col,
        group_thresholds=thresholds,
        dab_intensity_col="DAB Intensity - Nuclei Phenotyping_norm",
        ck17_intensity_col="Mag Intensity - Cytoplasm Phenotyping_norm",
        out_suffix="",
    )

    # 4. Core-level endpoints (same pipeline as main analysis)
    # 4a. Broadcast H-scores back to cell level (required by validation_calculations)
    data_norm, _ = add_core_hscores_to_cells(
        cell_df=data_norm,
        group_cols=("TMA", "Core", "Patient", "ID"),
        out_suffix="",
    )
    # 4b. Run validation_calculations for all metrics (populations, H-scores, heterogeneity)
    core_endpoints = validation_calculations(
        df=data_norm,
        output_dir=os.path.join(outdir, "validation"),
        thresholds=thresholds,
        group_col=group_col,
    )

    # 4c. Export per-core cell tables in raw Visiopharm order for the control batches.
    # Order files are read from the workflow processed cache; outputs are written under
    # outdir (kept separate from the main analysis processed folder to avoid mixing batches).
    if verbose:
        print("  Exporting per-core cells in raw order (control batches)...")
    export_cells_per_core_in_raw_order(
        cell_df=data_norm,
        processed_folder=processed_folder,
        out_root=outdir,
    )

    # Shared constants
    pop_cols = ["% GATA6+CK17+", "% GATA6+CK17-", "% GATA6-CK17+", "% GATA6-CK17-"]
    subtype_labels = {
        "% GATA6+CK17+": "GATA6+CK17+ (Hybrid)",
        "% GATA6+CK17-": "GATA6+CK17- (Classical)",
        "% GATA6-CK17+": "GATA6-CK17+ (Basal)",
        "% GATA6-CK17-": "GATA6-CK17- (Undifferentiated)",
    }
    color_pop = {
        "% GATA6+CK17+": "#98df8a",
        "% GATA6+CK17-": "#2ca02c",
        "% GATA6-CK17+": "#d62728",
        "% GATA6-CK17-": "#808080",
    }
    BATCH_PALETTE = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

    batches = (
        batch_order
        if batch_order is not None
        else sorted(core_endpoints["TMA"].unique())
    )
    batch_colors = {b: BATCH_PALETTE[i % len(BATCH_PALETTE)] for i, b in enumerate(batches)}
    all_cores = sorted(core_endpoints["Core"].unique())
    core_index = {c: i for i, c in enumerate(all_cores)}

    # ── Plot 1: stacked horizontal bars, one panel per batch ─────────────────
    n_b = len(batches)
    n_cores = len(all_cores)
    fig1, axes1 = plt.subplots(
        1, n_b,
        figsize=(5 * n_b, max(6, n_cores * 0.28)),
        sharey=True,
    )
    if n_b == 1:
        axes1 = [axes1]

    # Shared y-positions: all cores from all batches in one consistent order.
    # Missing cores per batch are plotted as empty bars (0 %) with a gray band
    # so panels are aligned and comparable at a glance.
    y_pos = np.arange(n_cores)

    for ax, batch in zip(axes1, batches):
        batch_cores = set(core_endpoints.loc[core_endpoints["TMA"] == batch, "Core"])
        # Reindex to the full union core list; missing cores become NaN → 0 after fillna
        sub_vals = (
            core_endpoints[core_endpoints["TMA"] == batch]
            .set_index("Core")
            .reindex(all_cores)[pop_cols]
            .fillna(0)
        )
        pop_vals = sub_vals.values
        left = np.zeros(n_cores)
        for i, col in enumerate(pop_cols):
            ax.barh(
                y_pos, pop_vals[:, i], left=left,
                label=subtype_labels[col], color=color_pop[col],
            )
            left += pop_vals[:, i]
        # Gray band on rows where this batch has no data
        for j, core in enumerate(all_cores):
            if core not in batch_cores:
                ax.axhspan(j - 0.5, j + 0.5, color="#d8d8d8", alpha=0.85, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_cores, fontsize=6)
        ax.set_xlabel("% cells", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_title(batch, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # Legend on rightmost panel (deduplicated)
    handles_l, labels_l = axes1[-1].get_legend_handles_labels()
    seen: Dict[str, object] = {}
    for h, l in zip(handles_l, labels_l):
        if l not in seen:
            seen[l] = h
    axes1[-1].legend(
        list(seen.values()), list(seen.keys()),
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
    )
    fig1.suptitle(
        "Workflow reproducibility — population composition by core",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out1 = os.path.join(outdir, "workflow_population_composition")
    plt.savefig(f"{out1}.svg", bbox_inches="tight")
    plt.savefig(f"{out1}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    if verbose:
        print(f"  Plot 1 saved: {out1}.svg | .png")

    # ── Plot 2: overlaid lines per population, one panel per population ───────
    fig2, axes2 = plt.subplots(
        1, len(pop_cols),
        figsize=(6 * len(pop_cols), 6),
        sharey=False,
    )
    if len(pop_cols) == 1:
        axes2 = [axes2]

    for ax, col in zip(axes2, pop_cols):
        for batch in batches:
            sub = core_endpoints[core_endpoints["TMA"] == batch].sort_values("Core")
            x = [core_index[c] for c in sub["Core"] if c in core_index]
            y = sub.loc[sub["Core"].isin(core_index), col].values
            ax.plot(
                x, y, color=batch_colors[batch],
                linewidth=1.5, marker="o", markersize=3, label=batch,
            )
        ax.set_xticks(range(len(all_cores)))
        ax.set_xticklabels(all_cores, rotation=90, fontsize=6)
        ax.set_xlabel("Core", fontsize=9)
        ax.set_ylabel("% cells", fontsize=9)
        ax.set_title(subtype_labels[col], fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    batch_handles = [
        Line2D([0], [0], color=batch_colors[b], linewidth=1.5, label=b)
        for b in batches
    ]
    fig2.legend(
        handles=batch_handles,
        bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=9,
    )
    fig2.suptitle(
        "Workflow reproducibility — per-population variability across batches",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out2 = os.path.join(outdir, "workflow_population_lines")
    plt.savefig(f"{out2}.svg", bbox_inches="tight")
    plt.savefig(f"{out2}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    if verbose:
        print(f"  Plot 2 saved: {out2}.svg | .png")

    # ── Metrics: mean variability across batches ──────────────────────────────
    metric_rows = []
    for col in pop_cols:
        pivot = core_endpoints.pivot_table(
            index="Core", columns="TMA", values=col,
        )
        pivot = pivot.reindex(columns=[b for b in batches if b in pivot.columns])
        ranges = pivot.max(axis=1) - pivot.min(axis=1)
        metric_rows.append({
            "Population": subtype_labels[col],
            "Mean range (%)": round(float(ranges.mean()), 2),
            "Std range (%)": round(float(ranges.std()), 2),
            "n cores": int(ranges.notna().sum()),
        })

    # ── Profile concordance: Kendall's W + dominant/full-rank consistency ─────
    # For each core: rank the 4 populations (highest % = rank 1) in each batch.
    # Kendall's W measures how concordant those rankings are across batches (0–1).
    pivots = {
        col: core_endpoints.pivot_table(index="Core", columns="TMA", values=col)
                           .reindex(columns=[b for b in batches if b in core_endpoints["TMA"].unique()])
        for col in pop_cols
    }
    complete_cores = pivots[pop_cols[0]].dropna().index.tolist()

    kendall_w_vals, dominant_ok, fullrank_ok, processed_cores = [], [], [], []
    profile_corr_vals, mean_abs_drift_vals = [], []
    for core in complete_cores:
        # mat shape: (n_batches, n_populations)
        mat = np.array([[pivots[col].loc[core, b] for col in pop_cols] for b in batches])
        if np.any(np.isnan(mat)):
            continue

        # Ranks within each batch (rank 1 = most abundant population)
        ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
        k, n = ranks.shape  # batches, populations
        R = ranks.sum(axis=0)
        S = np.sum((R - k * (n + 1) / 2) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n))
        kendall_w_vals.append(W)

        dominant_per_batch = [int(np.argmax(mat[i])) for i in range(k)]
        dominant_ok.append(len(set(dominant_per_batch)) == 1)

        full_rank_per_batch = [tuple(np.argsort(-mat[i])) for i in range(k)]
        fullrank_ok.append(len(set(full_rank_per_batch)) == 1)

        profile_corr, mean_abs_drift = _profile_stability_metrics(mat)
        profile_corr_vals.append(profile_corr)
        mean_abs_drift_vals.append(mean_abs_drift)

        processed_cores.append(core)

    # Save per-core concordance for downstream cross-reference analysis
    core_concordance_df = pd.DataFrame({
        "Core": processed_cores,
        "Kendall_W": kendall_w_vals,
        "dominant_consistent": dominant_ok,
        "fullrank_consistent": fullrank_ok,
        "profile_corr": profile_corr_vals,
        "mean_abs_drift_pct": mean_abs_drift_vals,
    })

    # Cell count per (Core, batch): number of cell-level rows in data_norm
    cell_count_pivot = (
        data_norm.groupby(["Core", "TMA"]).size()
        .unstack(fill_value=0)
        .rename(columns=lambda b: f"n_cells_{b}")
    )
    cell_count_cols = [f"n_cells_{b}" for b in batches if f"n_cells_{b}" in cell_count_pivot.columns]
    cell_count_pivot = cell_count_pivot[cell_count_cols].reset_index()
    core_concordance_df = core_concordance_df.merge(cell_count_pivot, on="Core", how="left")

    concordance_path = os.path.join(outdir, "workflow_core_concordance.csv")
    core_concordance_df.to_csv(concordance_path, index=False)

    n_complete = len(kendall_w_vals)
    mean_W  = round(float(np.mean(kendall_w_vals)), 3)
    std_W   = round(float(np.std(kendall_w_vals)),  3)
    pct_dom = round(100 * float(np.mean(dominant_ok)),  1)
    pct_full = round(100 * float(np.mean(fullrank_ok)), 1)
    mean_profile_corr = round(float(np.nanmean(profile_corr_vals)), 3)
    std_profile_corr  = round(float(np.nanstd(profile_corr_vals)),  3)
    mean_drift = round(float(np.mean(mean_abs_drift_vals)), 2)
    std_drift  = round(float(np.std(mean_abs_drift_vals)),  2)

    concordance_rows = [
        {"Population": "Kendall's W — profile concordance (mean ± std across cores)",
         "Mean range (%)": mean_W, "Std range (%)": std_W, "n cores": n_complete},
        {"Population": "% cores: same dominant population in all batches",
         "Mean range (%)": pct_dom, "Std range (%)": "", "n cores": n_complete},
        {"Population": "% cores: identical full ranking in all batches",
         "Mean range (%)": pct_full, "Std range (%)": "", "n cores": n_complete},
        {"Population": "Profile correlation — mean pairwise Pearson r across batches (mean ± std across cores)",
         "Mean range (%)": mean_profile_corr, "Std range (%)": std_profile_corr, "n cores": n_complete},
        {"Population": "Mean absolute drift per population, ± percentage points (mean ± std across cores)",
         "Mean range (%)": mean_drift, "Std range (%)": std_drift, "n cores": n_complete},
    ]

    metrics_df = pd.DataFrame(metric_rows + concordance_rows)

    print("\nWorkflow reproducibility — mean variability across batches "
          "(mean ± std between cores):")
    for _, row in metrics_df.head(len(pop_cols)).iterrows():
        print(
            f"  {row['Population']:<44}: "
            f"{row['Mean range (%)']:.2f} ± {row['Std range (%)']:.2f} %"
        )
    print(f"\nProfile concordance across {n_complete} cores (all 4 batches):")
    print(f"  Kendall's W:                   {mean_W:.3f} ± {std_W:.3f}")
    print(f"  Profile correlation (r):       {mean_profile_corr:.3f} ± {std_profile_corr:.3f}")
    print(f"  Mean abs. drift (± pts):       {mean_drift:.2f} ± {std_drift:.2f}")
    print(f"  Same dominant population:      {pct_dom:.1f}% of cores")
    print(f"  Identical full ranking:        {pct_full:.1f}% of cores")

    # ── Patient-level reproducibility ─────────────────────────────────────────
    # Aggregate per (Patient, TMA/batch): mean of the 4 population columns across
    # all cores of that patient in that batch — the same per-patient pseudo-bulk
    # idea as patient_aggregation.build_patient_pseudobulk. Concordance is then
    # measured across the 4 batches for each patient, exactly as for cores above.
    if verbose:
        print("\n  Computing patient-level reproducibility...")

    patient_col = "Patient"
    if patient_col not in core_endpoints.columns:
        if verbose:
            print(f"  Column '{patient_col}' not found — skipping patient-level analysis.")
        patient_metrics_df = pd.DataFrame()
        patient_concordance_df = pd.DataFrame()
    else:
        # Pseudo-bulk: mean across cores per (Patient, batch)
        patient_endpoints = (
            core_endpoints.groupby([patient_col, "TMA"])[pop_cols]
            .mean()
            .reset_index()
        )

        # Mean-range metrics (same logic as core-level)
        pat_metric_rows = []
        for col in pop_cols:
            ppivot = patient_endpoints.pivot_table(
                index=patient_col, columns="TMA", values=col,
            )
            ppivot = ppivot.reindex(columns=[b for b in batches if b in ppivot.columns])
            pranges = ppivot.max(axis=1) - ppivot.min(axis=1)
            pat_metric_rows.append({
                "Population": subtype_labels[col],
                "Mean range (%)": round(float(pranges.mean()), 2),
                "Std range (%)": round(float(pranges.std()), 2),
                "n patients": int(pranges.notna().sum()),
            })

        # Kendall's W + dominant/full-rank consistency per patient
        ppivots = {
            col: patient_endpoints.pivot_table(
                index=patient_col, columns="TMA", values=col
            ).reindex(columns=[b for b in batches if b in patient_endpoints["TMA"].unique()])
            for col in pop_cols
        }
        complete_patients = ppivots[pop_cols[0]].dropna().index.tolist()

        pkendall_w, pdom_ok, pfullrank_ok, processed_patients = [], [], [], []
        p_profile_corr_vals, p_mean_abs_drift_vals = [], []
        for patient in complete_patients:
            pmat = np.array(
                [[ppivots[col].loc[patient, b] for col in pop_cols] for b in batches]
            )
            if np.any(np.isnan(pmat)):
                continue

            pranks = np.argsort(np.argsort(-pmat, axis=1), axis=1) + 1
            pk, pn = pranks.shape
            pR = pranks.sum(axis=0)
            pS = np.sum((pR - pk * (pn + 1) / 2) ** 2)
            pW = 12 * pS / (pk ** 2 * (pn ** 3 - pn))
            pkendall_w.append(pW)

            pdom_per_batch = [int(np.argmax(pmat[i])) for i in range(pk)]
            pdom_ok.append(len(set(pdom_per_batch)) == 1)

            pfull_per_batch = [tuple(np.argsort(-pmat[i])) for i in range(pk)]
            pfullrank_ok.append(len(set(pfull_per_batch)) == 1)

            p_profile_corr, p_mean_abs_drift = _profile_stability_metrics(pmat)
            p_profile_corr_vals.append(p_profile_corr)
            p_mean_abs_drift_vals.append(p_mean_abs_drift)

            processed_patients.append(patient)

        n_pat = len(pkendall_w)
        pmean_W  = round(float(np.mean(pkendall_w)), 3) if n_pat else float("nan")
        pstd_W   = round(float(np.std(pkendall_w)),  3) if n_pat else float("nan")
        ppct_dom = round(100 * float(np.mean(pdom_ok)),  1) if n_pat else float("nan")
        ppct_full = round(100 * float(np.mean(pfullrank_ok)), 1) if n_pat else float("nan")
        p_mean_profile_corr = round(float(np.nanmean(p_profile_corr_vals)), 3) if n_pat else float("nan")
        p_std_profile_corr  = round(float(np.nanstd(p_profile_corr_vals)),  3) if n_pat else float("nan")
        p_mean_drift = round(float(np.mean(p_mean_abs_drift_vals)), 2) if n_pat else float("nan")
        p_std_drift  = round(float(np.std(p_mean_abs_drift_vals)),  2) if n_pat else float("nan")

        pat_concordance_rows = [
            {"Population": "Kendall's W — profile concordance (mean ± std across patients)",
             "Mean range (%)": pmean_W, "Std range (%)": pstd_W, "n patients": n_pat},
            {"Population": "% patients: same dominant population in all batches",
             "Mean range (%)": ppct_dom, "Std range (%)": "", "n patients": n_pat},
            {"Population": "% patients: identical full ranking in all batches",
             "Mean range (%)": ppct_full, "Std range (%)": "", "n patients": n_pat},
            {"Population": "Profile correlation — mean pairwise Pearson r across batches (mean ± std across patients)",
             "Mean range (%)": p_mean_profile_corr, "Std range (%)": p_std_profile_corr, "n patients": n_pat},
            {"Population": "Mean absolute drift per population, ± percentage points (mean ± std across patients)",
             "Mean range (%)": p_mean_drift, "Std range (%)": p_std_drift, "n patients": n_pat},
        ]
        patient_metrics_df = pd.DataFrame(pat_metric_rows + pat_concordance_rows)

        patient_concordance_df = pd.DataFrame({
            patient_col: processed_patients,
            "Kendall_W": pkendall_w,
            "dominant_consistent": pdom_ok,
            "fullrank_consistent": pfullrank_ok,
            "profile_corr": p_profile_corr_vals,
            "mean_abs_drift_pct": p_mean_abs_drift_vals,
        })

        print(f"\nWorkflow reproducibility — PATIENT level (n={n_pat} patients, all 4 batches):")
        for _, row in patient_metrics_df.head(len(pop_cols)).iterrows():
            print(
                f"  {row['Population']:<44}: "
                f"{row['Mean range (%)']:.2f} ± {row['Std range (%)']:.2f} %"
            )
        print(f"\n  Kendall's W:                   {pmean_W:.3f} ± {pstd_W:.3f}")
        print(f"  Profile correlation (r):       {p_mean_profile_corr:.3f} ± {p_std_profile_corr:.3f}")
        print(f"  Mean abs. drift (± pts):       {p_mean_drift:.2f} ± {p_std_drift:.2f}")
        print(f"  Same dominant population:      {ppct_dom:.1f}% of patients")
        print(f"  Identical full ranking:        {ppct_full:.1f}% of patients")

    # ── Save all metrics to a single XLSX (core and patient sheets) ───────────
    metrics_path = os.path.join(outdir, "workflow_reproducibility_metrics.xlsx")
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Core level", index=False)
        core_concordance_df.to_excel(writer, sheet_name="Core concordance", index=False)
        if not patient_metrics_df.empty:
            patient_metrics_df.to_excel(writer, sheet_name="Patient level", index=False)
            patient_concordance_df.to_excel(
                writer, sheet_name="Patient concordance", index=False
            )

    print(f"\n  Metrics saved → {metrics_path} (Excel, sheets: Core level / Patient level)\n")

    # ── Plain-language summary of what each reproducibility metric means ──────
    summary_path = os.path.join(outdir, "workflow_reproducibility_summary.md")
    with open(summary_path, "w") as f:
        f.write(
            "# Workflow reproducibility — metric definitions and this run's results\n\n"
            "All metrics compare the same control TMA, re-processed from scratch "
            f"(scanning, staining, classification) in each of the {len(batches)} "
            "batches (" + ", ".join(batches) + "). Each metric is computed per core "
            "and, separately, per patient (mean of that patient's cores within each "
            "batch, i.e. pseudo-bulk).\n\n"
            "- **Kendall's W** (0-1): rank-only concordance. For each batch, rank "
            "the 4 populations by abundance (1 = most abundant); W measures how "
            "consistent that ranking is across batches. 1 = identical ranking in "
            "every batch; 0 = rankings unrelated. Blind to magnitude: a batch with "
            "60% GATA6+CK17- and one with 30% still score W=1 if the ranking of the "
            "4 populations doesn't change.\n"
            "- **Same dominant population** / **Identical full ranking** (%): share "
            "of cores/patients where, respectively, only the top-ranked population "
            "or the entire 4-population ranking is identical in all batches. Coarser, "
            "easier-to-report versions of Kendall's W.\n"
            "- **Profile correlation (r)**: mean Pearson correlation between every "
            "pair of batches' 4-population vectors, for the same core/patient. "
            "Unlike W, it is sensitive to relative magnitude, not just order — two "
            "batches with the same ranking but very different proportions score "
            "lower here than in W.\n"
            "- **Mean absolute drift (± percentage points)**: for each population, "
            "the SD of its % across the batches; averaged over the 4 populations. "
            "This is the literal \"this core/patient's population shares move by "
            "about ± X points from batch to batch\" figure.\n\n"
            "## This run's results\n\n"
            f"Core level (n={n_complete} cores present in all {len(batches)} batches):\n"
            f"- Kendall's W: {mean_W:.3f} ± {std_W:.3f}\n"
            f"- Same dominant population: {pct_dom:.1f}%\n"
            f"- Identical full ranking: {pct_full:.1f}%\n"
            f"- Profile correlation (r): {mean_profile_corr:.3f} ± {std_profile_corr:.3f}\n"
            f"- Mean absolute drift: ± {mean_drift:.2f} ± {std_drift:.2f} percentage points\n\n"
        )
        if not patient_metrics_df.empty:
            f.write(
                f"Patient level (n={n_pat} patients with cores in all {len(batches)} batches):\n"
                f"- Kendall's W: {pmean_W:.3f} ± {pstd_W:.3f}\n"
                f"- Same dominant population: {ppct_dom:.1f}%\n"
                f"- Identical full ranking: {ppct_full:.1f}%\n"
                f"- Profile correlation (r): {p_mean_profile_corr:.3f} ± {p_std_profile_corr:.3f}\n"
                f"- Mean absolute drift: ± {p_mean_drift:.2f} ± {p_std_drift:.2f} percentage points\n"
            )
    print(f"  Summary saved → {summary_path}\n")

    return core_endpoints


def cross_reference_workflow_heterogeneity(
    workflow_outdir: str,
    outdir: str,
) -> pd.DataFrame:
    """Cross-reference workflow-discordant cores with heterogeneity metrics.

    Loads the per-core concordance table saved by run_workflow_reproducibility and the
    per-batch validation CSVs from the WORKFLOW analysis (same cells used for concordance),
    then compares Covariance area, Spatial variance (relative), Heterogeneity_local, and
    Heterogeneity_global between concordant and discordant cores using Mann-Whitney U tests.

    Numeric metrics are averaged across the 4 workflow batches per core before comparison.
    Categorical labels (Heterogeneity_local/global) are taken as the most frequent label
    across batches (mode). Using the workflow CSVs — rather than the main-analysis CSVs —
    ensures the heterogeneity metrics are computed on exactly the same cells and
    normalization as the concordance analysis.

    Heterogeneity metrics definitions (computed by validation_calculations):
      - Covariance area: sqrt(det(Cov(GATA6_raw, CK17_raw))) across all cells in the
        core. Geometrically, this is the semi-axes product of the 1-SD ellipse in the
        GATA6/CK17 raw intensity plane — large values = cells spread across multiple
        phenotypic zones.
      - Spatial variance (relative): mean pairwise Euclidean distance between all cells
        in the 2D (GATA6_raw, CK17_raw) space, divided by sqrt(255^2 + 255^2) to
        normalise to [0, 1]. Captures average cell-to-cell dissimilarity regardless of
        distributional shape.
      - Heterogeneity_local: IQR-based label (Low/Medium/High/VeryHigh) using Q1, Q3,
        IQR of Covariance area computed within each TMA batch separately.
      - Heterogeneity_global: same label using Q1, Q3, IQR pooled across all batches
        in the workflow dataset.

    Args:
        workflow_outdir: Directory where run_workflow_reproducibility saved its outputs
            (must contain workflow_core_concordance.csv and a validation/ subfolder).
        outdir: Directory where cross-reference results (XLSX) will be saved.

    Returns:
        pd.DataFrame: Per-core merged table (concordance + heterogeneity metrics).
    """
    from scipy.stats import mannwhitneyu

    concordance_path = os.path.join(workflow_outdir, "workflow_core_concordance.csv")
    if not os.path.isfile(concordance_path):
        raise FileNotFoundError(
            f"Per-core concordance file not found: {concordance_path}. "
            "Run run_workflow_reproducibility first."
        )
    concordance_df = pd.read_csv(concordance_path)

    # Load per-batch validation CSVs from the workflow reproducibility analysis
    # (these contain heterogeneity metrics computed on the same cells as the concordance)
    workflow_csv_dir = os.path.join(workflow_outdir, "validation")
    csv_files = glob.glob(os.path.join(workflow_csv_dir, "*", "*_validation.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No workflow validation CSVs found under {workflow_csv_dir}. "
            "Run run_workflow_reproducibility first."
        )

    hetero_numeric = ["Covariance area", "Spatial variance (relative)"]
    hetero_categ = ["Heterogeneity_local", "Heterogeneity_global"]
    needed = ["Core"] + hetero_numeric + hetero_categ

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        available = [c for c in needed if c in df.columns]
        if "Core" in available:
            frames.append(df[available])

    if not frames:
        raise ValueError("No heterogeneity columns found in workflow validation CSVs.")

    hetero_df = pd.concat(frames, ignore_index=True)

    # Average numeric metrics across the 4 batches per core
    available_numeric = [c for c in hetero_numeric if c in hetero_df.columns]
    core_hetero = hetero_df.groupby("Core")[available_numeric].mean().reset_index()

    # Most frequent categorical label per core across batches
    for cat_col in hetero_categ:
        if cat_col in hetero_df.columns:
            mode_series = (
                hetero_df.groupby("Core")[cat_col]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
                .reset_index()
            )
            core_hetero = core_hetero.merge(mode_series, on="Core", how="left")

    merged = concordance_df.merge(core_hetero, on="Core", how="inner")

    concordant = merged[merged["dominant_consistent"] == True]
    discordant = merged[merged["dominant_consistent"] == False]

    print(
        f"\nCross-reference: workflow concordance vs heterogeneity (workflow validation CSVs)\n"
        f"  Concordant cores: n={len(concordant)} | Discordant cores: n={len(discordant)}\n"
    )

    mw_rows = []
    for col in available_numeric:
        c_vals = concordant[col].dropna()
        d_vals = discordant[col].dropna()
        stat, pval = mannwhitneyu(d_vals, c_vals, alternative="greater")
        mw_rows.append({
            "Metric": col,
            "Concordant mean ± std": f"{c_vals.mean():.3f} ± {c_vals.std():.3f}",
            "Discordant mean ± std": f"{d_vals.mean():.3f} ± {d_vals.std():.3f}",
            "Mann-Whitney U": round(float(stat), 1),
            "p-value (one-sided)": round(float(pval), 4),
        })
        print(
            f"  {col}:\n"
            f"    Concordant: {c_vals.mean():.3f} ± {c_vals.std():.3f}\n"
            f"    Discordant: {d_vals.mean():.3f} ± {d_vals.std():.3f}\n"
            f"    p-value (discordant > concordant): {pval:.4f}\n"
        )

    for cat_col in hetero_categ:
        if cat_col in merged.columns:
            print(f"  {cat_col} distribution:")
            print(f"    Concordant: {concordant[cat_col].value_counts().to_dict()}")
            print(f"    Discordant: {discordant[cat_col].value_counts().to_dict()}\n")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "workflow_heterogeneity_crossref.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(mw_rows).to_excel(writer, sheet_name="Mann-Whitney", index=False)
        merged.to_excel(writer, sheet_name="Per-core", index=False)

    print(f"  Cross-reference results saved → {out_path}\n")
    return merged