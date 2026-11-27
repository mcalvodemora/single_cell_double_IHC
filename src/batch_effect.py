# batch_effect.py
"""
Batch-effect diagnostics for the double-IHC TMA cohort, borrowing common
large-scale omics QC practices (PCA, hierarchical clustering + heatmap, PVCA)
and applying them to image-derived intensity/classification features instead
of expression data.

Two diagnostic layers:
    PRE  - raw cell-level intensities aggregated to pseudo-bulk per core,
           checking for systematic shifts before any thresholding.
    POST - core-level endpoints from validation_calculations (percent
           positive, H-scores, heterogeneity metrics), checking whether
           batch structure survives into the final results.

For each layer and each batch definition (e.g. TMA, hospital), the module
produces a PCA scatter, a clustered heatmap with a batch color strip,
per-feature boxplots, and a PVCA-style variance decomposition.

Each patient in this cohort belongs to a single TMA and a single hospital, so
patient and batch are fully confounded and the full sample+batch+interaction
PVCA model isn't identifiable. The batch-only version used here just asks how
much of the PCA variance lines up with batch, and calls the rest residual.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

import statsmodels.api as sm
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Defaults / configuration
# -----------------------------------------------------------------------------

# Per-cell staining intensity columns used for PRE-threshold pseudo-bulk
# summaries: GATA6 read from the nucleus ("DAB Intensity - Nuclei") and CK17
# from the cytoplasm ("Mag Intensity - Cytoplasm").
DEFAULT_PRE_INTENSITY_COLS: Dict[str, str] = {
    "GATA6": "DAB Intensity - Nuclei",
    "CK17": "Mag Intensity - Cytoplasm",
}

# Fallback names seen in older exports or slightly different pipelines.
# If your export changes, add alternative spellings here.
PRE_INTENSITY_FALLBACKS: Dict[str, List[str]] = {
    "DAB Intensity - Nuclei": ["DAB Intensity - Nucleus"],
    "Mag Intensity - Cytoplasm": ["Mag Intensity - Cytoplasm"],
}

# PRE aggregation statistics to compute per marker, per core.
# Notes:
# - Percentiles + IQR are robust to outliers and non-Gaussian intensity distributions
# - Mean/std are also useful when shifts are approximately additive/multiplicative
DEFAULT_PRE_AGGS: List[str] = ["p10", "p50", "p90", "iqr", "mean", "std"]

# POST features expected from validation_calculations core-level output.
DEFAULT_POST_FEATURES: List[str] = [
    "% positive cell GATA6",
    "% positive cell CK17",
    "% GATA6+CK17+",
    "% GATA6+CK17-",
    "% GATA6-CK17+",
    "% GATA6-CK17-",
    "H-Score GATA6",
    "H-Score CK17",
    "Shannon entropy (bits)",
    "Shannon entropy (normalized)",
    "Spatial variance",
    "Spatial variance (relative)",
    "Heterogeneity_local",
    "Heterogeneity_global",
]

# If heterogeneity is stored as strings, we encode it to ordinal ranks for PCA/HCA/PVCA.
HETEROGENEITY_ORDER: List[str] = ["Low", "Medium", "High", "VeryHigh"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


# Mapping from patient ID prefix to standardised hospital code.
# Order matters: longer/more-specific prefixes must appear before shorter ones
# (e.g. "HUPrinc" before "HUP") so that startswith() matches correctly.
HOSPITAL_ID_PREFIX_MAP: Dict[str, str] = {
    "HUPrinc": "HPRIN",   # Hospital Universitario Príncesa (long form)
    "HUP":     "HPRIN",   # Hospital Universitario Princesa (short form)
    "H12O":    "H12O",    # Hospital Universitario 12 de Octubre
    "HUIS":    "HUIS",    # Hospital Universitario Infanta Sofía
    "HULP":    "HULP",    # Hospital Universitario La Paz
    "HRyC":    "HRYC",    # Hospital Universitario Ramón y Cajal
    "HPH":     "HUPH",    # Hospital Universitario Puerta de Hierro
    "HCB":     "HCB",     # Hospital Clínic de Barcelona
    "HGM":     "HGM",     # Hospital Gregorio Marañón
}


def add_hospital_from_id(
    df: pd.DataFrame,
    id_col: str = "ID",
    out_col: str = "Hospital",
    tma_col: str = "TMA",
    prefix_map: Dict[str, str] = HOSPITAL_ID_PREFIX_MAP,
) -> pd.DataFrame:
    """
    Derive a hospital identifier from the patient ID prefix.

    Iterates over ``prefix_map`` in insertion order and assigns the mapped
    hospital code to each row whose ID starts with that prefix.  IDs that
    are NaN or whose prefix is not in the map fall back to splitting the TMA
    name on ``'-'`` (first token).

    Examples::

        "HULP74"    -> "HULP"
        "H12O_001"  -> "H12O"
        "HUIS_ 004" -> "HUIS"   # internal space preserved; prefix still matches
        "HUPrinc1"  -> "HPRIN"
        "HUP2"      -> "HPRIN"
        "HPH1"      -> "HUPH"
        "HRyC1"     -> "HRYC"
        "HCB-01"    -> "HCB"

    Args:
        df: Input dataframe.
        id_col: Column containing patient/sample IDs (e.g. "HULP74").
        out_col: Name of the output hospital column.
        tma_col: Column used as fallback when ID is missing or unrecognised.
        prefix_map: Mapping of ID prefix -> hospital code.
            Defaults to :data:`HOSPITAL_ID_PREFIX_MAP`.

    Returns:
        Copy of ``df`` with an added ``out_col`` column.
    """
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not found in dataframe.")
    if tma_col not in df.columns:
        raise ValueError(f"'{tma_col}' not found in dataframe.")

    out = df.copy()

    def _map_id(val: object) -> str | None:
        if pd.isna(val):
            return None
        s = str(val)
        for prefix, code in prefix_map.items():
            if s.startswith(prefix):
                return code
        return None  # unrecognised prefix -> TMA fallback

    hospital_from_id = out[id_col].map(_map_id)
    tma_fallback = (
        out[tma_col].astype(str).str.strip().str.split("-", n=1, expand=True)[0]
    )
    out[out_col] = hospital_from_id.where(hospital_from_id.notna(), tma_fallback)
    return out


def _safe_mkdir(path: str) -> None:
    """Create directory if it does not exist (no error if already present)."""
    os.makedirs(path, exist_ok=True)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to reduce mismatches due to whitespace.
    This is helpful when reading CSV/TSV exports with inconsistent spacing.
    """
    out = df.copy()
    out.columns = (
        out.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return out


def _resolve_intensity_col(df: pd.DataFrame, preferred: str) -> str:
    """
    Return the first matching intensity column, allowing fallbacks.

    Args:
        df: DataFrame with columns to search.
        preferred: Preferred column name.

    Returns:
        Column name to use.

    Raises:
        ValueError if neither preferred nor any fallback exists.
    """
    if preferred in df.columns:
        return preferred
    for alt in PRE_INTENSITY_FALLBACKS.get(preferred, []):
        if alt in df.columns:
            return alt
    raise ValueError(
        f"Could not find intensity column '{preferred}' (nor fallbacks) in dataframe."
    )


def _encode_heterogeneity_labels(
    s: pd.Series,
    order: List[str] = HETEROGENEITY_ORDER,
) -> pd.Series:
    """
    Encode ordinal heterogeneity labels as numeric ranks.

    This enables PCA/HCA/PVCA to run when heterogeneity is stored as categories.
    If values do not match the expected vocabulary, they become NaN.

    Args:
        s: Input series with labels.
        order: Ordered list from lowest to highest.

    Returns:
        Numeric series (float), with NaNs for unmapped values.
    """
    if s is None:
        return s
    s2 = s.astype(str).str.strip()
    mapping = {lab: i for i, lab in enumerate(order)}
    out = s2.map(mapping)
    return pd.to_numeric(out, errors="coerce")


def _zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise z-score (mean=0, std=1), robust to NaNs.
    Any constant column becomes all zeros.
    Remaining NaNs after standardization are filled with 0.
    """
    Z = X.copy()
    for c in Z.columns:
        v = pd.to_numeric(Z[c], errors="coerce")
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if sd == 0 or np.isnan(sd):
            Z[c] = 0.0
        else:
            Z[c] = (v - mu) / sd
    return Z.fillna(0.0)


def _get_palette(n: int) -> List:
    """
    Qualitative palette for categorical annotations.

    Uses matplotlib 'tab20'. If more than 20 groups exist, colors repeat.
    """
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def _clean_labels(s: pd.Series) -> pd.Series:
    """Standardize labels: cast to str, fill NA, and strip whitespace."""
    return s.astype(str).fillna("NA").str.strip()


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce a series to numeric, handling commas as decimal separators.
    Useful for exports that use '1,23' instead of '1.23'.
    """
    if s.dtype == object:
        s2 = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _ensure_required_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """
    Raise a clear error if required columns are missing.

    Args:
        df: DataFrame to validate.
        cols: Required column names.
        name: Label used in error message.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")

def print_tma_scale_summary(
    df: pd.DataFrame,
    tma_col: str,
    intensity_cols: Dict[str, str],
    label: str = "RAW",
    scale_max: float | None = None,
    outdir: str | None = None,
) -> pd.DataFrame:
    """
    Print per-TMA mean and std for each marker, and return them as a long DataFrame.
    Designed to demonstrate inter-TMA scale differences.

    All printed output is unchanged from the original version. `scale_max` and
    `outdir` are purely additive:
      - scale_max: theoretical max of this layer's scale (e.g. 255 for 8-bit raw
        intensities, 1 for the q01-q99 normalized layer). When given, also prints
        the mean/std ranges on a 0-1 relative scale, and adds mean_rel/std_rel
        columns to the returned DataFrame, so different layers become comparable.
      - outdir: if given, save the returned long-format summary as
        f"tma_scale_summary_{label.lower()}.csv" in this directory.
    """

    print(f"\n========== TMA SCALE SUMMARY ({label}) ==========\n")

    records = []

    for marker, col in intensity_cols.items():

        if col not in df.columns:
            print(f"[{marker}] Column '{col}' not found.")
            continue

        summary = (
            df.groupby(tma_col)[col]
            .agg(["mean", "std"])
            .reset_index()
        )

        print(f"\nMarker: {marker}")
        print(summary)

        mean_range = summary["mean"].max() - summary["mean"].min()
        std_range = summary["std"].max() - summary["std"].min()

        print(f"\nRange of TMA means: {mean_range:.4f}")
        print(f"Range of TMA stds : {std_range:.4f}")
        if scale_max is not None:
            print(f"Range of TMA means (relative to {scale_max}): {mean_range / scale_max:.4f}")
            print(f"Range of TMA stds  (relative to {scale_max}): {std_range / scale_max:.4f}")
        print("-" * 50)

        for _, row in summary.iterrows():
            record = {
                "label": label,
                "marker": marker,
                "TMA": row[tma_col],
                "mean": row["mean"],
                "std": row["std"],
            }
            if scale_max is not None:
                record["mean_rel"] = row["mean"] / scale_max
                record["std_rel"] = row["std"] / scale_max
            records.append(record)

    print("\n==============================================\n")

    long_df = pd.DataFrame(records)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, f"tma_scale_summary_{label.lower()}.csv")
        long_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return long_df


def summarize_scale_change(
    scale_raw: pd.DataFrame,
    scale_norm: pd.DataFrame,
    outdir: str,
) -> pd.DataFrame:
    """
    Compare inter-TMA scale spread between the raw (0-255) and normalized (0-1)
    intensity layers, on a common 0-1 relative scale.

    Takes the long-format summaries returned by print_tma_scale_summary() for the
    RAW (scale_max=255) and NORMALIZED (scale_max=1) layers and builds one row per
    marker with the range (max-min across TMAs) of the per-TMA mean and std, in raw
    units and on the 0-1 relative scale. Saves "tma_scale_comparison.csv" in outdir.
    """
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for marker in scale_raw["marker"].unique():
        raw_m = scale_raw[scale_raw["marker"] == marker]
        norm_m = scale_norm[scale_norm["marker"] == marker]

        range_mean_raw = raw_m["mean"].max() - raw_m["mean"].min()
        range_std_raw = raw_m["std"].max() - raw_m["std"].min()
        range_mean_norm = norm_m["mean"].max() - norm_m["mean"].min()
        range_std_norm = norm_m["std"].max() - norm_m["std"].min()

        rows.append({
            "marker": marker,
            "range_mean_raw": range_mean_raw,
            "range_mean_raw_rel": range_mean_raw / 255,
            "range_mean_norm": range_mean_norm,
            "range_std_raw": range_std_raw,
            "range_std_raw_rel": range_std_raw / 255,
            "range_std_norm": range_std_norm,
        })

    comparison = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "tma_scale_comparison.csv")
    comparison.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return comparison

# -----------------------------------------------------------------------------
# PRE-threshold (cell-level -> pseudo-bulk per core)
# -----------------------------------------------------------------------------


def build_prethreshold_core_features(
    raw_df: pd.DataFrame,
    intensity_cols: Dict[str, str] = DEFAULT_PRE_INTENSITY_COLS,
    aggs: List[str] = DEFAULT_PRE_AGGS,
    group_cols: Tuple[str, str, str, str, str] = (
        "TMA",
        "Hospital",
        "Core",
        "Patient",
        "ID",
    ),
) -> pd.DataFrame:
    """
    Aggregate cell-level intensities into a pseudo-bulk per-core table.

    One output row corresponds to one "core sample" (as defined by group_cols).
    For each marker (e.g., GATA6, CK17), summary statistics are computed
    over all cells in that core.

    Why percentiles/IQR?
        Intensity distributions are often skewed/heavy-tailed. Percentiles and IQR are
        robust summaries that better capture distribution shifts than mean alone.

    Args:
        raw_df: Cell-level dataframe (one row per cell).
        intensity_cols: Mapping marker -> preferred column name in raw_df.
        aggs: Subset of {"p10","p50","p90","iqr","mean","std"}.
        group_cols: Columns defining the core sample and carrying metadata.

    Returns:
        Core-level DataFrame with:
            - group_cols
            - n_cells
            - one column per marker/stat, e.g., 'GATA6_p50'
    """
    df = _normalize_cols(raw_df)

    # Resolve intensity channels (preferred + fallbacks).
    resolved: Dict[str, str] = {}
    for marker, col in intensity_cols.items():
        resolved[marker] = _resolve_intensity_col(df, col)

    _ensure_required_cols(df, list(group_cols), "raw_df (PRE)")

    # Coerce intensity columns to numeric.
    for col in resolved.values():
        df[col] = _coerce_numeric_series(df[col])

    def _agg_marker(x: pd.Series) -> Dict[str, float]:
        """Compute distribution summaries for a single marker within one core."""
        x = _coerce_numeric_series(x).dropna()
        out: Dict[str, float] = {}

        if len(x) == 0:
            for k in aggs:
                out[k] = np.nan
            return out

        if "p10" in aggs:
            out["p10"] = float(np.percentile(x, 10))
        if "p50" in aggs:
            out["p50"] = float(np.percentile(x, 50))
        if "p90" in aggs:
            out["p90"] = float(np.percentile(x, 90))
        if "iqr" in aggs:
            out["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
        if "mean" in aggs:
            out["mean"] = float(np.mean(x))
        if "std" in aggs:
            # ddof=1 -> sample std; if only 1 value, std is 0 by definition here.
            out["std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

        return out

    rows: List[Dict[str, object]] = []
    g = df.groupby(list(group_cols), dropna=False)

    for keys, sub in g:
        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        row["n_cells"] = int(sub.shape[0])

        for marker, col in resolved.items():
            agg = _agg_marker(sub[col])
            for stat, val in agg.items():
                row[f"{marker}_{stat}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# POST-threshold (validation results -> core endpoints)
# -----------------------------------------------------------------------------


def build_postthreshold_core_features(
    validation_df: pd.DataFrame,
    features: List[str] = DEFAULT_POST_FEATURES,
    keep_cols: Tuple[str, str, str, str, str] = (
        "TMA",
        "Hospital",
        "Core",
        "Patient",
        "ID",
    ),
) -> pd.DataFrame:
    """
    Select core-level endpoints from validation_calculations output.

    Notes:
        - Heterogeneity_{local,global} may be categorical; we encode it as ordinal ranks.
        - All other features are coerced to numeric.

    Args:
        validation_df: Core-level dataframe (one row per core).
        features: Feature columns to select.
        keep_cols: Metadata columns to retain.

    Returns:
        DataFrame with keep_cols + features (numeric where applicable).
    """
    df = _normalize_cols(validation_df)

    _ensure_required_cols(df, list(keep_cols), "validation_df (POST)")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing post-threshold features in validation_df: {missing}")

    out = df[list(keep_cols) + features].copy()

    # Encode heterogeneity labels if still strings.
    for hc in ["Heterogeneity_local", "Heterogeneity_global"]:
        if hc in out.columns:
            out[hc] = _encode_heterogeneity_labels(out[hc])

    # Coerce numeric for the rest.
    for c in features:
        if c in ["Heterogeneity_local", "Heterogeneity_global"]:
            continue
        out[c] = _coerce_numeric_series(out[c])

    return out


# -----------------------------------------------------------------------------
# PCA (PC1 vs PC2) colored by batch
# -----------------------------------------------------------------------------


def plot_pca(
    core_df: pd.DataFrame,
    feature_cols: List[str],
    batch_col: str,
    outpath: str,
    title: str,
    min_samples: int = 6,
    dpi: int = 200,
) -> None:
    """
    PCA scatter plot (PC1 vs PC2) with points colored by batch.

    Implementation choices:
        - NaNs imputed with feature means (simple and stable for visualization QC)
        - StandardScaler used to avoid scale dominance by any single feature

    Args:
        core_df: Core-level dataframe.
        feature_cols: Numeric feature columns for PCA.
        batch_col: Column used for coloring points.
        outpath: Output SVG path.
        title: Plot title.
        min_samples: Minimum number of samples to run PCA.
        dpi: Image resolution.
    """
    _safe_mkdir(os.path.dirname(outpath))

    df = core_df.copy()
    if batch_col not in df.columns:
        raise ValueError(f"batch_col '{batch_col}' not in dataframe.")

    X = df[feature_cols].apply(_coerce_numeric_series)
    y = _clean_labels(df[batch_col])

    # Keep rows with at least one non-null feature.
    keep = X.notna().sum(axis=1) > 0
    X = X.loc[keep].copy()
    y = y.loc[keep].copy()

    if X.shape[0] < min_samples:
        print(f"[PCA] Skipping '{title}': only {X.shape[0]} samples.")
        return

    # Mean-impute remaining NaNs.
    X = X.fillna(X.mean())

    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)

    batches = sorted(y.unique())
    colors = _get_palette(len(batches))
    color_map = {b: colors[i] for i, b in enumerate(batches)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for b in batches:
        m = (y == b).values
        ax.scatter(
            pcs[m, 0],
            pcs[m, 1],
            s=60,
            alpha=0.85,
            label=b,
            color=color_map[b],
            edgecolor="white",
            linewidth=0.6,
        )

    evr = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    ax.legend(title=batch_col, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# HCA + heatmap (row + column clustering) with batch strip
# -----------------------------------------------------------------------------


def plot_hca_heatmap(
    core_df: pd.DataFrame,
    feature_cols: List[str],
    batch_col: str,
    outpath: str,
    title: str,
    metric: str = "euclidean",
    method: str = "average",
    dpi: int = 200,
) -> None:
    """
    HCA + heatmap with:
        - Row dendrogram (samples)
        - Column dendrogram (features)
        - Batch color strip on top
        - Legend placed outside (no overlap)

    Notes:
        - Heatmap values are z-scored per feature to compare patterns across features.
        - Row ordering is obtained from hierarchical clustering leaves.
        - Column ordering is also clustered (like typical omics QC heatmaps).

    Args:
        core_df: Core-level dataframe.
        feature_cols: Feature columns to visualize.
        batch_col: Batch annotation column (used in top strip and legend).
        outpath: Output SVG path.
        title: Plot title.
        metric: Distance metric for clustering (e.g., 'euclidean').
        method: Linkage method (e.g., 'average', 'ward' requires euclidean).
        dpi: Image resolution.
    """
    _safe_mkdir(os.path.dirname(outpath))

    df = core_df.copy()
    if batch_col not in df.columns:
        raise ValueError(f"batch_col '{batch_col}' not in dataframe.")

    X = df[feature_cols].apply(_coerce_numeric_series)
    y = _clean_labels(df[batch_col])

    keep = X.notna().sum(axis=1) > 0
    X = X.loc[keep].fillna(X.mean())
    y = y.loc[keep]

    Z = _zscore_df(X)

    # --- Row clustering (samples) ---
    row_dist = pdist(Z.values, metric=metric)
    row_link = linkage(row_dist, method=method)
    row_order = leaves_list(row_link)

    # --- Column clustering (features) ---
    col_dist = pdist(Z.values.T, metric=metric)
    col_link = linkage(col_dist, method=method)
    col_order = leaves_list(col_link)

    Z_ord = Z.iloc[row_order, :].iloc[:, col_order]
    y_ord = y.iloc[row_order]

    batches = sorted(y_ord.unique())
    colors = _get_palette(len(batches))
    color_map = {b: colors[i] for i, b in enumerate(batches)}
    batch_rgba = np.array([color_map[b] for b in y_ord])

    # Layout:
    #   col dendrogram (top)
    #   batch strip (under col dendrogram)
    #   row dendrogram (left) + heatmap (right)
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.4, 0.35, 6.0],
        width_ratios=[2.2, 8.5],
        hspace=0.05,
        wspace=0.05,
    )

    # Column dendrogram (top-right)
    ax_col = fig.add_subplot(gs[0, 1])
    dendrogram(col_link, ax=ax_col, no_labels=True, color_threshold=None)
    ax_col.axis("off")

    # Batch strip (middle-right)
    ax_strip = fig.add_subplot(gs[1, 1])
    ax_strip.imshow(batch_rgba.reshape(-1, 1, 4).transpose(1, 0, 2), aspect="auto")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    # Use pad to avoid overlap with the top dendrogram.
    ax_strip.set_title(title, pad=18)

    # Row dendrogram (bottom-left)
    ax_row = fig.add_subplot(gs[2, 0])
    dendrogram(
        row_link, orientation="left", ax=ax_row, no_labels=True, color_threshold=None
    )
    ax_row.invert_yaxis()
    ax_row.axis("off")

    # Heatmap (bottom-right)
    ax_hm = fig.add_subplot(gs[2, 1])
    im = ax_hm.imshow(Z_ord.values, aspect="auto", interpolation="nearest")
    ax_hm.set_yticks([])
    ax_hm.set_xticks(range(len(Z_ord.columns)))
    ax_hm.set_xticklabels(Z_ord.columns, rotation=45, ha="right", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.03, pad=0.02)
    cbar.set_label("z-score")

    # Legend outside on the right
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color_map[b],
            markersize=10,
            label=b,
        )
        for b in batches
    ]
    ax_hm.legend(
        handles=handles,
        title=batch_col,
        bbox_to_anchor=(1.18, 1),
        loc="upper left",
        frameon=True,
    )

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Feature plots across batches
# -----------------------------------------------------------------------------


def plot_feature_barplots(
    core_df: pd.DataFrame,
    feature_cols: List[str],
    batch_col: str,
    outdir: str,
    title_prefix: str,
    dpi: int = 200,
    max_levels: int = 30,
) -> None:
    """
    For each feature:
        - Boxplot per batch level (without fliers)
        - Jittered scatter overlay (shows dispersion + outliers)

    Args:
        core_df: Core-level dataframe.
        feature_cols: Features to plot.
        batch_col: Batch column defining groups.
        outdir: Output directory for SVGs.
        title_prefix: Prefix shown in plot title.
        dpi: Image resolution.
        max_levels: Soft warning threshold for number of batch levels (crowded labels).
    """
    _safe_mkdir(outdir)

    df = core_df.copy()
    if batch_col not in df.columns:
        raise ValueError(f"batch_col '{batch_col}' not in dataframe.")

    y = _clean_labels(df[batch_col])
    levels = sorted(y.unique())

    if len(levels) > max_levels:
        print(
            f"[Barplots] Warning: {batch_col} has {len(levels)} levels. Plots may be crowded."
        )

    rng = np.random.default_rng(0)

    for feat in feature_cols:
        x = _coerce_numeric_series(df[feat])
        m = x.notna() & y.notna()
        if int(m.sum()) < 3:
            continue

        x2 = x[m]
        y2 = y[m]

        data_by_level = [x2[y2 == lev].values for lev in levels]

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(levels)), 4.8))
        ax.boxplot(data_by_level, labels=levels, showfliers=False)

        # Jittered points per group
        for i, lev in enumerate(levels, start=1):
            vals = x2[y2 == lev].values
            if len(vals) == 0:
                continue
            jitter = rng.normal(0, 0.05, size=len(vals))
            ax.scatter(i + jitter, vals, s=22, alpha=0.75)

        ax.set_title(f"{title_prefix} | {feat}")
        ax.set_xlabel(batch_col)
        ax.set_ylabel(feat)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=45, ha="right")

        outpath = os.path.join(
            outdir, f"{feat.replace('/','_').replace(':','_')}_by_{batch_col}.svg"
        )
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# PVCA-style variance partitioning (batch-only, due to confounding)
# -----------------------------------------------------------------------------


def pvca_like_batch_only(
    core_df: pd.DataFrame,
    feature_cols: List[str],
    batch_col: str,
    pct_threshold: float = 0.60,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    PVCA-style variance partitioning (batch-only version).

    Motivation:
        Classical PVCA partitions variance into:
            - sample (biological)
            - batch (technical)
            - mixed (interaction)
            - residual
        However, if sample and batch are perfectly confounded (e.g., each patient appears in
        exactly one batch), the full model is not identifiable.

    This simplified approach estimates the fraction of variance in the PCA space that is
    explained by batch, treating everything else as residual.

    Steps:
        1) Standardize features (z-score), then run PCA
        2) Keep the smallest number of PCs whose cumulative explained variance >= pct_threshold
        3) For each retained PC score, fit one-way ANOVA via OLS:
               pc ~ C(batch)
           Partition variance into:
               - batch (between-groups)
               - residual (within-groups)
        4) Average variance fractions across PCs, weighted by each PC's explained variance

    Args:
        core_df: Core-level dataframe.
        feature_cols: Numeric features to use.
        batch_col: Batch column to test (e.g., TMA, Hospital).
        pct_threshold: Target cumulative explained variance for retained PCs (commonly 0.60).
        min_samples: Minimum number of cores required to run.

    Returns:
        DataFrame with columns:
            - Source: ["batch", "residual"]
            - Value: weighted average proportion of variance (sums ~1)
    """
    df = core_df.copy()
    if batch_col not in df.columns:
        raise ValueError(f"batch_col '{batch_col}' not in dataframe.")

    X = df[feature_cols].apply(_coerce_numeric_series)
    batch = _clean_labels(df[batch_col])

    keep = X.notna().sum(axis=1) > 0
    X = X.loc[keep].copy()
    batch = batch.loc[keep].copy()

    if X.shape[0] < min_samples:
        print(f"[PVCA] Skipping: only {X.shape[0]} samples.")
        return pd.DataFrame(
            {"Source": ["batch", "residual"], "Value": [np.nan, np.nan]}
        )

    # Mean imputation + standardization
    X = X.fillna(X.mean())
    Z = StandardScaler().fit_transform(X.values)

    # PCA over standardized features
    pca = PCA(n_components=min(Z.shape[0], Z.shape[1]), random_state=0)
    scores = pca.fit_transform(Z)
    evr = pca.explained_variance_ratio_

    # Determine how many PCs to keep (default: 60% cumulative variance)
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, pct_threshold) + 1)
    k = max(1, min(k, scores.shape[1]))

    batch_var = 0.0
    resid_var = 0.0
    wsum = float(np.sum(evr[:k]))

    model_df = pd.DataFrame({"batch": batch.values})

    for i in range(k):
        w = float(evr[i] / wsum) if wsum > 0 else 0.0
        model_df["pc"] = scores[:, i]

        # One-way ANOVA implemented via OLS + anova_lm
        try:
            fit = smf.ols("pc ~ C(batch)", data=model_df).fit()
            an = sm.stats.anova_lm(fit, typ=2)

            ss_batch = float(an["sum_sq"].get("C(batch)", 0.0))
            ss_resid = float(fit.ssr)
            ss_total = ss_batch + ss_resid

            if ss_total > 0:
                batch_var += w * (ss_batch / ss_total)
                resid_var += w * (ss_resid / ss_total)
        except Exception:
            # If a PC cannot be modeled (rare; typically due to degenerate design), skip it.
            continue

    return pd.DataFrame(
        {"Source": ["batch", "residual"], "Value": [batch_var, resid_var]}
    )


def plot_pvca_bar(
    pvca_df: pd.DataFrame,
    outpath: str,
    title: str,
    dpi: int = 200,
) -> None:
    """
    Barplot for PVCA-style variance partition results.

    Args:
        pvca_df: DataFrame with columns ["Source", "Value"].
        outpath: Output SVG path.
        title: Title shown on the plot.
        dpi: Image resolution.
    """
    _safe_mkdir(os.path.dirname(outpath))

    d = pvca_df.copy()
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(d["Source"], d["Value"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of variance")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# High-level runner
# -----------------------------------------------------------------------------

def run_batch_effect_diagnosis(
    mode: str,
    raw_df: pd.DataFrame | None,
    validation_results: pd.DataFrame | None,
    outdir: str,
    batch_cols: List[str] = ("TMA", "Hospital", "Staining Date", "Scan Date"),
    sample_col_for_pvca: str = "Patient",
    pre_intensity_cols: Dict[str, str] = DEFAULT_PRE_INTENSITY_COLS,
    pre_aggs: List[str] = DEFAULT_PRE_AGGS,
    post_features: List[str] = DEFAULT_POST_FEATURES,
    pvca_pct_threshold: float = 0.60,
    min_pairs: int = 10,
) -> Dict[str, Dict[str, object]]:
    """
    Run batch-effect diagnosis in PRE / POST / BOTH modes.

    - PRE: builds pseudo-bulk core table from raw cell-level intensities.
    - POST: uses final core-level endpoints (validation_calculations output).

    Outputs (per batch_col, per layer):
        PRE_by_<batch_col>/
            PCA.svg
            HCA_heatmap.svg
            feature_barplots/*.svg
            PVCA.csv
            PVCA.svg
        POST_by_<batch_col>/
            PCA.svg
            HCA_heatmap.svg
            feature_barplots/*.svg
            PVCA.csv
            PVCA.svg

    Args:
        mode:  "PRE", "POST", or "BOTH" (for reporting consistency; does not affect calculations).
        raw_df: Cell-level dataframe (PRE).
        validation_results: Core-level dataframe (POST).
        outdir: Output directory.
        batch_cols: Batch definitions to evaluate (e.g., ["TMA", "Hospital"]).
        sample_col_for_pvca: Kept for reporting consistency (printed in titles);
            in batch-only PVCA this is not used in the model.
        pre_intensity_cols: Marker->column for PRE aggregation.
        pre_aggs: Aggregations used for PRE.
        post_features: Feature list used for POST.
        pvca_pct_threshold: Cumulative explained variance threshold for PVCA PCA step.
        min_pairs: Minimum number of samples required for PVCA.

    Returns:
        Nested dict with:
            results[batch_col]["PRE_core"], results[batch_col]["PRE_pvca"]
            results[batch_col]["POST_core"], results[batch_col]["POST_pvca"]
    """
    mode_u = str(mode).strip().upper()
    if mode_u not in ("PRE", "POST", "BOTH"):
        raise ValueError("mode must be 'PRE', 'POST', or 'BOTH'")

    do_pre = mode_u in ("PRE", "BOTH")
    do_post = mode_u in ("POST", "BOTH")

    _safe_mkdir(outdir)
    results: Dict[str, Dict[str, object]] = {}

    # Metadata columns to always keep (in this order), plus any batch_cols that are not in the base set.
    base_meta = ["TMA", "Hospital", "Core", "Patient", "ID"]
    meta_cols = base_meta + [c for c in batch_cols if c not in base_meta]
    meta_cols_tuple = tuple(meta_cols)

    # PRE: build core-level pseudo-bulk features from raw cell-level data, only if mode includes PRE
    pre_core = None
    pre_feature_cols = None
    if do_pre:
        if raw_df is None:
            raise ValueError("raw_df is required when mode is PRE or BOTH")

        pre_core = build_prethreshold_core_features(
            raw_df=raw_df,
            intensity_cols=pre_intensity_cols,
            aggs=pre_aggs,
            group_cols=meta_cols_tuple,
        )

        pre_feature_cols = [c for c in pre_core.columns if c not in meta_cols]
        pre_core.to_csv(os.path.join(outdir, "prethreshold_core_features.csv"), index=False)

    # POST: select core-level endpoints from validation results, only if mode includes POST
    post_core = None
    post_feature_cols = None
    if do_post:
        if validation_results is None:
            raise ValueError("validation_results is required when mode is POST or BOTH")

        post_core = build_postthreshold_core_features(
            validation_df=validation_results,
            features=post_features,
            keep_cols=meta_cols_tuple,    
        )

        post_feature_cols = post_features.copy()
        post_core.to_csv(os.path.join(outdir, "postthreshold_core_features.csv"), index=False)

    # Per-batch analyses (PCA, HCA, feature plots, PVCA) for both PRE and POST cores (if they exist)
    for batch_col in list(batch_cols):
        results[batch_col] = {}

        # ===== PRE =====
        if do_pre and pre_core is not None:
            layer_dir = os.path.join(outdir, f"PRE_by_{batch_col}")
            _safe_mkdir(layer_dir)

            plot_pca(
                core_df=pre_core,
                feature_cols=pre_feature_cols,
                batch_col=batch_col,
                outpath=os.path.join(layer_dir, "PCA.svg"),
                title=f"PRE-threshold PCA (core pseudo-bulk) | batch={batch_col}",
            )

            plot_hca_heatmap(
                core_df=pre_core,
                feature_cols=pre_feature_cols,
                batch_col=batch_col,
                outpath=os.path.join(layer_dir, "HCA_heatmap.svg"),
                title=f"PRE-threshold HCA + heatmap | batch={batch_col}",
            )

            plot_feature_barplots(
                core_df=pre_core,
                feature_cols=pre_feature_cols,
                batch_col=batch_col,
                outdir=os.path.join(layer_dir, "feature_barplots"),
                title_prefix="PRE-threshold",
            )

            pv_pre = pvca_like_batch_only(
                core_df=pre_core,
                feature_cols=pre_feature_cols,
                batch_col=batch_col,
                pct_threshold=pvca_pct_threshold,
                min_samples=min_pairs,
            )
            pv_pre.to_csv(os.path.join(layer_dir, "PVCA.csv"), index=False)

            plot_pvca_bar(
                pvca_df=pv_pre,
                outpath=os.path.join(layer_dir, "PVCA.svg"),
                title=f"PRE PVCA | sample={sample_col_for_pvca} | batch={batch_col}",
            )

            results[batch_col]["PRE_core"] = pre_core
            results[batch_col]["PRE_pvca"] = pv_pre

        # ===== POST =====
        if do_post and post_core is not None:
            layer_dir = os.path.join(outdir, f"POST_by_{batch_col}")
            _safe_mkdir(layer_dir)

            plot_pca(
                core_df=post_core,
                feature_cols=post_feature_cols,
                batch_col=batch_col,
                outpath=os.path.join(layer_dir, "PCA.svg"),
                title=f"POST-threshold PCA (core endpoints) | batch={batch_col}",
            )

            plot_hca_heatmap(
                core_df=post_core,
                feature_cols=post_feature_cols,
                batch_col=batch_col,
                outpath=os.path.join(layer_dir, "HCA_heatmap.svg"),
                title=f"POST-threshold HCA + heatmap | batch={batch_col}",
            )

            plot_feature_barplots(
                core_df=post_core,
                feature_cols=post_feature_cols,
                batch_col=batch_col,
                outdir=os.path.join(layer_dir, "feature_barplots"),
                title_prefix="POST-threshold",
            )

            pv_post = pvca_like_batch_only(
                core_df=post_core,
                feature_cols=post_feature_cols,
                batch_col=batch_col,
                pct_threshold=pvca_pct_threshold,
                min_samples=min_pairs,
            )
            pv_post.to_csv(os.path.join(layer_dir, "PVCA.csv"), index=False)

            plot_pvca_bar(
                pvca_df=pv_post,
                outpath=os.path.join(layer_dir, "PVCA.svg"),
                title=f"POST PVCA | sample={sample_col_for_pvca} | batch={batch_col}",
            )

            results[batch_col]["POST_core"] = post_core
            results[batch_col]["POST_pvca"] = pv_post

    return results