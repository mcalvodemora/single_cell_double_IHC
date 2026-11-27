""" Script gmm_thresholds.py """

"""
Gaussian Mixture Model (GMM) thresholding utilities for single-cell IHC intensity data.

This module performs Gaussian Mixture Model (GMM) clustering on single-cell intensity 
distributions to identify latent subpopulations and determine objective threshold values.

The workflow consists of:

1) Model selection (automatic k determination)
   - Sweep k from 2 to max_clusters
   - Compute:
        • SSQ (reconstruction error via vector quantization to GMM means)
        • Silhouette score
        • BIC (Bayesian Information Criterion)
   - Apply a hybrid selection strategy:
        a) Detect elbow_k from SSQ curve
        b) Restrict search window to [elbow_k, elbow_k + elbow_margin]
        c) Keep models within BIC_min + bic_delta
        d) Choose model with highest silhouette (tie → smaller k)

2) Final GMM fitting
   - Fit model with chosen_k
   - Define threshold as mean(component means)
   - Save KDE + Gaussian components plot

Designed for brightfield IHC pipelines (GATA6 / CK17) after normalization.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis  
Developer: María Calvo de Mora  
Last Update: Feb 2026
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import vq

# Helper
def _safe_slug(s: str) -> str:
    """Convert a string to a safe slug for filenames.
    - Lowercase, replace spaces and special chars with underscores.
    - Remove non-alphanumeric chars except underscores.
    """
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("/", "_").replace("-", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return s

# =============================================================================
# Core: Single-k evaluation
# =============================================================================

def fit_gmm_and_evaluate(n_clusters, X):
    """
    Fit a Gaussian Mixture Model (GMM) with n_clusters and compute quality metrics.

    Metrics:
        - SSQ (reconstruction error): 
            Sum of squared distances from each point to nearest GMM mean.
        - Silhouette score:
            Measures cluster separation quality.
        - BIC:
            Penalized likelihood model selection criterion.

    Parameters
    ----------
    n_clusters : int
        Number of Gaussian components.
    X : np.ndarray
        2D array of shape (n_samples, 1).

    Returns
    -------
    tuple
        (ssq, silhouette, bic)
    """

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=10
    )

    labels = model.fit_predict(X)

    # Reconstruction error via VQ to component means
    _, err = vq(X, model.means_)
    ssq = float(np.sum(err ** 2))

    # Silhouette score — capped at 20k samples to avoid O(n²) cost on large datasets
    n = X.shape[0]
    ss = min(20_000, n) if n > 20_000 else None
    sil = float(silhouette_score(X, labels, sample_size=ss, random_state=10))

    # Bayesian Information Criterion
    bic = float(model.bic(X))

    return ssq, sil, bic


# =============================================================================
# Automatic k selection logic
# =============================================================================

def _detect_elbow_k(ks, costs):
    """
    Detect elbow point using maximum distance to straight line method.

    Classic "knee detection":
    - Draw a line between first and last SSQ points.
    - Choose k with maximum vertical deviation.

    Parameters
    ----------
    ks : array-like
        k values evaluated.
    costs : array-like
        Corresponding SSQ values.

    Returns
    -------
    int
        Detected elbow k.
    """

    ks = np.asarray(ks, dtype=float)
    cs = np.asarray(costs, dtype=float)

    if len(ks) < 3:
        return int(ks[0])

    x0, y0 = ks[0], cs[0]
    x1, y1 = ks[-1], cs[-1]

    if x1 == x0:
        return int(ks[0])

    y_line = y0 + (ks - x0) * (y1 - y0) / (x1 - x0)
    residual = np.abs(cs - y_line)

    return int(ks[np.argmax(residual)])


def _choose_optimal_k(metrics_df, bic_delta=0.05, elbow_margin=1):
    """
    Hybrid k-selection strategy.

    Steps:
    ------
    1) Detect elbow_k from SSQ.
    2) Restrict search window to [elbow_k, elbow_k + elbow_margin].
    3) Among window models:
         Keep those with BIC <= BIC_min + bic_delta_eff,
         where bic_delta_eff = bic_delta * (BIC_max - BIC_min) across ALL
         evaluated k values. This makes the tolerance adaptive to dataset scale.
    4) Select model with highest silhouette.
       If tie → choose smallest k.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Must contain columns: k, ssq, silhouette, bic.
    bic_delta : float
        BIC tolerance as a fraction of the global BIC range (all k evaluated).
        Default 0.05 = 5%. Scale-invariant: works regardless of dataset size.
    elbow_margin : int
        Maximum distance from elbow_k allowed.

    Returns
    -------
    chosen_k : int
    elbow_k : int
    k_cap : int
    """

    m = metrics_df.sort_values("k").copy()

    elbow_k = _detect_elbow_k(m["k"], m["ssq"])

    k_min_eval = int(m["k"].min())
    k_max_eval = int(m["k"].max())

    k0 = int(np.clip(elbow_k, k_min_eval, k_max_eval))
    k1 = int(np.clip(elbow_k + elbow_margin, k_min_eval, k_max_eval))

    window = m[(m["k"] >= k0) & (m["k"] <= k1)].copy()
    if window.empty:
        window = m.copy()

    # Adaptive tolerance: fraction of global BIC range across all evaluated k
    bic_global_range = float(m["bic"].max() - m["bic"].min())
    bic_delta_eff = bic_delta * bic_global_range

    bic_min = float(window["bic"].min())
    candidates = window[window["bic"] <= bic_min + bic_delta_eff].copy()

    if candidates.empty:
        candidates = window.copy()

    # Replace NaN silhouette with -inf to avoid selection
    candidates["_sil"] = candidates["silhouette"].fillna(-np.inf)

    # Highest silhouette wins, tie -> smallest k
    candidates = candidates.sort_values(["_sil", "k"], ascending=[False, True])
    chosen_k = int(candidates.iloc[0]["k"])

    return chosen_k, int(elbow_k), int(k1)


# =============================================================================
# k evaluation wrapper
# =============================================================================

def evaluate_gmm_clusters(
    data,
    col_name,
    output_dir,
    max_clusters=6,
    bic_delta=0.05,
    elbow_margin=1,
    group_label: str | None = None,
    marker_label: str | None = None,
):
    """
    Evaluate GMM models across k values and select optimal k automatically.

    Saves evaluation plot (Elbow + Silhouette).

    Parameters
    ----------
    bic_delta : float
        BIC tolerance as a fraction of the global BIC range (all k evaluated).
        Default 0.05 = 5%. Scale-invariant across dataset sizes.

    Returns
    -------
    chosen_k : int
    metrics_df : pd.DataFrame
    elbow_k : int
    """

    os.makedirs(output_dir, exist_ok=True)

    X = data[col_name].dropna().values.reshape(-1, 1)
    ks = list(range(2, max_clusters + 1))

    results = Parallel(n_jobs=-1)(
        delayed(fit_gmm_and_evaluate)(k, X) for k in ks
    )

    ssq, sil, bic = zip(*results)

    metrics_df = pd.DataFrame({
        "k": ks,
        "ssq": ssq,
        "silhouette": sil,
        "bic": bic,
    })

    chosen_k, elbow_k, k_cap = _choose_optimal_k(
        metrics_df,
        bic_delta=bic_delta,
        elbow_margin=elbow_margin,
    )
    # ---------------- Plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(ks, ssq, marker='o')
    axes[0].axvline(elbow_k, linestyle="--", label=f"elbow_k={elbow_k}")
    axes[0].axvline(chosen_k, linestyle="-", label=f"chosen_k={chosen_k}")
    axes[0].set_title("Elbow (SSQ)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Reconstruction Error")
    axes[0].legend()

    axes[1].plot(ks, sil, marker='o')
    axes[1].axvline(chosen_k, linestyle="-", label=f"chosen_k={chosen_k}")
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    axes[2].plot(ks, bic, marker='o')
    axes[2].axvline(chosen_k, linestyle="-", label=f"chosen_k={chosen_k}")
    axes[2].set_title("BIC")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("BIC")
    axes[2].legend()

    title_parts = ["GMM k selection"]
    if group_label:
        title_parts.append(f"group={group_label}")
    if marker_label:
        title_parts.append(f"marker={marker_label}")
    fig.suptitle(" | ".join(title_parts))

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    g = _safe_slug(group_label) if group_label else "group"
    m = _safe_slug(marker_label) if marker_label else _safe_slug(col_name)

    out_eval = os.path.join(output_dir, f"eval__{g}__{m}.svg")
    plt.savefig(out_eval)
    plt.close(fig)

    print(f"[AUTO-K] {col_name}: elbow_k={elbow_k}, chosen_k={chosen_k} | saved: {out_eval}")

    return chosen_k, metrics_df, elbow_k


# =============================================================================
# Final GMM fitting + thresholding
# =============================================================================

def plot_kde_and_gmm(
    data,
    col_name,
    output_dir,
    n_components=2,
    group_label: str | None = None,
    marker_label: str | None = None,
):
    """
    Fit final GMM and save KDE + Gaussian components plot.

    Threshold definition:
        threshold = mean(component_means)

    Returns
    -------
    threshold : float
    means : np.ndarray
    stds : np.ndarray
    weights : np.ndarray
    """

    os.makedirs(output_dir, exist_ok=True)

    X = data[col_name].dropna().values.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data[col_name], fill=True, bw_adjust=0.5, ax=ax)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=10
    )
    model.fit(X)

    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_).flatten()

    x = np.linspace(X.min(), X.max(), 1000)

    for i in range(n_components):
        y = model.weights_[i] * (
            1 / (stds[i] * np.sqrt(2 * np.pi))
        ) * np.exp(-0.5 * ((x - means[i]) / stds[i]) ** 2)
        ax.plot(x, y, label=f"Comp {i+1}")

    threshold = float(np.mean(means))
    ax.axvline(threshold, linestyle='--', label=f"Threshold={threshold:.3f}")

    title_parts = ["GMM fit"]
    if group_label:
        title_parts.append(f"group={group_label}")
    if marker_label:
        title_parts.append(f"marker={marker_label}")
    title_parts.append(f"k={n_components}")
    ax.set_title(" | ".join(title_parts))
    ax.legend()

    g = _safe_slug(group_label) if group_label else "group"
    m = _safe_slug(marker_label) if marker_label else _safe_slug(col_name)

    out_fit = os.path.join(output_dir, f"fit__{g}__{m}__k{n_components}.svg")
    plt.savefig(out_fit)
    plt.close(fig)

    return threshold, means, stds, model.weights_


# =============================================================================
# High-level automatic pipeline
# =============================================================================

def auto_gmm_threshold(
    data,
    col_name,
    output_dir,
    max_clusters=6,
    bic_delta=0.05,
    elbow_margin=1,
    group_label: str = "group",
    marker_label: str = "marker",
):
    """
    Full automatic pipeline:

    1) Evaluate ks and choose optimal k.
    2) Fit final GMM with chosen_k.
    3) Return threshold and metrics.

    Parameters
    ----------
    bic_delta : float
        BIC tolerance as a fraction of the global BIC range (all k evaluated).
        Default 0.05 = 5%. Scale-invariant across dataset sizes.

    Returns
    -------
    threshold : float
    chosen_k : int
    metrics_df : pd.DataFrame
    fit_params : dict
        {"chosen_k", "elbow_k", "means", "stds", "weights", "threshold"} for the
        final fit — convenience bundle for downstream reporting/plotting, does not
        change the fitting or k-selection logic above.
    """
    group_dir = os.path.join(output_dir, _safe_slug(group_label))
    os.makedirs(group_dir, exist_ok=True)
    chosen_k, metrics_df, elbow_k = evaluate_gmm_clusters(
        data,
        col_name,
        output_dir=group_dir,
        max_clusters=max_clusters,
        bic_delta=bic_delta,
        elbow_margin=elbow_margin,
        group_label=group_label,
        marker_label=marker_label,
    )

    threshold, means, stds, weights = plot_kde_and_gmm(
        data,
        col_name,
        n_components=chosen_k,
        output_dir=group_dir,
        group_label=group_label,
        marker_label=marker_label,
    )
    print(f"[AUTO-GMM] {col_name}: k={chosen_k}, threshold={threshold:.4f}")

    fit_params = {
        "chosen_k": chosen_k,
        "elbow_k": elbow_k,
        "means": means,
        "stds": stds,
        "weights": weights,
        "threshold": threshold,
    }

    return threshold, chosen_k, metrics_df, fit_params


# =============================================================================
# Supplementary figure: GMM k-selection + final fits, across the 5 staining groups
# =============================================================================

# Fixed display order / labels / colors for the 5 staining groups, shared between
# Panel A and Panel B so a group keeps the same color in both.
GMM_GROUP_ORDER = ["DAB++", "Low_dab", "Low_mag+dab", "Mag++", "Mag+DAB"]
GMM_GROUP_LABELS = {
    "DAB++": "DAB++",
    "Low_dab": "Low_DAB",
    "Low_mag+dab": "Low_MAG+DAB",
    "Mag++": "MAG++",
    "Mag+DAB": "MAG+DAB",
}
GMM_GROUP_COLORS = {
    "DAB++": "#1f77b4",
    "Low_dab": "#ff7f0e",
    "Low_mag+dab": "#2ca02c",
    "Mag++": "#d62728",
    "Mag+DAB": "#9467bd",
}


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1]; returns 0.5 everywhere if the array is constant."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def plot_gmm_supplementary_figure(
    metrics_all: pd.DataFrame,
    fits: dict,
    data: pd.DataFrame,
    group_col: str,
    marker_cols: dict,
    outdir: str,
    kde_max_cells: int = 200_000,
    kde_random_state: int = 42,
) -> None:
    """
    Build the "Figure_S_GMM_evaluation" supplementary figure (saved as .svg, .pdf
    and .png @300dpi in `outdir`), summarizing k-selection and the final GMM fits
    across the 5 staining groups, for both markers.

    This is purely a reporting/plotting function: it only reads `metrics_all` and
    `fits` (already produced by the evaluate/fit loop) and `data` (for the KDE
    backdrop), and does not recompute or alter any threshold or k selection.

    Panel A: 2x3 grid (rows GATA6/CK17, columns reconstruction error / silhouette /
    BIC). One line per staining group vs k=2..6, each curve min-max scaled to [0,1]
    within its own (group, marker, criterion), with the chosen k marked by a larger
    filled point. A single shared legend lists the 5 groups.

    Panel B: 5x2 grid (rows = the 5 staining groups, columns = GATA6, CK17). Each
    panel shows the KDE of the group's raw normalized intensities (bw_adjust=0.5;
    drawn from a fixed-seed subsample of up to `kde_max_cells` cells — subsampling
    is only ever used for drawing this KDE curve, never for fitting any GMM),
    the weighted Gaussian components as thin lines, and a dashed vertical line at
    the chosen threshold.

    Parameters
    ----------
    metrics_all : pd.DataFrame
        Long-format k-sweep metrics with columns:
        group, marker, k, ssq, silhouette, bic, elbow_k, chosen_k.
    fits : dict
        Nested dict fits[group][marker] -> fit_params dict as returned by
        auto_gmm_threshold (keys: chosen_k, elbow_k, means, stds, weights, threshold).
    data : pd.DataFrame
        Cell-level table used only to redraw the KDE backdrop in Panel B.
    group_col : str
        Column in `data` that encodes the staining group (e.g. "Staining").
    marker_cols : dict
        {"GATA6": <raw normalized intensity column>, "CK17": <raw normalized intensity column>}.
    outdir : str
        Output directory for the figure files.
    """
    os.makedirs(outdir, exist_ok=True)

    markers = list(marker_cols.keys())
    criteria = [("ssq", "Reconstruction error (SSQ)"),
                ("silhouette", "Silhouette score"),
                ("bic", "BIC")]

    fig = plt.figure(figsize=(15, 21))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[2, 5], hspace=0.28)
    gs_A = gs_outer[0].subgridspec(2, 3, wspace=0.3, hspace=0.45)
    gs_B = gs_outer[1].subgridspec(5, 2, wspace=0.25, hspace=0.9)

    fig.text(0.02, 0.985, "A", fontsize=18, fontweight="bold", fontfamily="sans-serif")
    fig.text(0.02, 0.70, "B", fontsize=18, fontweight="bold", fontfamily="sans-serif")

    # ---------------- Panel A: k-selection curves ----------------
    legend_handles = {}
    for row, marker in enumerate(markers):
        for col, (crit_col, crit_title) in enumerate(criteria):
            ax = fig.add_subplot(gs_A[row, col])
            for group in GMM_GROUP_ORDER:
                sub = metrics_all[
                    (metrics_all["group"] == group) & (metrics_all["marker"] == marker)
                ].sort_values("k")
                if sub.empty:
                    continue
                color = GMM_GROUP_COLORS[group]
                scaled = _minmax_scale(sub[crit_col].values)
                line, = ax.plot(sub["k"], scaled, marker="o", markersize=4,
                                 color=color, label=GMM_GROUP_LABELS[group])
                legend_handles[group] = line

                chosen_k = sub["chosen_k"].iloc[0]
                chosen_row = sub[sub["k"] == chosen_k]
                if not chosen_row.empty:
                    chosen_idx = chosen_row.index[0]
                    chosen_scaled = scaled[sub.index.get_loc(chosen_idx)]
                    ax.scatter([chosen_k], [chosen_scaled], s=90, color=color,
                               edgecolor="black", linewidth=0.8, zorder=5)

            if row == 0:
                ax.set_title(crit_title, fontsize=10, fontweight="bold", fontfamily="sans-serif")
            if col == 0:
                ax.set_ylabel(f"{marker}\nscaled", fontsize=9, fontweight="bold", fontfamily="sans-serif")
            else:
                ax.set_ylabel("scaled", fontsize=8, fontfamily="sans-serif")
            if row == len(markers) - 1:
                ax.set_xlabel("k", fontsize=9, fontfamily="sans-serif")
            ax.set_xticks(sorted(metrics_all["k"].unique()))
            ax.tick_params(labelsize=7)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

    fig.legend(
        handles=[legend_handles[g] for g in GMM_GROUP_ORDER if g in legend_handles],
        labels=[GMM_GROUP_LABELS[g] for g in GMM_GROUP_ORDER if g in legend_handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.985 - 0.015),
        ncol=5, fontsize=9, frameon=False,
    )

    # ---------------- Panel B: KDE + GMM components per group/marker ----------------
    rng = np.random.RandomState(kde_random_state)

    for row, group in enumerate(GMM_GROUP_ORDER):
        if group not in fits:
            continue
        group_data = data[data[group_col] == group]
        color = GMM_GROUP_COLORS[group]

        for col, marker in enumerate(markers):
            ax = fig.add_subplot(gs_B[row, col])
            if marker not in fits[group]:
                ax.axis("off")
                continue

            fp = fits[group][marker]
            means = np.asarray(fp["means"]).flatten()
            stds = np.asarray(fp["stds"]).flatten()
            weights = np.asarray(fp["weights"]).flatten()
            threshold = fp["threshold"]
            chosen_k = fp["chosen_k"]

            col_name = marker_cols[marker]
            values = group_data[col_name].dropna().values
            if len(values) > kde_max_cells:
                idx = rng.choice(len(values), size=kde_max_cells, replace=False)
                kde_values = values[idx]
            else:
                kde_values = values

            sns.kdeplot(kde_values, fill=True, bw_adjust=0.5, ax=ax,
                        color=color, alpha=0.35, linewidth=1.2)
            # Grab the just-drawn KDE curve's own (x, y) data — read-only, purely
            # for the CK17 y-axis rescaling below, does not touch the KDE itself.
            # fill=True makes seaborn draw a filled PolyCollection (no Line2D), so
            # read its path vertices instead (includes the y=0 baseline, which
            # doesn't affect the max() used below).
            kde_verts = ax.collections[-1].get_paths()[0].vertices
            kde_line_x, kde_line_y = kde_verts[:, 0], kde_verts[:, 1]

            x = np.linspace(0, 1, 1000)
            component_curves = []
            for i in range(len(means)):
                y = weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x - means[i]) / stds[i]) ** 2
                )
                component_curves.append(y)
                ax.plot(x, y, linewidth=0.8, color="black", alpha=0.6)

            ax.axvline(threshold, linestyle="--", color="black", linewidth=1.0)

            ax.set_title(
                f"{GMM_GROUP_LABELS[group]} · {marker} · k={chosen_k} · T={threshold:.4f}",
                fontsize=9, fontfamily="sans-serif",
            )
            ax.set_xlim(0, 1)

            # CK17-only: the normalization clip piles mass at x=1, so the KDE/
            # component peak there dwarfs everything else in [0, 0.95]. Rescale
            # the y-axis to that sub-range and flag the truncated peak with text.
            # GATA6 subplots are left untouched (no ylim change, no text).
            if marker == "CK17":
                in_range_mask = kde_line_x <= 0.95
                kde_max_in_range = float(np.max(kde_line_y[in_range_mask])) if in_range_mask.any() else float(np.max(kde_line_y))
                comp_max_in_range = 0.0
                x_in_range = x <= 0.95
                for y in component_curves:
                    if x_in_range.any():
                        comp_max_in_range = max(comp_max_in_range, float(np.max(y[x_in_range])))
                ylim_top = 1.35 * max(kde_max_in_range, comp_max_in_range)
                ax.set_ylim(0, ylim_top)
                ax.text(0.985, 0.95 * ylim_top, "peak truncated",
                         fontsize=6.5, style="italic", color="black",
                         ha="right", va="top", fontfamily="sans-serif")
            if row == len(GMM_GROUP_ORDER) - 1:
                ax.set_xlabel("Normalized raw intensity", fontsize=8, fontfamily="sans-serif")
            ax.set_ylabel("Density", fontsize=8, fontfamily="sans-serif")
            ax.tick_params(labelsize=7)

    for ext in ("svg", "pdf", "png"):
        out_path = os.path.join(outdir, f"Figure_S_GMM_evaluation.{ext}")
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_path, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"[GMM FIGURE] saved: {os.path.join(outdir, 'Figure_S_GMM_evaluation.*')}")