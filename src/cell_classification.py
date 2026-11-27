"""
Cell-level threshold-based classification for the double-IHC assay.

Reproduces in Python the same per-cell gating logic previously done inside
Visiopharm, applied here to the GMM thresholds computed earlier in the
pipeline.

Lower intensity means stronger staining in this assay (0 = strongly
positive, 255 = white/negative), so positivity is intensity <= threshold.
Thresholds are group-specific: each of the five staining groups (DAB++,
Low_dab, Low_mag+dab, Mag++, Mag+DAB, given by the `Staining` column) has its
own GATA6/CK17 threshold pair, fit by GMM on that group's normalized
intensities and applied to that same group's phenotyping-normalized
intensities.

Adds per-cell 0/1 columns "DAB+/- (#)" and "CK17+/- (#)", plus intensity-tier
columns "Negative/1+/2+/3+ DAB (#)" and the CK17 equivalents, splitting the
positive range [0, threshold] into three equal thirds (3+ closest to 0, 1+
closest to the threshold).

Developer: María Calvo de Mora
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _as_float(s: pd.Series) -> pd.Series:
    """Coerce series to float safely."""
    return pd.to_numeric(s, errors="coerce")


def _bin_positive_intensity(v: pd.Series, thr: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create 3+, 2+, 1+ boolean bins for positive cells, where positive means v <= thr.

    Bins are defined as:
        3+ : v <= thr/3
        2+ : thr/3 < v <= 2*thr/3
        1+ : 2*thr/3 < v <= thr

    Returns
    -------
    (is_3p, is_2p, is_1p) as boolean series.
    """
    t3 = thr / 3.0
    t2 = 2 * thr / 3.0

    is_3p = v.le(t3)
    is_2p = v.gt(t3) & v.le(t2)
    is_1p = v.gt(t2) & v.le(thr)

    return is_3p, is_2p, is_1p

def _mean01(x: pd.Series) -> float:
    """
    Safe mean for 0/1 columns (ignores NaNs).
    Returns NaN if the series is empty.
    """
    if x is None or x.size == 0:
        return float("nan")
    return float(pd.to_numeric(x, errors="coerce").mean())


def compute_core_hscores(
    cell_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("TMA", "Hospital", "Core", "Patient", "ID"),
    out_suffix: str = "",
) -> pd.DataFrame:
    """
    Aggregate cell-level bin columns into core-level percentages and H-scores.

    Expected columns (0/1 ints) created by apply_group_specific_thresholds():
        - "Negative DAB (#)", "1+ DAB (#)", "2+ DAB (#)", "3+ DAB (#)"
        - "Negative CK17 (#)", "1+ CK17 (#)", "2+ CK17 (#)", "3+ CK17 (#)"
      (optionally with `out_suffix` appended)

    Output columns (one row per core):
        - n_cells
        - % Negative / % 1+ / % 2+ / % 3+ for DAB and CK17 (0-100)
        - H-Score DAB, H-Score CK17 (0-300)

    Notes
    -----
    H-score definition:
        H = 0*(%Negative) + 1*(%1+) + 2*(%2+) + 3*(%3+)
    using percentages in [0,100], so H ranges in [0,300].
    """
    df = cell_df.copy()

    req = [
        f"Negative DAB (#){out_suffix}", f"1+ DAB (#){out_suffix}", f"2+ DAB (#){out_suffix}", f"3+ DAB (#){out_suffix}",
        f"Negative CK17 (#){out_suffix}", f"1+ CK17 (#){out_suffix}", f"2+ CK17 (#){out_suffix}", f"3+ CK17 (#){out_suffix}",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required gating columns for core aggregation: {missing}")

    # Ensure numeric 0/1
    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(list(group_cols), dropna=False)

    rows = []
    for keys, sub in g:
        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        row["n_cells"] = int(sub.shape[0])

        # Percentages in [0,100]
        p_neg_dab = 100.0 * _mean01(sub[f"Negative DAB (#){out_suffix}"])
        p_1_dab   = 100.0 * _mean01(sub[f"1+ DAB (#){out_suffix}"])
        p_2_dab   = 100.0 * _mean01(sub[f"2+ DAB (#){out_suffix}"])
        p_3_dab   = 100.0 * _mean01(sub[f"3+ DAB (#){out_suffix}"])

        p_neg_ck  = 100.0 * _mean01(sub[f"Negative CK17 (#){out_suffix}"])
        p_1_ck    = 100.0 * _mean01(sub[f"1+ CK17 (#){out_suffix}"])
        p_2_ck    = 100.0 * _mean01(sub[f"2+ CK17 (#){out_suffix}"])
        p_3_ck    = 100.0 * _mean01(sub[f"3+ CK17 (#){out_suffix}"])

        row["% Negative DAB"] = p_neg_dab
        row["% 1+ DAB"] = p_1_dab
        row["% 2+ DAB"] = p_2_dab
        row["% 3+ DAB"] = p_3_dab

        row["% Negative CK17"] = p_neg_ck
        row["% 1+ CK17"] = p_1_ck
        row["% 2+ CK17"] = p_2_ck
        row["% 3+ CK17"] = p_3_ck

        # H-scores in [0,300]
        row["H-Score DAB"]  = 0.0 * p_neg_dab + 1.0 * p_1_dab + 2.0 * p_2_dab + 3.0 * p_3_dab
        row["H-Score CK17"] = 0.0 * p_neg_ck  + 1.0 * p_1_ck  + 2.0 * p_2_ck  + 3.0 * p_3_ck

        rows.append(row)

    return pd.DataFrame(rows)


def broadcast_core_columns_to_cells(
    cell_df: pd.DataFrame,
    core_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("TMA", "Hospital", "Core", "Patient", "ID"),
    cols_to_broadcast: Tuple[str, ...] = ("H-Score DAB", "H-Score CK17"),
) -> pd.DataFrame:
    """
    Merge core-level endpoints back into the cell-level table, so every cell in the
    same core carries the same H-score (and any other selected core-level columns).

    This is convenient when downstream code expects a single cell-level table.
    """
    missing = [c for c in list(group_cols) + list(cols_to_broadcast) if c not in core_df.columns]
    if missing:
        raise ValueError(f"core_df missing required columns: {missing}")

    core_small = core_df[list(group_cols) + list(cols_to_broadcast)].copy()
    out = cell_df.merge(core_small, on=list(group_cols), how="left")
    return out


def add_core_hscores_to_cells(
    cell_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("TMA", "Hospital", "Core", "Patient", "ID"),
    out_suffix: str = "",
    cols_to_broadcast: Tuple[str, ...] = ("H-Score DAB", "H-Score CK17"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
      1) compute core-level H-scores from cell-level bins
      2) broadcast H-scores back to each cell row

    Returns
    -------
    cell_with_core : pd.DataFrame
        Cell-level table with added core-level columns.
    core_endpoints : pd.DataFrame
        Core-level endpoints table (one row per core).
    """
    core_endpoints = compute_core_hscores(
        cell_df=cell_df,
        group_cols=group_cols,
        out_suffix=out_suffix,
    )
    cell_with_core = broadcast_core_columns_to_cells(
        cell_df=cell_df,
        core_df=core_endpoints,
        group_cols=group_cols,
        cols_to_broadcast=cols_to_broadcast,
    )
    return cell_with_core, core_endpoints

# -------------------------------------------------------------------------
# Main API
# -------------------------------------------------------------------------

def apply_group_specific_thresholds_5groups(
    df: pd.DataFrame,
    staining_col: str,
    group_thresholds: Dict[str, Dict[str, float]],
    dab_intensity_col: str,
    ck17_intensity_col: str,
    out_suffix: str = "",
) -> pd.DataFrame:
    """
    Apply per-staining-group thresholds (5 groups, one GMM pair each) and generate
    Visiopharm-like cell classification columns.

    Unlike apply_group_specific_thresholds (2 macro-groups DAB/MAG), here each of the
    5 raw values of `staining_col` (DAB++, Low_dab, Low_mag+dab, Mag++, Mag+DAB) gets
    its own GATA6/CK17 threshold pair, since mixing them for GMM fitting was producing
    misclassification on weakly-stained GATA6 cases.

    Parameters
    ----------
    df : pd.DataFrame
        Full cell-level table (one row per cell).
    staining_col : str
        Column that encodes the staining group (e.g. "Staining").
    group_thresholds : dict
        Mapping {staining_value: {"GATA6": float, "CK17": float}}, one entry per
        value of staining_col to be classified.
    dab_intensity_col : str
        Column with DAB nuclear intensity (e.g., "DAB Intensity - Nuclei Phenotyping_norm").
    ck17_intensity_col : str
        Column with MAG cytoplasm intensity (e.g., "Mag Intensity - Cytoplasm Phenotyping_norm").
    out_suffix : str
        Optional suffix appended to generated column names.

    Returns
    -------
    pd.DataFrame
        Copy of df with extra classification columns.
    """

    out = df.copy()

    out[dab_intensity_col] = _as_float(out[dab_intensity_col])
    out[ck17_intensity_col] = _as_float(out[ck17_intensity_col])

    cols_to_create = [
        f"DAB+ (#){out_suffix}", f"DAB- (#){out_suffix}",
        f"CK17+ (#){out_suffix}", f"CK17- (#){out_suffix}",
        f"Negative DAB (#){out_suffix}", f"1+ DAB (#){out_suffix}", f"2+ DAB (#){out_suffix}", f"3+ DAB (#){out_suffix}",
        f"Negative CK17 (#){out_suffix}", f"1+ CK17 (#){out_suffix}", f"2+ CK17 (#){out_suffix}", f"3+ CK17 (#){out_suffix}",
    ]
    for c in cols_to_create:
        out[c] = 0

    for staining_value, thr in group_thresholds.items():
        is_group = out[staining_col] == staining_value

        thr_gata6 = float(thr["GATA6"])
        thr_ck17 = float(thr["CK17"])

        v_dab = out.loc[is_group, dab_intensity_col]
        v_ck17 = out.loc[is_group, ck17_intensity_col]

        dab_pos = v_dab.le(thr_gata6)
        ck17_pos = v_ck17.le(thr_ck17)

        out.loc[is_group, f"DAB+ (#){out_suffix}"] = dab_pos.astype(int)
        out.loc[is_group, f"DAB- (#){out_suffix}"] = (1 - dab_pos.astype(int))

        out.loc[is_group, f"CK17+ (#){out_suffix}"] = ck17_pos.astype(int)
        out.loc[is_group, f"CK17- (#){out_suffix}"] = (1 - ck17_pos.astype(int))

        is_3p, is_2p, is_1p = _bin_positive_intensity(v_dab, thr_gata6)
        out.loc[is_group, f"3+ DAB (#){out_suffix}"] = (dab_pos & is_3p).astype(int)
        out.loc[is_group, f"2+ DAB (#){out_suffix}"] = (dab_pos & is_2p).astype(int)
        out.loc[is_group, f"1+ DAB (#){out_suffix}"] = (dab_pos & is_1p).astype(int)
        out.loc[is_group, f"Negative DAB (#){out_suffix}"] = (~dab_pos).astype(int)

        is_3p, is_2p, is_1p = _bin_positive_intensity(v_ck17, thr_ck17)
        out.loc[is_group, f"3+ CK17 (#){out_suffix}"] = (ck17_pos & is_3p).astype(int)
        out.loc[is_group, f"2+ CK17 (#){out_suffix}"] = (ck17_pos & is_2p).astype(int)
        out.loc[is_group, f"1+ CK17 (#){out_suffix}"] = (ck17_pos & is_1p).astype(int)
        out.loc[is_group, f"Negative CK17 (#){out_suffix}"] = (~ck17_pos).astype(int)

    return out