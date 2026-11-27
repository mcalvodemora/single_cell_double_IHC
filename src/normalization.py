# normalization.py
"""
Quantile-based per-TMA intensity normalization for the double-IHC cohort.

Absolute brightfield intensity scales drift across staining runs, antibody
lots, scanner profiles and TMA processing, enough to bias steps that pool or
compare across TMAs (fitting a single GMM, applying shared thresholds), even
when that drift isn't obvious as batch clustering on its own. This module
harmonizes each marker's intensity per TMA with a robust quantile min-max
scaling:

    NewVal = (OldVal - q01) / (q99 - q01)

where q01/q99 are the 1st/99th percentiles of that marker's intensity within
that TMA. Values can fall outside [0, 1] by construction (clipping is
optional) since only the central 98% of values are guaranteed to land inside
it. Returns the dataframe with the added normalized columns, plus the
per-TMA quantile table used, which is also saved to disk for traceability.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


## Defaults / configuration
# Per-cell staining intensity columns: GATA6 read from the nucleus ("DAB
# Intensity - Nuclei") and CK17 from the cytoplasm ("Mag Intensity -
# Cytoplasm").
DEFAULT_INTENSITY_COLS: Dict[str, str] = {
    "GATA6": "DAB Intensity - Nuclei",
    "CK17": "Mag Intensity - Cytoplasm",
}


# Core normalization

def normalize_intensities_by_tma_quantiles(
    df: pd.DataFrame,
    tma_col: str = "TMA",
    intensity_cols: Optional[Dict[str, str]] = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
    suffix: str = "_q01q99_norm",
    clip: bool = False,
    outdir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize selected intensity columns per TMA using quantile-based min-max scaling.

    The normalization is applied independently for each TMA and each marker:

        NewVal = (OldVal - q_low) / (q_high - q_low)

    where q_low and q_high are computed from the distribution of OldVal
    within each TMA.

    Args:
        df:
            Cell-level dataframe (one row per cell). Must contain `tma_col`
            and all columns listed in `intensity_cols`.
        tma_col:
            Name of the TMA column.
        intensity_cols:
            Mapping {marker_name: column_name_in_df}. If None, uses
            DEFAULT_INTENSITY_COLS (GATA6/CK17).
        q_low:
            Lower quantile (default 0.01).
        q_high:
            Upper quantile (default 0.99).
        suffix:
            Suffix appended to each normalized output column name.
            Example: "DAB Intensity - Nuclei" -> "DAB Intensity - Nuclei_q01q99_norm"
        clip:
            If True, clip normalized values to [0, 1]. Default False to preserve
            out-of-range values as described in the method.
        outdir:
            If provided, saves the per-TMA quantile table to:
                <outdir>/tma_quantiles_q01_q99.csv

    Returns:
        df_out:
            Copy of df with additional normalized columns (one per intensity column).
        q_table:
            Per-TMA quantile table (q01, q99, denom) for each marker.
            Useful for QC, reproducibility, and paper reporting.

    Raises:
        ValueError:
            If required columns are missing or if q_low >= q_high.
    """
    if q_low >= q_high:
        raise ValueError("q_low must be < q_high.")

    if intensity_cols is None:
        intensity_cols = DEFAULT_INTENSITY_COLS

    if tma_col not in df.columns:
        raise ValueError(f"'{tma_col}' not found in dataframe.")

    for marker, col in intensity_cols.items():
        if col not in df.columns:
            raise ValueError(f"Intensity column not found for {marker}: '{col}'")

    # Work on a copy to avoid side effects.
    df_out = df.copy()

    # Ensure intensity columns are numeric (strings -> NaN where invalid).
    for col in intensity_cols.values():
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Build a per-TMA quantile table (traceability)
    rows = []
    for tma, sub in df_out.groupby(tma_col, dropna=False):
        row = {tma_col: tma}

        for marker, col in intensity_cols.items():
            vals = sub[col].dropna()

            if len(vals) == 0:
                ql = np.nan
                qh = np.nan
                denom = np.nan
            else:
                ql = float(vals.quantile(q_low))
                qh = float(vals.quantile(q_high))
                denom = float(qh - ql)

                # Avoid division by zero / near-zero
                if denom == 0 or np.isclose(denom, 0):
                    denom = np.nan

            row[f"{marker}_q{int(q_low * 100):02d}"] = ql
            row[f"{marker}_q{int(q_high * 100):02d}"] = qh
            row[f"{marker}_denom"] = denom

        rows.append(row)

    q_table = pd.DataFrame(rows)

    # Optionally persist the quantile table for reproducibility.
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        q_table.to_csv(os.path.join(outdir, "tma_quantiles_q01_q99.csv"), index=False)

    # Apply normalization per TMA by merging quantiles back to the cell-level table
    for marker, col in intensity_cols.items():
        new_col = f"{col}{suffix}"

        ql_name = f"{marker}_q{int(q_low * 100):02d}"
        qh_name = f"{marker}_q{int(q_high * 100):02d}"
        denom_name = f"{marker}_denom"

        # Merge quantiles to each cell row based on its TMA.
        df_out = df_out.merge(
            q_table[[tma_col, ql_name, qh_name, denom_name]],
            on=tma_col,
            how="left",
        )

        # Quantile-based scaling. (No clipping by default.)
        df_out[new_col] = (df_out[col] - df_out[ql_name]) / df_out[denom_name]

        if clip:
            df_out[new_col] = df_out[new_col].clip(0, 1)

        # Drop helper columns for this marker to keep the dataframe tidy.
        df_out.drop(columns=[ql_name, qh_name, denom_name], inplace=True, errors="ignore")

    return df_out, q_table
