"""summary_tables.py"""

"""
Generates descriptive summary tables for the pancreatic cancer TMA cohort.

Two tables are produced:

1. Clinical characteristics table (Table 1):
   - One row per patient (deduplicated from core-level merged data).
   - Continuous variables → median (Q1–Q3).
   - Categorical variables → n (%) per category.
   - Includes a 'n_missing' column.

2. TMA metadata table:
   - One column per TMA + a 'Total' column.
   - Rows: n patients, n cores, cores-per-patient distribution,
     staining date, scan date, Lot GATA6, Lot CK17.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os

import numpy as np
import pandas as pd

from clinical import CATEGORICAL_COLS, NUMERIC_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_date(val) -> str:
    """Format a date value as dd/mm/yyyy, or return '—' if missing."""
    try:
        return pd.to_datetime(val).strftime("%d/%m/%Y")
    except Exception:
        return str(val) if pd.notna(val) else "—"


# ---------------------------------------------------------------------------
# Clinical characteristics table (Table 1)
# ---------------------------------------------------------------------------


def generate_clinical_table(
    merged_df: pd.DataFrame,
    outdir: str,
    patient_id_col: str = "ID",
    numeric_cols: list = None,
    categorical_cols: list = None,
    outname: str = "clinical_characteristics.csv",
) -> pd.DataFrame:
    """
    Generate a descriptive clinical characteristics table (Table 1).

    The input merged_df has one row per core (output of
    run_clinical_multivariate_exploration). It is deduplicated internally to
    one row per patient using patient_id_col before computing statistics.

    For each variable:
    - Continuous (numeric_cols): median (Q1–Q3) across patients.
    - Categorical (categorical_cols): n (%) for each observed category,
      with a header row showing total missing for the variable.

    Variables not found in merged_df are silently skipped.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Core-level DataFrame with clinical + image columns.
    outdir : str
        Directory where the output CSV will be saved.
    patient_id_col : str
        Column used to identify unique patients. Default "ID".
    numeric_cols : list, optional
        Columns treated as continuous. Defaults to NUMERIC_COLS from clinical.py.
    categorical_cols : list, optional
        Columns treated as categorical. Defaults to CATEGORICAL_COLS from clinical.py.
    outname : str
        Output CSV filename.

    Returns
    -------
    pd.DataFrame
        Table with columns: Variable, Value, n_missing.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    _safe_mkdir(outdir)

    patient_df = merged_df.drop_duplicates(subset=[patient_id_col]).copy()
    n_total = len(patient_df)
    print(f"  Clinical table: {n_total} unique patients.")

    rows = []

    # --- Continuous variables ---
    for col in numeric_cols:
        if col not in patient_df.columns:
            continue
        series = pd.to_numeric(patient_df[col], errors="coerce")
        n_miss = int(series.isna().sum())
        valid = series.dropna()
        if len(valid) == 0:
            value_str = "—"
        else:
            med = valid.median()
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            value_str = f"{med:.1f} ({q1:.1f}–{q3:.1f})"
        rows.append(
            {
                "Variable": col,
                "Value": value_str,
                "n_missing": n_miss if n_miss > 0 else "",
            }
        )

    # --- Categorical variables ---
    for col in categorical_cols:
        if col not in patient_df.columns:
            continue
        series = patient_df[col].copy()
        n_miss = int(series.isna().sum())
        valid = series.dropna()
        denom = len(valid)

        # Header row for the variable (shows total missing)
        rows.append(
            {
                "Variable": col,
                "Value": "",
                "n_missing": n_miss if n_miss > 0 else "",
            }
        )
        # One indented row per category, sorted by category label
        for cat_val, cnt in valid.value_counts(sort=False).sort_index().items():
            pct = 100 * cnt / denom if denom > 0 else 0
            rows.append(
                {
                    "Variable": f"  {cat_val}",
                    "Value": f"{cnt} ({pct:.1f}%)",
                    "n_missing": "",
                }
            )

    table = pd.DataFrame(rows, columns=["Variable", "Value", "n_missing"])
    out_path = os.path.join(outdir, outname)
    table.to_csv(out_path, index=False)
    print(f"Clinical characteristics table saved → {out_path}")
    return table


# ---------------------------------------------------------------------------
# TMA metadata table
# ---------------------------------------------------------------------------


def generate_metadata_table(
    df: pd.DataFrame,
    outdir: str,
    tma_col: str = "TMA",
    core_col: str = "Core",
    patient_col: str = "Patient",
    staining_date_col: str = "Staining Date",
    scan_date_col: str = "Scan Date",
    lot_gata6_col: str = "Lot GATA6",
    lot_ck17_col: str = "Lot CK17",
    scan_time_col: str = "Scan Time",
    scan_profile_col: str = "Scan Profile",
    protocol_gata6_col: str = "Protocol GATA6",
    protocol_ck17_col: str = "Protocol CK17",
    batch_col: str = "Batch",
    outname: str = "tma_metadata.csv",
) -> pd.DataFrame:
    """
    Generate a TMA metadata summary table.

    The input df is deduplicated internally to one row per core before
    computing statistics.

    Output columns: one per TMA (sorted) plus a 'Total' column.
    Output rows:
        n patients, n cores,
        1 core n (%), 2 cores n (%), 3 cores n (%),
        Staining date, Scan date, Scan time, Scan profile,
        Lot GATA6, Lot CK17, Protocol GATA6, Protocol CK17.

    Staining date, scan date, and lots are assumed unique per TMA (first
    non-null value is taken). These fields are set to '—' in the Total column.

    Parameters
    ----------
    df : pd.DataFrame
        Cell-level (or core-level) DataFrame. Deduplicated to core level
        internally using tma_col + core_col as composite key.
    outdir : str
        Directory where the output CSV will be saved.
    tma_col, core_col, patient_col : str
        Column names for TMA, Core, and Patient identifiers.
    staining_date_col, scan_date_col, scan_time_col, scan_profile_col : str
    lot_gata6_col, lot_ck17_col, protocol_gata6_col, protocol_ck17_col : str
        Column names for metadata fields.
    outname : str
        Output CSV filename.

    Returns
    -------
    pd.DataFrame
        Table with variables as row index, TMAs + Total as columns.
    """
    _safe_mkdir(outdir)

    available = [
        c
        for c in [
            tma_col,
            core_col,
            patient_col,
            staining_date_col,
            scan_date_col,
            lot_gata6_col,
            lot_ck17_col,
            scan_time_col,
            scan_profile_col,
            protocol_gata6_col,
            protocol_ck17_col,
            batch_col,
        ]
        if c in df.columns
    ]
    core_df = df[available].drop_duplicates(subset=[tma_col, core_col])

    tma_list = sorted(core_df[tma_col].dropna().unique())

    def _first_value(sub: pd.DataFrame, col: str, as_date: bool = False) -> str:
        if col not in sub.columns:
            return "—"
        vals = sub[col].dropna()
        if len(vals) == 0:
            return "—"
        if as_date:
            return _fmt_date(vals.iloc[0])
        val = vals.iloc[0]
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)

    def _tma_block(sub: pd.DataFrame, include_dates: bool = True) -> dict:
        n_patients = sub[patient_col].nunique()
        n_cores = len(sub)
        cores_per_pt = sub.groupby(patient_col)[core_col].count()
        n_1 = int((cores_per_pt == 1).sum())
        n_2 = int((cores_per_pt == 2).sum())
        n_3 = int((cores_per_pt >= 3).sum())
        denom = n_patients if n_patients > 0 else 1

        block = {
            "n patients": n_patients,
            "n cores": n_cores,
            "1 core, n (%)": f"{n_1} ({100 * n_1 / denom:.1f}%)",
            "2 cores, n (%)": f"{n_2} ({100 * n_2 / denom:.1f}%)",
            "3 cores, n (%)": f"{n_3} ({100 * n_3 / denom:.1f}%)",
            "Staining date": _first_value(sub, staining_date_col, as_date=True) if include_dates else "—",
            "Scan date": _first_value(sub, scan_date_col, as_date=True) if include_dates else "—",
            "Lot GATA6": _first_value(sub, lot_gata6_col, as_date=False) if include_dates else "—",
            "Lot CK17": _first_value(sub, lot_ck17_col, as_date=False) if include_dates else "—",
            "Scan time": _first_value(sub, scan_time_col, as_date=False) if include_dates else "—",
            "Scan profile": _first_value(sub, scan_profile_col, as_date=False) if include_dates else "—",
            "Protocol GATA6": _first_value(sub, protocol_gata6_col, as_date=False) if include_dates else "—",
            "Protocol CK17": _first_value(sub, protocol_ck17_col, as_date=False) if include_dates else "—",
            "Batch": _first_value(sub, batch_col, as_date=False) if include_dates else "—",
        }
        return block

    result = {}
    for tma in tma_list:
        result[tma] = _tma_block(core_df[core_df[tma_col] == tma], include_dates=True)

    result["Total"] = _tma_block(core_df, include_dates=False)

    table = pd.DataFrame(result)
    out_path = os.path.join(outdir, outname)
    table.to_csv(out_path)
    print(f"TMA metadata table saved → {out_path}")
    return table
