# clinical.py
"""
Clinical × image-derived exploratory analysis.

This module implements a reproducible workflow to:
1) Load clinical data and drop rows with missing IDs.
2) Left-join clinical data (left) with image-derived validation results (right) by ID, keeping
   clinical IDs without image material (image-derived columns set to NaN).
3) Report and remove clinical samples with no image-derived material after merging.
4) Run basic EDA (info, describe, missingness table).
5) Apply dtype conversions using predefined column lists.
6) Generate multivariate exploratory plots:
   - Numeric vs numeric: correlation heatmap
   - Categorical vs categorical: countplot grid (with hue)
   - Numeric vs categorical: pairplot (with hue, chunked)
7) Generate an interactive Plotly HTML dashboard to visualize image-derived variables vs OS by treatment.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend suitable for servers / headless environments (must be set before pyplot).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Defaults (selected image-derived variables)
# ---------------------------

ARTEFACTUAL_COLS = ["Stage Group (I=1, IIa=2, IIb=3, III-IV=4)"]

DEFAULT_IMAGE_VARS = [
    "% positive cell GATA6",
    "% positive cell CK17",
    "% GATA6+CK17+",
    "% GATA6+CK17-",
    "% GATA6-CK17+",
    "% GATA6-CK17-",
    "H-Score GATA6",
    "H-Score CK17",
    "Covariance area",
    "Spatial variance",
    "Heterogeneity_local",
    "Heterogeneity_global",
]

HETEROGENEITY_ORDER = ["Low", "Medium", "High", "VeryHigh"]

META_COLS = ["TMA", "Core", "Patient", "ID", "NHC"]

# ---------------------------
# Clinical typing lists (kept as provided; do not rename)
# ---------------------------

DATE_COLS = [
    "Date of Birth",
    "Date of Diagnosis",
    "Adjuvant Therapy Start Date (after Surgery)",
    "Date of Relapse",
    "Date of Last Follow-up/Death",
]

NUMERIC_COLS = [
    "Age at Diagnosis (years)",
    "Weight (kg)",
    "Height (cm)",
    "BMI",
    "% Weight Loss at Diagnosis",
    "Maximum Diameter (cm)",
    "No. Affected Lymph Nodes",
    "No. Resected Lymph Nodes",
    "CA19.9 at Diagnosis",
    "CA19.9 after Surgery",
    "Adjuvant Therapy: Weeks to Start after Surgery",
    "No. Months of Treatment",
    "% Dose Administered",
    "Disease-Free Time from Surgery (months)",
    "Overall Survival (months)",
]

# Pure integer-coded categoricals: Excel stores the codes as floats (1.0, 2.0),
# which _float_to_int_cat converts to int strings ("1", "2") so category labels
# don't show up as "1.0". Any non-numeric entry is coerced to NaN.
INT_ENCODED_CAT_COLS = {
    "Sex (female=1, male=2)",
    "ECOG PS",
    "Diabetes (yes=1, no=2)",
    "Smoking (yes=1, no=2)",
    "Alcohol Consumption (0=none, 1=mild, 2=moderate-severe)",
    "Chronic Pancreatitis (yes=1, no=2)",
    "Family History of Pancreatic Cancer (yes=1, no=2)",
    "Jaundice at Diagnosis (yes=1, no=2)",
    "Primary Tumor Location (head=1, body=2, tail=3, multiple=4)",
    "Differentiation Grade (well=1, moderately=2, poorly=3)",
    "Lymphovascular Invasion (yes=1, no=2)",
    "Perineural Invasion (yes=1, no=2)",
    "Resection Type (R1=1, R0=2)",
    "pT",
    "pN",
    "Stage Group (I=1, IIa=2, IIb=3, III-IV=4)",
    "Surgery (Whipple=1, Total Pancreatectomy=2, Other=3)",
    "Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)",
    "RT Administered (yes=1, no=2)",
    "Relapse (yes=1, no=2)",
    "Vital Status (dead=1, alive=2)",
}

# Mixed categoricals: mostly integer codes but may also contain valid text
# entries (e.g. "NR", "RECHAZA EL TRATAMIENTO") that must be kept as their
# own category rather than converted to NaN.
MIXED_INT_CAT_COLS = {
    "Pain at Diagnosis (VAS 0-10)",
    "Asthenia (VAS 0-10)",
    "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)",
}

CATEGORICAL_COLS = [
    # Binary / coded (float→int handled in clean_clinical_data)
    "Sex (female=1, male=2)",
    "ECOG PS",
    "Diabetes (yes=1, no=2)",
    "Smoking (yes=1, no=2)",
    "Alcohol Consumption (0=none, 1=mild, 2=moderate-severe)",
    "Pain at Diagnosis (VAS 0-10)",
    "Asthenia (VAS 0-10)",
    "Chronic Pancreatitis (yes=1, no=2)",
    "Family History of Pancreatic Cancer (yes=1, no=2)",
    "Familial Pancreatic Cancer Mutation Type",
    "Jaundice at Diagnosis (yes=1, no=2)",
    "Primary Tumor Location (head=1, body=2, tail=3, multiple=4)",
    "Differentiation Grade (well=1, moderately=2, poorly=3)",
    "Lymphovascular Invasion (yes=1, no=2)",
    "Perineural Invasion (yes=1, no=2)",
    "Resection Type (R1=1, R0=2)",
    "pT",
    "pN",
    "Stage (TNM)",
    "Stage Group (I=1, IIa=2, IIb=3, III-IV=4)",
    "Surgery (Whipple=1, Total Pancreatectomy=2, Other=3)",
    "Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)",
    "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)",
    "RT Administered (yes=1, no=2)",
    "Relapse (yes=1, no=2)",
    "Relapse Type (local=1, distant=2)",
    "Site of Metastasis at Relapse (1=none, 2=liver, 3=lung, 4=peritoneum, 5=bone, 6=two or more of the above, 7=other)",
    "Vital Status (dead=1, alive=2)",
    "Cause of Death (progression=1, toxicity=2, other=3)",
    "Molecular Study Available (yes=1, no=2)",
    # "Segundos tumores..." excluded: contains free-text clinical notes
]


# ---------------------------
# Utilities
# ---------------------------


def _safe_mkdir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.

    Returns
    -------
    None
    """
    if path:
        os.makedirs(path, exist_ok=True)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric, handling comma decimal separators.

    Spanish-locale Excel files store decimals with commas (e.g. "74,00").
    When those cells are read as strings, pd.to_numeric fails silently and
    returns NaN.  This function replaces commas with dots first.

    Parameters
    ----------
    s : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Numeric series (float), with non-convertible values set to NaN.
    """
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _float_to_int_cat(s: pd.Series) -> pd.Series:
    """
    Convert a float-encoded categorical series to integer-string categories.

    Excel reads integer codes (1, 2, 3) as floats (1.0, 2.0, 3.0), which
    results in ugly category labels like "1.0".  This function converts
    whole-number floats to their integer string representation ("1", "2", "3")
    and leaves NaN as NaN before calling astype("category").

    Non-numeric values (e.g. free text) are coerced to NaN.

    Parameters
    ----------
    s : pandas.Series
        Input series with float-encoded integer codes.

    Returns
    -------
    pandas.Series
        Categorical series with clean integer-string labels.
    """
    numeric = pd.to_numeric(s, errors="coerce")

    def _fmt(x):
        if pd.isna(x):
            return pd.NA
        return str(int(round(x)))

    return numeric.map(_fmt).astype("category")


def _mixed_int_cat(s: pd.Series) -> pd.Series:
    """
    Convert a mixed series to categorical, preserving non-numeric text entries.

    Like _float_to_int_cat, but instead of coercing unrecognised text to NaN,
    it keeps the original stripped string as its own category label.

    Use this for columns that are *mostly* integer-coded but may contain
    valid free-text entries (e.g. "NR", "RECHAZA EL TRATAMIENTO").

    Parameters
    ----------
    s : pandas.Series
        Input series with integer codes and/or text values.

    Returns
    -------
    pandas.Series
        Categorical series: numeric floats become int strings ("1", "2", ...),
        text values are kept as-is (stripped), NaN stays NaN.
    """
    def _fmt(val):
        if pd.isna(val):
            return pd.NA
        s_str = str(val).strip().replace(",", ".")
        try:
            f = float(s_str)
            return str(int(round(f)))
        except (ValueError, TypeError):
            return s_str  # preserve text as its own category

    return s.map(_fmt).astype("category")


def encode_heterogeneity_labels(
    s: pd.Series,
    order: List[str] = HETEROGENEITY_ORDER,
) -> pd.Series:
    """
    Encode ordinal heterogeneity labels into numeric ranks (0..k-1).

    This is intended for correlation / modeling where ordinal ranks are needed.

    Parameters
    ----------
    s : pandas.Series
        Series containing ordinal labels (e.g., Low/Medium/High/VeryHigh).
    order : list of str, optional
        Ordered list of label levels from low to high.

    Returns
    -------
    pandas.Series
        Numeric-encoded series where unknown labels are mapped to NaN.
    """
    if s is None:
        return s
    s2 = s.astype(str).str.strip()
    mapping = {lab: i for i, lab in enumerate(order)}
    out = s2.map(mapping)
    return pd.to_numeric(out, errors="coerce")


# ---------------------------
# Loading + cleaning
# ---------------------------


def load_clinical_data(clinical_path: str) -> pd.DataFrame:
    """
    Load a clinical file (xlsx/xls/csv).

    Parameters
    ----------
    clinical_path : str
        Path to the clinical file.

    Returns
    -------
    pandas.DataFrame
        Loaded clinical dataframe.
    """
    if clinical_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(clinical_path)
    return pd.read_csv(clinical_path)


def drop_rows_with_nan_id(clin_df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """
    Drop rows with NaN in the ID column.

    If the specified `id_col` is not present, the first column is used as a fallback.

    Parameters
    ----------
    clin_df : pandas.DataFrame
        Raw clinical dataframe.
    id_col : str, optional
        ID column name.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe containing only rows with non-null IDs.
    """
    if id_col not in clin_df.columns:
        # Fallback to first column as ID if "ID" is missing
        id_col = clin_df.columns[0]
    before = len(clin_df)
    df = clin_df[clin_df[id_col].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN in ID column '{id_col}'.")
    return df


def clean_clinical_data(
    clin_df: pd.DataFrame,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Clean and type the clinical dataframe using predefined column lists.

    Notes
    -----
    - This function does not drop columns to avoid accidentally removing the ID column.

    Parameters
    ----------
    clin_df : pandas.DataFrame
        Clinical dataframe after dropping rows with missing IDs.
    id_col : str, optional
        ID column name.

    Returns
    -------
    pandas.DataFrame
        Cleaned clinical dataframe with converted dtypes.
    """
    df = clin_df.copy()

    # Ensure the merge key is a string for stable merges
    if id_col in df.columns:
        df[id_col] = df[id_col].astype("string").str.strip()

    # Optional legacy column: keep typed if present (may be dropped later after merging)
    if "Pathology Accession Number" in df.columns:
        df["Pathology Accession Number"] = df["Pathology Accession Number"].astype("string")

    # Dates
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Numeric (handles comma decimal separators from Spanish-locale Excel)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])

    # --- Pre-cleaning for specific categorical columns ---

    # Tipo de resección: value 0 is a data-entry error → NaN
    reseccion_col = "Resection Type (R1=1, R0=2)"
    if reseccion_col in df.columns:
        num = pd.to_numeric(df[reseccion_col], errors="coerce")
        df[reseccion_col] = num.where(num != 0, other=pd.NA)

    # Tipo de recaída: normalise free-text variants of combined relapse
    # "1y2", "1 y 2", "1 Y 2", "1Y2" → "1Y2"
    recaida_col = "Relapse Type (local=1, distant=2)"
    if recaida_col in df.columns:
        def _norm_recaida(val):
            if pd.isna(val):
                return pd.NA
            # Collapse spaces and capitalise: "1 y 2", "1y2", "1 Y 2" → "1Y2"
            s_norm = str(val).strip().upper().replace(" ", "")
            if s_norm == "1Y2":
                return "1Y2"
            return str(val).strip()
        df[recaida_col] = df[recaida_col].map(_norm_recaida)

    # Lugar de metástasis: map text synonyms to their numeric code
    lugar_col = (
        "Site of Metastasis at Relapse (1=none, 2=liver, 3=lung, 4=peritoneum, "
        "5=bone, 6=two or more of the above, 7=other)"
    )
    _LUGAR_TEXT_MAP = {
        "pulmón": "3",
        "pulmon": "3",
        "hígado": "2",
        "higado": "2",
        "peritoneo": "4",
        "ósea": "5",
        "osea": "5",
        "otros": "7",
    }
    if lugar_col in df.columns:
        def _norm_lugar(val):
            if pd.isna(val):
                return pd.NA
            s = str(val).strip()
            s_lower = s.lower()
            if s_lower in _LUGAR_TEXT_MAP:
                return _LUGAR_TEXT_MAP[s_lower]
            # Normalise combined-value separators ("2 Y 3", "2y3" → "2Y3")
            normalised = s.upper().replace(" ", "")
            return normalised
        df[lugar_col] = df[lugar_col].map(_norm_lugar)

    # Causa exitus: date-like values are data-entry errors → NaN; keep 1, 2, 3
    causa_col = "Cause of Death (progression=1, toxicity=2, other=3)"
    if causa_col in df.columns:
        def _clean_causa(val):
            if pd.isna(val):
                return pd.NA
            s = str(val).strip()
            # Accept only single-digit codes 1, 2, 3 (possibly as float "1.0")
            try:
                n = int(round(float(s.replace(",", "."))))
                if n in {1, 2, 3}:
                    return str(n)
            except (ValueError, OverflowError):
                pass
            return pd.NA  # dates, notes, or other noise
        df[causa_col] = df[causa_col].map(_clean_causa)

    # Standardise Stage (TNM) capitalisation (e.g. "IIb" → "IIB")
    estadio_col = "Stage (TNM)"
    if estadio_col in df.columns:
        df[estadio_col] = (
            df[estadio_col].astype(str).str.strip().str.upper().replace("NAN", pd.NA)
        )

    # --- Categorical typing ---
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            continue
        if c in INT_ENCODED_CAT_COLS:
            # Pure integer-coded: non-numeric values → NaN
            df[c] = _float_to_int_cat(df[c])
        elif c in MIXED_INT_CAT_COLS:
            # Mixed: integer codes become int strings; text kept as own category
            df[c] = _mixed_int_cat(df[c])
        else:
            df[c] = df[c].astype(str).where(df[c].notna(), other=pd.NA).astype("category")

    return df


def prepare_validation_results_for_clinical(
    validation_results: pd.DataFrame,
    id_col: str = "ID",
    image_vars: List[str] = DEFAULT_IMAGE_VARS,
) -> pd.DataFrame:
    """
    Prepare `validation_results` for merging with clinical data.

    This function:
    - Ensures the merge key exists and is cast to string.
    - Encodes heterogeneity labels into ordinal numeric ranks.
    - Coerces image-derived variables to numeric where appropriate.
    - Keeps only metadata columns (if available) plus the requested image variables.

    Parameters
    ----------
    validation_results : pandas.DataFrame
        Core-level image-derived dataframe.
    id_col : str, optional
        Column name used as merge key.
    image_vars : list of str, optional
        Image-derived variable names to keep and prepare.

    Returns
    -------
    pandas.DataFrame
        Prepared dataframe containing meta columns (if present) plus image-derived variables.

    Raises
    ------
    ValueError
        If `id_col` is missing or if any requested `image_vars` are missing.
    """
    df = validation_results.copy()

    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' column not found in validation_results.")

    missing = [c for c in image_vars if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing image-derived columns in validation_results: {missing}"
        )

    # Ensure merge key is string
    df[id_col] = df[id_col].astype("string").str.strip()

    # Encode ordinal heterogeneity labels (keep numeric)
    if "Heterogeneity_local" in df.columns:
        df["Heterogeneity_local"] = encode_heterogeneity_labels(
            df["Heterogeneity_local"]
        )
    if "Heterogeneity_global" in df.columns:
        df["Heterogeneity_global"] = encode_heterogeneity_labels(
            df["Heterogeneity_global"]
        )

    # Coerce numeric for other image vars
    for c in image_vars:
        if c in ["Heterogeneity_local", "Heterogeneity_global"]:
            continue
        df[c] = _coerce_numeric(df[c])

    # Keep only meta columns that exist + image vars (prevents duplicating clinical cols in merge)
    keep_cols = [c for c in META_COLS if c in df.columns] + image_vars
    keep_cols = list(dict.fromkeys(keep_cols))  # Preserve order, avoid duplicates
    return df[keep_cols].copy()


# ---------------------------
# Merge + checks
# ---------------------------


def merge_clinical_with_results_left(
    clin_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Left-join clinical data with validation results by ID.

    Clinical IDs absent in `validation_results` are kept, with image-derived columns set to NaN.

    Parameters
    ----------
    clin_df : pandas.DataFrame
        Cleaned clinical dataframe (left).
    val_df : pandas.DataFrame
        Prepared validation dataframe (right).
    id_col : str, optional
        Merge key column name.

    Returns
    -------
    pandas.DataFrame
        Merged dataframe.

    Raises
    ------
    ValueError
        If `id_col` is not present in either dataframe.
    """
    if id_col not in clin_df.columns:
        raise ValueError(f"'{id_col}' not found in clinical dataframe.")
    if id_col not in val_df.columns:
        raise ValueError(f"'{id_col}' not found in validation dataframe.")

    left = clin_df.copy()
    right = val_df.copy()

    left[id_col] = left[id_col].astype("string").str.strip()
    right[id_col] = right[id_col].astype("string").str.strip()

    merged = pd.merge(left, right, on=id_col, how="left")
    return merged


def report_missing_image_material(
    merged_df: pd.DataFrame,
    id_col: str,
    image_vars: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify and remove samples with no image-derived material after merging.

    A sample is considered missing image material if all columns in `image_vars` are NaN.
    These samples are reported and removed from the merged dataframe.

    The function also prints simple summary counts by TMA and by treatment (if columns exist).

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Output of the left-merge.
    id_col : str
        ID column name.
    image_vars : list of str
        Image-derived columns to check for missingness.

    Returns
    -------
    missing_ids : list of str
        IDs for which all image-derived columns are NaN.
    merged_df_filtered : pandas.DataFrame
        Merged dataframe with rows removed where all image-derived columns are NaN.
    """
    # Rows where ALL image-derived columns are NaN
    mask = merged_df[image_vars].isna().all(axis=1)
    missing_ids = merged_df.loc[mask, id_col].astype(str).tolist()

    if missing_ids:
        print("\nClinical samples dropped due to missing image material:")
        for _id in missing_ids:
            print(f" - ID={_id}")
    else:
        print(
            "\nNo missing image material detected after merge (all clinical IDs have image-derived data)."
        )

    # Remove rows with all image-derived columns as NaN
    merged_df = merged_df.drop(index=merged_df[mask].index).copy()

    # Print number of cases by TMA
    print("\n--- Case counts by TMA ---")
    if "TMA" in merged_df.columns:
        tma_counts = (
            merged_df.groupby("TMA")[id_col]
            .nunique()
            .rename("n_cases")
            .sort_values(ascending=False)
            .to_frame()
        )
        print(tma_counts)
    else:
        print("Column 'TMA' not found in merged dataframe.")

    # Print number of patients per treatment (if available)
    treatment_col = "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)"
    if treatment_col in merged_df.columns:
        print("\n--- Patient counts by treatment ---")
        trt_counts = (
            merged_df.groupby(treatment_col)[id_col]
            .nunique()
            .rename("n_patients")
            .sort_values(ascending=False)
            .to_frame()
        )
        print(trt_counts)
    
    # Print total number of patients
    total_patients = merged_df[id_col].nunique()
    print(f"\nTotal patients with image-derived data after merge: {total_patients}")

    return missing_ids, merged_df


# ---------------------------
# EDA
# ---------------------------


def eda_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a missingness table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with missing counts and missing percentage per column.
    """
    missing_count = df.isnull().sum()
    total = df.shape[0]
    missing_percent = (missing_count / total) * 100 if total > 0 else 0
    return pd.DataFrame(
        {"Missing Count": missing_count, "Missing %": missing_percent}
    ).sort_values("Missing %", ascending=False)


def run_eda(df: pd.DataFrame) -> None:
    """
    Run basic EDA: info, describe, and missingness table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    None
    """
    print("\n--- EDA: info() ---")
    print(df.info())

    print("\n--- EDA: describe(include='all') ---")
    desc = df.describe(include="all").T
    print(desc)

    miss = eda_table(df)
    print("\n--- EDA: missingness (% NaNs) ---")
    # Force pandas to print all rows/columns (useful for wide clinical tables).
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        None,
    ):
        print(miss)


# ---------------------------
# Multivariate analysis plots
# ---------------------------


def plot_numeric_correlation_heatmap(
    df: pd.DataFrame,
    outpath: str,
    method: str = "spearman",
    figsize: Tuple[int, int] = (18, 14),
    tick_fontsize: int = 7,
    max_label_len: int = 35,
) -> None:
    """
    Plot a numeric vs numeric correlation heatmap with readable axis labels.

    The heatmap is computed on numeric columns only.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe (merged clinical + image-derived data).
    outpath : str
        Output path for the PNG file.
    method : str, optional
        Correlation method (e.g., "spearman", "pearson").
    figsize : tuple of int, optional
        Figure size (width, height).
    tick_fontsize : int, optional
        Font size for axis tick labels.
    max_label_len : int, optional
        Maximum label length to display (labels are truncated for plotting only).

    Returns
    -------
    None
    """
    _safe_mkdir(os.path.dirname(outpath))

    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        print("Not enough numeric columns to compute a correlation heatmap.")
        return

    corr = num_df.corr(method=method)

    def _shorten(s: str) -> str:
        s = str(s)
        return s if len(s) <= max_label_len else (s[: max_label_len - 1] + "…")

    xlabels = [_shorten(c) for c in corr.columns]
    ylabels = [_shorten(r) for r in corr.index]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=False,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5},
    )

    ax.set_title(f"Numeric vs Numeric correlation heatmap ({method})", pad=12)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(ylabels, rotation=0, va="center", fontsize=tick_fontsize)

    fig.subplots_adjust(left=0.30, bottom=0.30)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def choose_default_hue_cat(df: pd.DataFrame, max_unique: int = 6) -> Optional[str]:
    """
    Choose a reasonable default categorical column to use as hue.

    Strategy:
    - Prefer categorical dtype columns with a limited number of unique values.
    - If none match the constraint, fall back to the first categorical column (if any).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    max_unique : int, optional
        Maximum number of unique values allowed for the chosen hue column.

    Returns
    -------
    str or None
        Column name to use as hue, or None if no categorical columns are available.
    """
    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= max_unique:
            return c
    return cat_cols[0] if cat_cols else None


def plot_categorical_countplots_grid(
    df: pd.DataFrame,
    outpath: str,
    hue_col: str,
    max_cols: int = 12,
    max_levels: int = 15,
    ncols: int = 3,
    figsize_per_row: float = 4.2,
) -> None:
    """
    Plot categorical vs categorical countplots in a grid, using a shared hue.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    outpath : str
        Output path for the PNG file (single figure).
    hue_col : str
        Categorical column used for hue grouping.
    max_cols : int, optional
        Maximum number of categorical columns to include (excluding `hue_col`).
    max_levels : int, optional
        Skip categorical columns with more than this number of unique levels.
    ncols : int, optional
        Number of subplot columns in the grid.
    figsize_per_row : float, optional
        Figure height per subplot row (inches).

    Returns
    -------
    None
    """
    _safe_mkdir(os.path.dirname(outpath))

    if hue_col not in df.columns:
        print(f"Cannot create countplot grid: hue_col '{hue_col}' not found.")
        return

    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    cat_cols = [c for c in cat_cols if c != hue_col]
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_levels]

    if not cat_cols:
        print("No categorical columns available for categorical vs categorical plots.")
        return

    cat_cols = cat_cols[:max_cols]

    n_plots = len(cat_cols)
    nrows = int(np.ceil(n_plots / ncols))
    figsize = (ncols * 6.0, nrows * figsize_per_row)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, x_col in zip(axes, cat_cols):
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)
        ax.set_title(x_col, fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

        # Remove subplot legends to avoid duplication; a shared legend is added below.
        if ax.legend_ is not None:
            ax.legend_.remove()

    for ax in axes[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=hue_col, loc="upper right", bbox_to_anchor=(0.99, 0.99)
    )

    fig.suptitle(f"Categorical vs Categorical countplots (hue={hue_col})", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_vs_categorical_pairplots_chunked(
    df: pd.DataFrame,
    outpath_base: str,
    hue_col: str,
    chunk_size: int = 6,
    height: float = 3.0,
    aspect: float = 1.1,
) -> None:
    """
    Create Seaborn pairplots in chunks to limit the number of numeric variables per figure.

    This function:
    - Selects numeric columns with more than one unique value.
    - Splits them into chunks of size `chunk_size`.
    - Saves one pairplot per chunk to `<outpath_base>_<i>.svg`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    outpath_base : str
        Base output path (without index suffix and extension).
    hue_col : str
        Categorical column used for hue grouping.
    chunk_size : int, optional
        Maximum number of numeric variables per pairplot.
    height : float, optional
        Height of each facet in inches.
    aspect : float, optional
        Aspect ratio of each facet.

    Returns
    -------
    None
    """
    if hue_col not in df.columns:
        print(f"Cannot create pairplots: hue_col '{hue_col}' not found.")
        return

    work = df.copy()
    work[hue_col] = work[hue_col].astype("category")

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if work[c].nunique(dropna=True) > 1]

    chunks = [
        numeric_cols[i : i + chunk_size]
        for i in range(0, len(numeric_cols), chunk_size)
    ]

    for i, cols in enumerate(chunks, start=1):
        if len(cols) < 2:
            continue

        plot_df = work[[hue_col] + cols].dropna(subset=[hue_col])

        g = sns.pairplot(
            plot_df,
            hue=hue_col,
            diag_kind="hist",
            height=height,
            aspect=aspect,
            plot_kws={"alpha": 0.6, "s": 18},
        )
        g.figure.subplots_adjust(top=0.92)
        g.figure.suptitle(
            f"Numeric vs Categorical pairplot (hue={hue_col}) — block {i}",
            y=0.98,
            fontsize=14,
        )

        outpath = f"{outpath_base}_{i}.svg"
        g.figure.savefig(outpath, dpi=300)
        plt.close(g.figure)

        print(f"Saved: {outpath}")


# ---------------------------
# Interactive dashboard (Plotly)
# ---------------------------


def make_optipanc_dashboard_html(
    merged: pd.DataFrame,
    out_html: str,
    id_col: str = "ID",
    patient_col: str = "Patient",
    tma_col: str = "TMA",
    core_col: str = "Core",
    treatment_col: str = "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)",
    os_col: str = "Overall Survival (months)",
) -> None:
    """
    Build an interactive Plotly HTML dashboard (6 rows × 4 columns) with secondary y-axes.

    Layout
    ------
    Columns:
        - One column per treatment group (Gemcitabine, Gem-Cap, FOLFIRINOX, Rejects treatment).
    Rows:
        1) % GATA6+ and % CK17+ (stacked) + OS overlay
        2) 4 GATA6/CK17 populations (stacked) + OS overlay
        3) H-scores (lines) + OS overlay
        4) Covariance area (points) + OS overlay
        5) Spatial variance (points) + OS overlay
        6) Heterogeneity (local/global rank) (points) + OS overlay

    Ordering
    --------
    Within each treatment column, patients are ordered by OS (ascending) to enable vertical comparison.

    Parameters
    ----------
    merged : pandas.DataFrame
        Merged dataframe (clinical + image-derived), core-level.
    out_html : str
        Output HTML file path.
    id_col : str, optional
        ID column name.
    patient_col : str, optional
        Patient column name (if available).
    tma_col : str, optional
        TMA column name.
    core_col : str, optional
        Core column name.
    treatment_col : str, optional
        Treatment column name.
    os_col : str, optional
        Overall survival column name (months).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required columns for the dashboard are missing.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    # Image-derived columns
    col_gata6 = "% positive cell GATA6"
    col_ck17 = "% positive cell CK17"
    pop_cols = ["% GATA6+CK17+", "% GATA6+CK17-", "% GATA6-CK17+", "% GATA6-CK17-"]
    subtype_labels = {
        "% GATA6+CK17+": "GATA6+CK17+ (Hybrid)",
        "% GATA6+CK17-": "GATA6+CK17- (Classical)",
        "% GATA6-CK17+": "GATA6-CK17+ (Basal)",
        "% GATA6-CK17-": "GATA6-CK17- (Undifferentiated)",
    }
    hs_gata6 = "H-Score GATA6"
    hs_ck17 = "H-Score CK17"
    cov_area = "Covariance area"
    spatial_var = "Spatial variance"
    het_local = "Heterogeneity_local"
    het_global = "Heterogeneity_global"

    needed = [
        id_col,
        tma_col,
        core_col,
        treatment_col,
        os_col,
        col_gata6,
        col_ck17,
        hs_gata6,
        hs_ck17,
        cov_area,
        spatial_var,
        het_local,
        het_global,
    ] + pop_cols
    miss = [c for c in needed if c not in merged.columns]
    if miss:
        raise ValueError(f"Missing required columns for dashboard: {miss}")

    df = merged.copy()

    # Numeric conversions
    num_cols = [
        os_col,
        col_gata6,
        col_ck17,
        hs_gata6,
        hs_ck17,
        cov_area,
        spatial_var,
    ] + pop_cols
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize treatment labels (supports 1/2/3 and free-text containing 'RECHAZA')
    trt_raw = df[treatment_col].astype(str).str.strip().str.upper()
    trt_num = pd.to_numeric(trt_raw, errors="coerce")

    def _map_trt_label(raw_upper: str, num_code: float) -> str:
        if isinstance(raw_upper, str) and "RECHAZA" in raw_upper:
            return "RECHAZA EL TRATAMIENTO"
        if pd.notna(num_code):
            if int(num_code) == 1:
                return "Gemcitabina (1)"
            if int(num_code) == 2:
                return "Gem-Cap (2)"
            if int(num_code) == 3:
                return "FOLFIRINOX (3)"
        return raw_upper

    df["_trt_label"] = [
        _map_trt_label(r, n) for r, n in zip(trt_raw.tolist(), trt_num.tolist())
    ]

    # Canonical column order (treatments); the actual panels/col_titles used
    # are determined dynamically below (after get_panel_df is defined), so a
    # treatment arm with no patients in this cohort doesn't get an empty column.
    TRT_ORDER = ["Gemcitabina (1)", "Gem-Cap (2)", "FOLFIRINOX (3)", "RECHAZA EL TRATAMIENTO"]

    # Colors (kept fixed for interpretability across runs)
    COLOR = {
        "GATA6+ (%)": "#1f77b4",
        "CK17+ (%)": "#ff7f0e",
        "OS": "#000000",
        "% GATA6-CK17-": "#808080",
        "% GATA6-CK17+": "#d62728",
        "% GATA6+CK17-": "#2ca02c",
        "% GATA6+CK17+": "#98df8a",
        "H-Score GATA6": "#1f77b4",
        "H-Score CK17": "#ff7f0e",
        "Heterogeneity_local (rank)": "#7f7f7f",
        "Heterogeneity_global (rank)": "#bcbd22",
        "Covariance area": "#17becf",
        "Spatial variance": "#e377c2",
    }

    # Patient-level OS used for ordering within treatment
    df["_OS_patient"] = df.groupby(id_col)[os_col].transform(
        lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan
    )

    # Composite x-axis label (traceability): TMA | ID | Core
    df["_x_label"] = (
        df[tma_col].astype(str).fillna("NA")
        + " | "
        + df[id_col].astype(str).fillna("NA")
        + " | "
        + df[core_col].astype(str).fillna("NA")
    )

    def get_panel_df(trt_label_value: str) -> pd.DataFrame:
        d = df[df["_trt_label"] == trt_label_value].copy()
        d = d.dropna(subset=[id_col, core_col, "_OS_patient", "_x_label"])

        patient_order = (
            d[[id_col, "_OS_patient"]]
            .drop_duplicates()
            .sort_values("_OS_patient", ascending=True)[id_col]
            .tolist()
        )
        rank = {pid: i for i, pid in enumerate(patient_order)}
        d["_rank"] = d[id_col].map(rank)
        d = d.sort_values(by=["_rank", id_col, core_col])

        x_order = d["_x_label"].tolist()
        d["_x"] = pd.Categorical(d["_x_label"], categories=x_order, ordered=True)
        return d

    # A treatment arm only gets a column if at least one patient in this
    # cohort falls into it; e.g. "RECHAZA EL TRATAMIENTO" is skipped rather
    # than rendered as an empty column.
    panels = [trt for trt in TRT_ORDER if not get_panel_df(trt).empty]
    extra_labels = sorted(
        set(df["_trt_label"].dropna().unique()) - set(TRT_ORDER) - {"NAN"}
    )
    panels += [trt for trt in extra_labels if not get_panel_df(trt).empty]
    col_titles = panels

    def make_hover(extra_line: str) -> str:
        # Kept bilingual variable names as they come from the dataset; template is English-friendly.
        return (
            "<b>%{customdata[0]}</b><br>"
            "ID: %{customdata[1]}<br>"
            "Patient: %{customdata[2]}<br>"
            "TMA: %{customdata[3]}<br>"
            "Core: %{customdata[4]}<br>"
            "Treatment: %{customdata[5]}<br>"
            "OS (months): %{customdata[6]:.2f}<br>" + extra_line + "<extra></extra>"
        )

    def build_customdata(d: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                d["_x_label"].astype(str),
                d[id_col].astype(str),
                d.get(patient_col, pd.Series(["NA"] * len(d))).astype(str),
                d[tma_col].astype(str),
                d[core_col].astype(str),
                d["_trt_label"].astype(str),
                d[os_col].to_numpy(),
            ],
            axis=1,
        )

    def add_os_overlay(d: pd.DataFrame, row: int, col: int, showlegend: bool) -> None:
        custom = build_customdata(d)
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[os_col],
                mode="markers+lines",
                name="OS",
                line=dict(color=COLOR["OS"]),
                marker=dict(color=COLOR["OS"]),
                showlegend=showlegend,
                customdata=custom,
                hovertemplate=make_hover(""),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

    # Create subplot grid with secondary y-axis everywhere
    NROWS = 6
    NCOLS = len(panels)
    specs = [[{"secondary_y": True} for _ in range(NCOLS)] for __ in range(NROWS)]

    fig = make_subplots(
        rows=NROWS,
        cols=NCOLS,
        specs=specs,
        column_titles=col_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    # Row titles (left annotations)
    row_titles = [
        "GATA6/CK17 Positive Cells (%) + OS",
        "4 Populations (%) + OS",
        "H-Scores + OS",
        "Covariance area + OS",
        "Spatial variance + OS",
        "Heterogeneity (rank) + OS",
    ]
    for i, title in enumerate(row_titles, start=1):
        fig.add_annotation(
            dict(
                text=f"<b>{title}</b>",
                xref="paper",
                yref="paper",
                x=-0.06,
                y=1 - (i - 0.5) / NROWS,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                align="right",
                font=dict(size=11),
                textangle=-90,
            )
        )

    # Fill each treatment column
    for j, trt in enumerate(panels, start=1):
        d = get_panel_df(trt)
        if d.empty:
            continue

        custom = build_customdata(d)

        # Row 1: % GATA6 / % CK17 (stacked bars) + OS
        fig.add_trace(
            go.Bar(
                x=d["_x"],
                y=d[col_gata6],
                name="GATA6+ (%)",
                marker_color=COLOR["GATA6+ (%)"],
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("GATA6+: %{y:.2f}<br>"),
            ),
            row=1,
            col=j,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=d["_x"],
                y=d[col_ck17],
                name="CK17+ (%)",
                marker_color=COLOR["CK17+ (%)"],
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("CK17+: %{y:.2f}<br>"),
            ),
            row=1,
            col=j,
            secondary_y=False,
        )
        add_os_overlay(d, row=1, col=j, showlegend=(j == 1))

        # Row 2: 4 populations (stacked) + OS
        for pc in pop_cols:
            fig.add_trace(
                go.Bar(
                    x=d["_x"],
                    y=d[pc],
                    name=subtype_labels[pc],
                    marker_color=COLOR[pc],
                    showlegend=(j == 1),
                    customdata=custom,
                    hovertemplate=make_hover(f"{subtype_labels[pc]}: %{{y:.2f}}<br>"),
                ),
                row=2,
                col=j,
                secondary_y=False,
            )
        add_os_overlay(d, row=2, col=j, showlegend=False)

        # Row 3: H-scores (lines) + OS
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[hs_gata6],
                mode="lines+markers",
                name="H-Score GATA6",
                line=dict(color=COLOR["H-Score GATA6"]),
                marker=dict(color=COLOR["H-Score GATA6"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("H-Score GATA6: %{y:.2f}<br>"),
            ),
            row=3,
            col=j,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[hs_ck17],
                mode="lines+markers",
                name="H-Score CK17",
                line=dict(color=COLOR["H-Score CK17"]),
                marker=dict(color=COLOR["H-Score CK17"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("H-Score CK17: %{y:.2f}<br>"),
            ),
            row=3,
            col=j,
            secondary_y=False,
        )
        add_os_overlay(d, row=3, col=j, showlegend=False)

        # Row 4: Covariance area (points) + OS
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[cov_area],
                mode="markers",
                name="Covariance area",
                marker=dict(color=COLOR["Covariance area"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("Covariance area: %{y:.3f}<br>"),
            ),
            row=4,
            col=j,
            secondary_y=False,
        )
        add_os_overlay(d, row=4, col=j, showlegend=False)

        # Row 5: Spatial variance (points) + OS
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[spatial_var],
                mode="markers",
                name="Spatial variance",
                marker=dict(color=COLOR["Spatial variance"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("Spatial variance: %{y:.3f}<br>"),
            ),
            row=5,
            col=j,
            secondary_y=False,
        )
        add_os_overlay(d, row=5, col=j, showlegend=False)

        # Row 6: Heterogeneity ranks (points) + OS
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[het_local],
                mode="markers",
                name="Heterogeneity_local (rank)",
                marker=dict(color=COLOR["Heterogeneity_local (rank)"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("Heterogeneity_local (rank): %{y:.3f}<br>"),
            ),
            row=6,
            col=j,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[het_global],
                mode="markers",
                name="Heterogeneity_global (rank)",
                marker=dict(color=COLOR["Heterogeneity_global (rank)"]),
                showlegend=(j == 1),
                customdata=custom,
                hovertemplate=make_hover("Heterogeneity_global (rank): %{y:.3f}<br>"),
            ),
            row=6,
            col=j,
            secondary_y=False,
        )
        add_os_overlay(d, row=6, col=j, showlegend=False)

    # Figure layout
    fig.update_layout(
        autosize=True,
        height=3200,
        barmode="stack",
        margin=dict(l=5, r=5, t=90, b=120),
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1.0,
            yanchor="top",
            font=dict(size=8),
            itemsizing="constant",
            itemwidth=30,
        ),
        title=(
            "OPTIPANC - Exploratory Dashboard - Image analysis vs. Clinical "
            "(core-level; patients ordered by OS within each treatment)"
        ),
    )

    # Global maxima for consistent scaling
    max_cov = (
        float(np.nanmax(df[cov_area].to_numpy())) if df[cov_area].notna().any() else 1.0
    )
    max_spa = (
        float(np.nanmax(df[spatial_var].to_numpy()))
        if df[spatial_var].notna().any()
        else 1.0
    )
    max_het = (
        float(np.nanmax(pd.concat([df[het_local], df[het_global]], axis=0).to_numpy()))
        if (df[het_local].notna().any() or df[het_global].notna().any())
        else 3.0
    )

    # Axis styling and fixed y ranges (primary axis)
    for r in range(1, NROWS + 1):
        for c in range(1, NCOLS + 1):
            fig.update_xaxes(
                showticklabels=True,
                tickangle=90,
                tickfont=dict(size=7),
                title_font=dict(size=9),
                row=r,
                col=c,
            )

            fig.update_yaxes(
                tickfont=dict(size=7),
                title_font=dict(size=9),
                row=r,
                col=c,
                secondary_y=False,
            )
            fig.update_yaxes(
                tickfont=dict(size=7),
                title_font=dict(size=9),
                row=r,
                col=c,
                secondary_y=True,
            )

            if r == 1:
                fig.update_yaxes(range=[-10, 210], row=r, col=c, secondary_y=False)
            elif r == 2:
                fig.update_yaxes(range=[-10, 110], row=r, col=c, secondary_y=False)
            elif r == 3:
                fig.update_yaxes(range=[-10, 310], row=r, col=c, secondary_y=False)
            elif r == 4:
                fig.update_yaxes(
                    range=[-10, max_cov + 10], row=r, col=c, secondary_y=False
                )
            elif r == 5:
                fig.update_yaxes(
                    range=[-10, max_spa + 10], row=r, col=c, secondary_y=False
                )
            elif r == 6:
                fig.update_yaxes(
                    range=[-10, max_het + 10], row=r, col=c, secondary_y=False
                )

    # Axis titles:
    # - Clear all y titles everywhere
    # - Show primary y (y1) only in column 1
    # - Show secondary y (y2) only in column 4
    for r in range(1, NROWS + 1):
        for c in range(1, NCOLS + 1):
            fig.update_yaxes(title_text="", row=r, col=c, secondary_y=False)
            fig.update_yaxes(title_text="", row=r, col=c, secondary_y=True)

    c = 1
    fig.update_yaxes(title_text="% (stacked)", row=1, col=c, secondary_y=False)
    fig.update_yaxes(title_text="% (stacked)", row=2, col=c, secondary_y=False)
    fig.update_yaxes(title_text="H-score", row=3, col=c, secondary_y=False)
    fig.update_yaxes(title_text="Covariance area", row=4, col=c, secondary_y=False)
    fig.update_yaxes(title_text="Spatial variance", row=5, col=c, secondary_y=False)
    fig.update_yaxes(title_text="Heterogeneity rank", row=6, col=c, secondary_y=False)

    c = 4
    fig.update_yaxes(title_text="OS (months)", row=1, col=c, secondary_y=True)

    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )
    print(f"Saved dashboard: {out_html}")
def make_optipanc_dashboard_html_patient(
    merged: pd.DataFrame,
    out_html: str,
    id_col: str = "ID",
    patient_col: str = "Patient",
    tma_col: str = "TMA",
    treatment_col: str = "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)",
    os_col: str = "Overall Survival (months)",
) -> None:
    """
    Dashboard aggregated at (TMA, ID) level:
    - Grouping key: (TMA, ID)
    - Image vars: mean across cores within each (TMA, ID)
    - OS: first non-null within group (should be constant)
    - Treatment: first non-null within group (should be constant)
    - X axis: "TMA | ID | n=<num_cores>"
    """
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    # --- image cols used in the dashboard
    col_gata6 = "% positive cell GATA6"
    col_ck17 = "% positive cell CK17"
    pop_cols = ["% GATA6+CK17+", "% GATA6+CK17-", "% GATA6-CK17+", "% GATA6-CK17-"]
    subtype_labels = {
        "% GATA6+CK17+": "GATA6+CK17+ (Hybrid)",
        "% GATA6+CK17-": "GATA6+CK17- (Classical)",
        "% GATA6-CK17+": "GATA6-CK17+ (Basal)",
        "% GATA6-CK17-": "GATA6-CK17- (Undifferentiated)",
    }
    hs_gata6 = "H-Score GATA6"
    hs_ck17 = "H-Score CK17"
    cov_area = "Covariance area"
    spatial_var = "Spatial variance"
    het_local = "Heterogeneity_local"
    het_global = "Heterogeneity_global"

    needed = [id_col, tma_col, treatment_col, os_col,
              col_gata6, col_ck17, hs_gata6, hs_ck17,
              cov_area, spatial_var, het_local, het_global] + pop_cols
    miss = [c for c in needed if c not in merged.columns]
    if miss:
        raise ValueError(f"Missing required columns for patient-level dashboard: {miss}")

    df = merged.copy()

    # Ensure grouping keys are clean strings
    df[id_col] = df[id_col].astype("string").str.strip()
    df[tma_col] = df[tma_col].astype("string").str.strip()

    # Numeric conversions for image vars + OS + heterogeneity ranks
    num_cols = [os_col, col_gata6, col_ck17, hs_gata6, hs_ck17,
                cov_area, spatial_var, het_local, het_global] + pop_cols
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize treatment labels (1/2/3 + "RECHAZA")
    trt_raw = df[treatment_col].astype(str).str.strip().str.upper()
    trt_num = pd.to_numeric(trt_raw, errors="coerce")

    def _map_trt_label(raw_upper: str, num_code: float) -> str:
        if isinstance(raw_upper, str) and "RECHAZA" in raw_upper:
            return "RECHAZA EL TRATAMIENTO"
        if pd.notna(num_code):
            if int(num_code) == 1:
                return "Gemcitabina (1)"
            if int(num_code) == 2:
                return "Gem-Cap (2)"
            if int(num_code) == 3:
                return "FOLFIRINOX (3)"
        return raw_upper

    df["_trt_label"] = [_map_trt_label(r, n) for r, n in zip(trt_raw.tolist(), trt_num.tolist())]

    # ---------- aggregation helpers ----------
    def _first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    def _warn_if_not_constant(d: pd.DataFrame, group_cols, check_col: str):
        # Optional: warns if OS or treatment differ inside same (TMA, ID)
        tmp = (d.groupby(group_cols)[check_col].nunique(dropna=True))
        bad = tmp[tmp > 1]
        if len(bad):
            print(f"[WARN] '{check_col}' is not constant within some groups (showing up to 10):")
            print(bad.head(10))

    group_cols = [tma_col, id_col]

    # (Optional but useful) sanity checks
    _warn_if_not_constant(df, group_cols, os_col)
    _warn_if_not_constant(df, group_cols, "_trt_label")

    # --- Count cores (= number of rows in merged per (TMA, ID))
    core_counts = (
        df.groupby(group_cols, dropna=False)
          .size()
          .rename("n_cores")
          .reset_index()
    )

    # Define aggregation:
    # - image vars: mean
    # - OS, treatment, Patient: first non-null (should be constant)
    agg = {c: "mean" for c in [col_gata6, col_ck17, hs_gata6, hs_ck17,
                               cov_area, spatial_var, het_local, het_global] + pop_cols}
    agg[os_col] = _first_non_null
    agg["_trt_label"] = _first_non_null
    if patient_col in df.columns:
        agg[patient_col] = _first_non_null

    dfp = df.groupby(group_cols, dropna=False).agg(agg).reset_index()

    # --- Attach n_cores
    dfp = dfp.merge(core_counts, on=group_cols, how="left")

    # Drop rows that cannot be placed
    dfp = dfp.dropna(subset=[tma_col, id_col, os_col, "_trt_label"])

    # X label: TMA | ID | n=#
    dfp["_x_label"] = (
        dfp[tma_col].astype(str).fillna("NA")
        + " | "
        + dfp[id_col].astype(str).fillna("NA")
        + " | n="
        + dfp["n_cores"].fillna(0).astype(int).astype(str)
    )

    # Canonical panel order; the actual panels/col_titles used are determined
    # dynamically below (after get_panel_df is defined), so a treatment arm
    # with no patients in this cohort doesn't get an empty column.
    TRT_ORDER = ["Gemcitabina (1)", "Gem-Cap (2)", "FOLFIRINOX (3)", "RECHAZA EL TRATAMIENTO"]

    # Colors
    COLOR = {
        "GATA6+ (%)": "#1f77b4",
        "CK17+ (%)": "#ff7f0e",
        "OS": "#000000",
        "% GATA6-CK17-": "#808080",
        "% GATA6-CK17+": "#d62728",
        "% GATA6+CK17-": "#2ca02c",
        "% GATA6+CK17+": "#98df8a",
        "H-Score GATA6": "#1f77b4",
        "H-Score CK17": "#ff7f0e",
        "Heterogeneity_local (rank)": "#7f7f7f",
        "Heterogeneity_global (rank)": "#bcbd22",
        "Covariance area": "#17becf",
        "Spatial variance": "#e377c2",
    }

    # Helper: build ordered df per treatment
    def get_panel_df(trt_label_value: str) -> pd.DataFrame:
        d = dfp[dfp["_trt_label"] == trt_label_value].copy()
        d = d.dropna(subset=[os_col, "_x_label"])
        d = d.sort_values(os_col, ascending=True)  # order by OS within treatment
        x_order = d["_x_label"].tolist()
        d["_x"] = pd.Categorical(d["_x_label"], categories=x_order, ordered=True)
        return d

    # A treatment arm only gets a column if at least one patient in this
    # cohort falls into it; e.g. "RECHAZA EL TRATAMIENTO" is skipped rather
    # than rendered as an empty column.
    panels = [trt for trt in TRT_ORDER if not get_panel_df(trt).empty]
    extra_labels = sorted(
        set(dfp["_trt_label"].dropna().unique()) - set(TRT_ORDER) - {"NAN"}
    )
    panels += [trt for trt in extra_labels if not get_panel_df(trt).empty]
    col_titles = panels

    # Hover
    def make_hover(extra_line: str) -> str:
        return (
            "<b>%{customdata[0]}</b><br>"
            "ID: %{customdata[1]}<br>"
            "Patient: %{customdata[2]}<br>"
            "TMA: %{customdata[3]}<br>"
            "Treatment: %{customdata[4]}<br>"
            "OS (months): %{customdata[5]:.2f}<br>"
            "n_cores: %{customdata[6]}<br>"
            + extra_line +
            "<extra></extra>"
        )

    def build_customdata(d: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                d["_x_label"].astype(str),
                d[id_col].astype(str),
                d.get(patient_col, pd.Series(["NA"] * len(d))).astype(str),
                d[tma_col].astype(str),
                d["_trt_label"].astype(str),
                d[os_col].to_numpy(),
                d["n_cores"].to_numpy(),
            ],
            axis=1,
        )

    def add_os_overlay(d: pd.DataFrame, row: int, col: int, showlegend: bool) -> None:
        custom = build_customdata(d)
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d[os_col],
                mode="markers+lines",
                name="OS",
                line=dict(color=COLOR["OS"]),
                marker=dict(color=COLOR["OS"]),
                showlegend=showlegend,
                customdata=custom,
                hovertemplate=make_hover(""),
            ),
            row=row, col=col, secondary_y=True
        )

    # --- subplots
    NROWS, NCOLS = 6, len(panels)
    specs = [[{"secondary_y": True} for _ in range(NCOLS)] for __ in range(NROWS)]

    fig = make_subplots(
        rows=NROWS, cols=NCOLS,
        specs=specs,
        column_titles=col_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
    )

    # Row titles
    row_titles = [
        "GATA6/CK17 Positive Cells (%) + OS (mean cores)",
        "4 Populations (%) + OS (mean cores)",
        "H-Scores + OS (mean cores)",
        "Covariance area + OS (mean cores)",
        "Spatial variance + OS (mean cores)",
        "Heterogeneity (rank) + OS (mean cores)",
    ]
    for i, title in enumerate(row_titles, start=1):
        fig.add_annotation(
            dict(
                text=f"<b>{title}</b>",
                xref="paper", yref="paper",
                x=-0.06, y=1 - (i - 0.5) / NROWS,
                showarrow=False,
                xanchor="right", yanchor="middle",
                align="right",
                font=dict(size=11),
                textangle=-90,
            )
        )

    # --- fill each treatment column
    for j, trt in enumerate(panels, start=1):
        d = get_panel_df(trt)
        if d.empty:
            continue
        custom = build_customdata(d)

        # Row 1
        fig.add_trace(go.Bar(
            x=d["_x"], y=d[col_gata6], name="GATA6+ (%)",
            marker_color=COLOR["GATA6+ (%)"],
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("GATA6+: %{y:.2f}<br>"),
        ), row=1, col=j, secondary_y=False)

        fig.add_trace(go.Bar(
            x=d["_x"], y=d[col_ck17], name="CK17+ (%)",
            marker_color=COLOR["CK17+ (%)"],
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("CK17+: %{y:.2f}<br>"),
        ), row=1, col=j, secondary_y=False)

        add_os_overlay(d, row=1, col=j, showlegend=(j == 1))

        # Row 2
        for pc in pop_cols:
            fig.add_trace(go.Bar(
                x=d["_x"], y=d[pc], name=subtype_labels[pc],
                marker_color=COLOR[pc],
                showlegend=(j == 1), customdata=custom,
                hovertemplate=make_hover(f"{subtype_labels[pc]}: %{{y:.2f}}<br>"),
            ), row=2, col=j, secondary_y=False)

        add_os_overlay(d, row=2, col=j, showlegend=False)

        # Row 3
        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[hs_gata6],
            mode="lines+markers", name="H-Score GATA6",
            line=dict(color=COLOR["H-Score GATA6"]),
            marker=dict(color=COLOR["H-Score GATA6"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("H-Score GATA6: %{y:.2f}<br>"),
        ), row=3, col=j, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[hs_ck17],
            mode="lines+markers", name="H-Score CK17",
            line=dict(color=COLOR["H-Score CK17"]),
            marker=dict(color=COLOR["H-Score CK17"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("H-Score CK17: %{y:.2f}<br>"),
        ), row=3, col=j, secondary_y=False)

        add_os_overlay(d, row=3, col=j, showlegend=False)

        # Row 4
        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[cov_area],
            mode="markers", name="Covariance area",
            marker=dict(color=COLOR["Covariance area"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("Covariance area: %{y:.3f}<br>"),
        ), row=4, col=j, secondary_y=False)

        add_os_overlay(d, row=4, col=j, showlegend=False)

        # Row 5
        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[spatial_var],
            mode="markers", name="Spatial variance",
            marker=dict(color=COLOR["Spatial variance"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("Spatial variance: %{y:.3f}<br>"),
        ), row=5, col=j, secondary_y=False)

        add_os_overlay(d, row=5, col=j, showlegend=False)

        # Row 6
        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[het_local],
            mode="markers", name="Heterogeneity_local (rank)",
            marker=dict(color=COLOR["Heterogeneity_local (rank)"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("Heterogeneity_local (rank): %{y:.3f}<br>"),
        ), row=6, col=j, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=d["_x"], y=d[het_global],
            mode="markers", name="Heterogeneity_global (rank)",
            marker=dict(color=COLOR["Heterogeneity_global (rank)"]),
            showlegend=(j == 1), customdata=custom,
            hovertemplate=make_hover("Heterogeneity_global (rank): %{y:.3f}<br>"),
        ), row=6, col=j, secondary_y=False)

        add_os_overlay(d, row=6, col=j, showlegend=False)

    # Layout
    fig.update_layout(
        autosize=True,
        height=3200,
        barmode="stack",
        margin=dict(l=90, r=70, t=90, b=120),
        legend=dict(
            orientation="v",
            x=1.01, xanchor="left",
            y=1.0, yanchor="top",
            font=dict(size=8),
            itemsizing="constant",
            itemwidth=30,
        ),
        title=(
            "OPTIPANC - Exploratory Dashboard - Image analysis vs. Clinical "
            "(patient-level = mean across cores; grouped by TMA & ID; ordered by OS within each treatment)"
        ),
    )

    # Scaling
    max_cov = float(np.nanmax(dfp[cov_area].to_numpy())) if dfp[cov_area].notna().any() else 1.0
    max_spa = float(np.nanmax(dfp[spatial_var].to_numpy())) if dfp[spatial_var].notna().any() else 1.0
    max_het = float(np.nanmax(pd.concat([dfp[het_local], dfp[het_global]], axis=0).to_numpy())) \
        if (dfp[het_local].notna().any() or dfp[het_global].notna().any()) else 3.0

    for r in range(1, NROWS + 1):
        for c in range(1, NCOLS + 1):
            fig.update_xaxes(showticklabels=True, tickangle=90, tickfont=dict(size=8), row=r, col=c)
            fig.update_yaxes(tickfont=dict(size=7), row=r, col=c, secondary_y=False)
            fig.update_yaxes(tickfont=dict(size=7), row=r, col=c, secondary_y=True)

            if r == 1:
                fig.update_yaxes(range=[0, 200], row=r, col=c, secondary_y=False)
            elif r == 2:
                fig.update_yaxes(range=[0, 100], row=r, col=c, secondary_y=False)
            elif r == 3:
                fig.update_yaxes(range=[0, 300], row=r, col=c, secondary_y=False)
            elif r == 4:
                fig.update_yaxes(range=[0, max_cov], row=r, col=c, secondary_y=False)
            elif r == 5:
                fig.update_yaxes(range=[0, max_spa], row=r, col=c, secondary_y=False)
            elif r == 6:
                fig.update_yaxes(range=[0, max_het], row=r, col=c, secondary_y=False)

    # Axis titles: show primary only in col 1; OS only in col 4 (row 1)
    for r in range(1, NROWS + 1):
        for c in range(1, NCOLS + 1):
            fig.update_yaxes(title_text="", row=r, col=c, secondary_y=False)
            fig.update_yaxes(title_text="", row=r, col=c, secondary_y=True)

    c1 = 1
    fig.update_yaxes(title_text="% (stacked)", row=1, col=c1, secondary_y=False)
    fig.update_yaxes(title_text="% (stacked)", row=2, col=c1, secondary_y=False)
    fig.update_yaxes(title_text="H-score", row=3, col=c1, secondary_y=False)
    fig.update_yaxes(title_text="Covariance area", row=4, col=c1, secondary_y=False)
    fig.update_yaxes(title_text="Spatial variance", row=5, col=c1, secondary_y=False)
    fig.update_yaxes(title_text="Heterogeneity rank", row=6, col=c1, secondary_y=False)

    c4 = 4
    fig.update_yaxes(title_text="OS (months)", row=1, col=c4, secondary_y=True)

    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )
    print(f"Saved patient-level dashboard: {out_html}")

# ---------------------------
# High-level runner
# ---------------------------


def run_clinical_multivariate_exploration(
    validation_results: pd.DataFrame,
    clinical_path: str,
    outdir: str,
    id_col: str = "ID",
    image_vars: List[str] = DEFAULT_IMAGE_VARS,
    hue_cat: Optional[str] = None,
    pairplot_numeric_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete clinical × image-derived exploratory workflow.

    Workflow
    --------
    1) Load clinical and drop rows with missing IDs.
    2) Apply dtype conversions using predefined column lists.
    3) Prepare validation results (image-derived metrics) for merging.
    4) Left-join clinical data with validation results by ID.
    5) Optionally drop legacy key column 'Pathology Accession Number' after merge (if present).
    6) Report and remove samples missing all image-derived variables.
    7) Save merged table to CSV.
    8) Run EDA and generate exploratory plots.
    9) Generate the Plotly HTML dashboard.

    Parameters
    ----------
    validation_results : pandas.DataFrame
        Dataframe containing image-derived metrics and an ID column.
    clinical_path : str
        Path to the clinical file (Excel/CSV).
    outdir : str
        Output directory for plots, tables, and dashboard.
    id_col : str, optional
        ID column name used for merging in both datasets.
    image_vars : list of str, optional
        Image-derived columns to prepare and check for missing material.
    hue_cat : str or None, optional
        Categorical column name to use as hue. If None, a default is auto-selected.
    pairplot_numeric_cols : list of str or None, optional
        Reserved for future use (explicit numeric columns list for pairplots).

    Returns
    -------
    dict of str -> pandas.DataFrame
        Dictionary containing:
            - "clinical_clean": typed clinical dataframe
            - "validation_prepared": prepared validation results dataframe
            - "merged": merged dataframe used for downstream analysis
    """
    _safe_mkdir(outdir)

    # 1) Load clinical + drop NaN IDs
    clin_raw = load_clinical_data(clinical_path)
    clin_raw = drop_rows_with_nan_id(clin_raw, id_col=id_col)

    # 2) Apply dtypes (hardcoded lists preserved)
    clin_clean = clean_clinical_data(clin_raw, id_col=id_col)

    # 3) Prepare validation
    val_prep = prepare_validation_results_for_clinical(
        validation_results=validation_results,
        id_col=id_col,
        image_vars=image_vars,
    )

    # 5) Left merge by ID
    merged = merge_clinical_with_results_left(
        clin_df=clin_clean,
        val_df=val_prep,
        id_col=id_col,
    )

    # 6) Drop legacy key if present
    if "Pathology Accession Number" in merged.columns and "Patient" in merged.columns:
        merged = merged.drop(columns=["Pathology Accession Number"])

    # 7) Report missing image material and filter
    _, merged = report_missing_image_material(
        merged_df=merged,
        id_col=id_col,
        image_vars=image_vars,
    )

    # Ensure meta columns are string
    for c in META_COLS:
        if c in merged.columns:
            merged[c] = merged[c].astype("string").str.strip()

    # Save merged table (never persist patient-identifying fields to output/)
    merged.drop(
        columns=[c for c in ("NHC", "Date of Birth") if c in merged.columns]
    ).to_csv(os.path.join(outdir, "clinical_image_merged.csv"), index=False)

    # 8) EDA
    run_eda(merged)

    # 9) Drop artefactual columns if present
    for col in ARTEFACTUAL_COLS:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    # 10) Multivariate analysis outputs
    plots_dir = os.path.join(outdir, "plots")
    _safe_mkdir(plots_dir)

    plot_numeric_correlation_heatmap(
        df=merged,
        outpath=os.path.join(plots_dir, "numeric_correlation_heatmap_spearman.svg"),
        method="spearman",
    )

    if hue_cat is None:
        hue_cat = choose_default_hue_cat(merged)

    if hue_cat is None:
        print("No categorical column found for hue. Skipping categorical-based plots.")
    else:
        if hue_cat in merged.columns and str(merged[hue_cat].dtype) != "category":
            merged[hue_cat] = merged[hue_cat].astype("category")

        plot_categorical_countplots_grid(
            df=merged,
            outpath=os.path.join(plots_dir, f"countplots_grid_hue_{hue_cat[:40]}.svg"),
            hue_col=hue_cat,
            max_cols=12,
            ncols=3,
        )

        plot_numeric_vs_categorical_pairplots_chunked(
            df=merged,
            outpath_base=os.path.join(plots_dir, f"pairplot_numeric_vs_{hue_cat[:40]}"),
            hue_col=hue_cat,
            chunk_size=6,
            height=3.0,
            aspect=1.1,
        )

    # Dashboard at core level
    dashboard_path = os.path.join(outdir, "optipanc_dashboard_core_level.html")
    make_optipanc_dashboard_html(
        merged=merged,
        out_html=dashboard_path,
    )

    # Dashboard at patient level
    dashboard_patient_path = os.path.join(
        outdir, "optipanc_dashboard_patient_level.html"
    )
    make_optipanc_dashboard_html_patient(
        merged=merged,
        out_html=dashboard_patient_path,
    )

    return {
        "clinical_clean": clin_clean,
        "validation_prepared": val_prep,
        "merged": merged,
    }
