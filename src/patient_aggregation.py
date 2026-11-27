"""patient_aggregation.py"""

"""
Patient-level pseudo-bulk aggregation: collapses the core-level merged table
to one row per patient, averaging the image-derived population percentages
across a patient's cores and carrying over the differentiation grade used by
the grade-vs-molecular-profile correlation.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — column names used by build_patient_pseudobulk
# ---------------------------------------------------------------------------

POPULATION_COLS: List[str] = [
    "% GATA6+CK17+",
    "% GATA6+CK17-",
    "% GATA6-CK17+",
    "% GATA6-CK17-",
]

GRADE_COL: str = "Differentiation Grade (well=1, moderately=2, poorly=3)"


# ---------------------------------------------------------------------------
# Pseudo-bulk aggregation
# ---------------------------------------------------------------------------

def build_patient_pseudobulk(
    merged_df: pd.DataFrame,
    patient_id_col: str = "ID",
) -> pd.DataFrame:
    """
    Aggregate the core-level merged DataFrame to one row per patient.

    Image columns (POPULATION_COLS): mean across cores per patient.
    GRADE_COL: first non-null value per patient (constant across cores).

    Derived columns added:
        % GATA6+ = % GATA6+CK17+ + % GATA6+CK17-
        % CK17+  = % GATA6+CK17+ + % GATA6-CK17+
    """
    df = merged_df.copy()

    # Coerce image columns to numeric
    pop_available = [c for c in POPULATION_COLS if c in df.columns]
    for col in pop_available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate image columns: mean per patient
    img_agg = (
        df.groupby(patient_id_col)[pop_available]
        .mean()
        .reset_index()
    )

    # Grade: first non-null per patient
    available_clin = [c for c in [GRADE_COL] if c in df.columns]
    clin_agg = (
        df.groupby(patient_id_col)[available_clin]
        .first()
        .reset_index()
    )

    patient_df = img_agg.merge(clin_agg, on=patient_id_col, how="inner")

    for col in available_clin:
        patient_df[col] = pd.to_numeric(patient_df[col], errors="coerce")

    # % GATA6+ and % CK17+
    if "% GATA6+CK17+" in patient_df.columns and "% GATA6+CK17-" in patient_df.columns:
        patient_df["% GATA6+"] = patient_df["% GATA6+CK17+"] + patient_df["% GATA6+CK17-"]
    if "% GATA6+CK17+" in patient_df.columns and "% GATA6-CK17+" in patient_df.columns:
        patient_df["% CK17+"] = patient_df["% GATA6+CK17+"] + patient_df["% GATA6-CK17+"]

    print(f"  Pseudo-bulk: {len(patient_df)} patients.")
    return patient_df
