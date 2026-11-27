"""molecular_profile_analysis.py"""

"""
Molecular profile analysis: pathologist grade vs quantitative IHC classification.

Grade vs Molecular Correlation
    Spearman correlation between the 4 molecular populations (+ % GATA6+ and
    % CK17+ totals) and pathologist differentiation grade (ordinal: 1=well,
    2=moderately, 3=poorly differentiated). Tests the lab's directional
    hypothesis: higher %GATA6 -> more differentiated (lower grade); higher
    %CK17 -> less differentiated (higher grade).

Outputs saved to:  <outdir>/
    grade_vs_molecular_correlation.csv
    grade_vs_molecular_correlation.xlsx

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from patient_aggregation import build_patient_pseudobulk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRADE_COL: str = (
    "Differentiation Grade (well=1, moderately=2, poorly=3)"
)

GRADE_CORRELATION_VARS = [
    ("% GATA6+CK17-", "negative", "Classical — most differentiated"),
    ("% GATA6+CK17+", "negative", "Hybrid"),
    ("% GATA6-CK17+", "positive", "Basal"),
    ("% GATA6-CK17-", "positive",
     "Undifferentiated — stromal cells near the tumor can be captured at this "
     "mask dilation; interpret with caution"),
    ("% GATA6+", "negative", "Total GATA6+, regardless of CK17"),
    ("% CK17+", "positive", "Total CK17+, regardless of GATA6"),
]


# ---------------------------------------------------------------------------
# Grade vs Molecular Correlation
# ---------------------------------------------------------------------------

def run_grade_molecular_correlation(
    merged_df: pd.DataFrame,
    outdir: str,
    patient_id_col: str = "ID",
) -> pd.DataFrame:
    """
    Spearman correlation between the 4 molecular populations (+ % GATA6+ and
    % CK17+ totals) and pathologist differentiation grade (ordinal: 1=well,
    2=moderately, 3=poorly differentiated).

    Tests the lab's directional hypothesis with a single-number statistic:
    higher %GATA6 -> more differentiated (lower grade); higher %CK17 -> less
    differentiated (higher grade). Spearman (not Pearson) because grade is
    ordinal, not interval.

    Builds the patient-level pseudo-bulk itself from merged_df (core-level
    output of run_clinical_multivariate_exploration) via build_patient_pseudobulk.

    Returns the correlation table (also saved as CSV + XLSX under outdir).
    """
    os.makedirs(outdir, exist_ok=True)

    patient_df = build_patient_pseudobulk(merged_df, patient_id_col=patient_id_col)
    df = patient_df.dropna(subset=[GRADE_COL]).copy()
    df[GRADE_COL] = pd.to_numeric(df[GRADE_COL], errors="coerce")
    df = df.dropna(subset=[GRADE_COL])

    rows = []
    for col, expected, note in GRADE_CORRELATION_VARS:
        if col not in df.columns:
            continue
        sub = df[[col, GRADE_COL]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        n = len(sub)
        if n < 3:
            rho, p, observed, matches = np.nan, np.nan, "n/a", None
        else:
            rho, p = stats.spearmanr(sub[col], sub[GRADE_COL])
            observed = "negative" if rho < 0 else ("positive" if rho > 0 else "none")
            matches = observed == expected
        rows.append({
            "Variable": col,
            "Description": note,
            "n": n,
            "Spearman rho": round(float(rho), 3) if n >= 3 else np.nan,
            "p-value": round(float(p), 4) if n >= 3 else np.nan,
            "Expected direction": expected,
            "Observed direction": observed,
            "Matches hypothesis": matches,
        })

    result_df = pd.DataFrame(rows)

    csv_path = os.path.join(outdir, "grade_vs_molecular_correlation.csv")
    result_df.to_csv(csv_path, index=False)
    xlsx_path = os.path.join(outdir, "grade_vs_molecular_correlation.xlsx")
    result_df.to_excel(xlsx_path, index=False)

    print(f"\nGrade vs molecular — Spearman correlation (n={len(df)} patients with grade):")
    for _, row in result_df.iterrows():
        flag = "OK" if row["Matches hypothesis"] else ("MISMATCH" if row["Matches hypothesis"] is False else "—")
        print(
            f"  {row['Variable']:<16} rho={row['Spearman rho']:+.3f}  p={row['p-value']:.4f}  "
            f"expected={row['Expected direction']:<8} observed={row['Observed direction']:<8} {flag}"
        )
    print(f"  Table saved → {csv_path}")

    return result_df
