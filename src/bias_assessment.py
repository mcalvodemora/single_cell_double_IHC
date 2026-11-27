"""
bias_assessment.py

Compares baseline clinical characteristics between an "included" and an
"excluded" group of patients, to assess whether sample loss introduced
selection bias. Which patients land in each group depends on `cohort`
(see load_and_split):

  cohort="complete":
    - Included: patients NOT listed in OPTIPANC_ExcludedPatients.xlsx
    - Excluded: patients listed there (insufficient material, non-evaluable
      cores, etc.)

  cohort="clean":
    - Included: patients present in the final publication roster
      (Patient_TMA_Core_Clean.xlsx)
    - Excluded: everyone else in the clinical database

Output:
  - <outdir>/bias_assessment_table.xlsx
  - <outdir>/bias_assessment_table.csv

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Loading & splitting
# ---------------------------------------------------------------------------

def load_and_split(
    clinical_path: str,
    excluded_path: str,
    included_path: str,
    cohort: str = "clean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (included, excluded) DataFrames for the requested cohort definition.

    IDs are always compared with whitespace stripped, since the clinical DB has
    at least one entry with a stray internal space (e.g. "HUIS_ 004").

    Args:
        cohort: "complete" -> included = clinical DB minus excluded_path roster.
                "clean"    -> included = final publication roster (included_path).
        clinical_path: path to the clinical Excel (see main.py's config block).
        excluded_path: cohort="complete" only — roster of excluded patients.
        included_path: cohort="clean" only — roster of the final publication cohort.
    """
    if cohort not in ("complete", "clean"):
        raise ValueError(f"cohort must be 'complete' or 'clean', got {cohort!r}")

    clin = pd.read_excel(clinical_path)
    clin_ids_norm = clin["ID"].astype(str).str.replace(r"\s+", "", regex=True)

    if cohort == "complete":
        excl = pd.read_excel(excluded_path)
        excl_ids = set(
            excl["Patient ID"].astype(str).str.replace(r"\s+", "", regex=True).unique()
        )

        included = clin[~clin_ids_norm.isin(excl_ids)].copy()
        excluded = clin[clin_ids_norm.isin(excl_ids)].copy()

        print(f"[cohort=complete] Total in clinical DB : {len(clin)}")
        print(f"Excluded IDs listed  : {len(excl_ids)}")
        print(f"  Found in clinical  : {len(excluded)}")
        missing = excl_ids - set(clin_ids_norm)
        if missing:
            print(f"  NOT found (skipped): {sorted(missing)}")
        print(f"Included             : {len(included)}")
        return included, excluded

    # cohort == "clean"
    roster = pd.read_excel(included_path)
    roster_ids = set(
        roster["Patient ID"].astype(str).str.replace(r"\s+", "", regex=True).unique()
    )

    included = clin[clin_ids_norm.isin(roster_ids)].copy()
    excluded = clin[~clin_ids_norm.isin(roster_ids)].copy()

    print(f"[cohort=clean] Total in clinical DB      : {len(clin)}")
    print(f"Final TMA cohort patients : {len(roster_ids)}")
    print(f"  Found in clinical       : {len(included)}")
    missing = roster_ids - set(clin_ids_norm)
    if missing:
        print(f"  NOT found (skipped)    : {sorted(missing)}")
    print(f"Excluded                  : {len(excluded)}")
    return included, excluded


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def clean_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Continuous ---
    df["Maximum Diameter (cm)"] = _to_numeric(df["Maximum Diameter (cm)"])
    df["BMI"] = _to_numeric(df["BMI"])
    df["% Weight Loss at Diagnosis"] = _to_numeric(
        df["% Weight Loss at Diagnosis"]
        # "NR" and "Pérdida ponderal no cuantificada" → NaN via coerce
    )

    # --- pT: entry "2.3" in HCB-09 is a data-entry error (diameter pasted); → NaN ---
    df.loc[df["pT"] == 2.3, "pT"] = np.nan

    # --- Resección: entry "0" in HUPrinc1 is uninterpretable; → NaN ---
    df.loc[df["Resection Type (R1=1, R0=2)"] == 0, "Resection Type (R1=1, R0=2)"] = np.nan

    # --- Pancreatitis / Antecedentes: isolated "0" entries (2 and 1 occurrences)
    #     are likely data-entry errors (coding is 1=yes, 2=no); → NaN ---
    df.loc[df["Chronic Pancreatitis (yes=1, no=2)"] == 0, "Chronic Pancreatitis (yes=1, no=2)"] = np.nan
    df.loc[
        df["Family History of Pancreatic Cancer (yes=1, no=2)"] == 0,
        "Family History of Pancreatic Cancer (yes=1, no=2)",
    ] = np.nan

    # --- Ictericia: "0" appears 17 times across hospitals, consistent with
    #     centres using 0=no instead of 2=no; treat 0 as 2 (no ictericia) ---
    df.loc[df["Jaundice at Diagnosis (yes=1, no=2)"] == 0, "Jaundice at Diagnosis (yes=1, no=2)"] = 2.0

    # --- Localización: single "0" entry; → NaN ---
    df.loc[
        df["Primary Tumor Location (head=1, body=2, tail=3, multiple=4)"] == 0,
        "Primary Tumor Location (head=1, body=2, tail=3, multiple=4)",
    ] = np.nan

    # --- NPMI: harmonise text coding to numeric (1=yes, 2=no) ---
    npmi_col = "Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)"
    df[npmi_col] = df[npmi_col].astype(object).map(
        lambda v: 1 if str(v).strip().lower() == "sí"
        else (2 if str(v).strip().lower() == "no" else v)
    )
    df[npmi_col] = pd.to_numeric(df[npmi_col], errors="coerce")

    # --- Tratamiento: "RECHAZA EL TRATAMIENTO" is not a valid treatment arm; → NaN ---
    tto_col = "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)"
    df[tto_col] = df[tto_col].astype(object).map(
        lambda v: np.nan if str(v).strip() == "RECHAZA EL TRATAMIENTO" else v
    )
    df[tto_col] = pd.to_numeric(df[tto_col], errors="coerce")

    # --- Estadio TNM: harmonise mixed capitalisation to upper-case standard ---
    stage_map = {"Ia": "IA", "Ib": "IB", "IIa": "IIA", "IIb": "IIB"}
    df["Stage (TNM)"] = df["Stage (TNM)"].replace(stage_map)

    # --- EVA scales stored as object (mixed int/float entries); convert to numeric ---
    df["Pain at Diagnosis (VAS 0-10)"] = _to_numeric(df["Pain at Diagnosis (VAS 0-10)"])
    df["Asthenia (VAS 0-10)"] = _to_numeric(df["Asthenia (VAS 0-10)"])

    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _mann_whitney(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return p


def _chi2_or_fisher(inc: pd.Series, exc: pd.Series) -> Tuple[float, str]:
    """Return (p-value, test_name). Uses Fisher's exact for 2×2, chi-square otherwise."""
    all_cats = sorted(set(inc.dropna().unique()) | set(exc.dropna().unique()))
    counts_inc = [inc[inc == c].shape[0] for c in all_cats]
    counts_exc = [exc[exc == c].shape[0] for c in all_cats]
    table = np.array([counts_inc, counts_exc])

    if table.shape[1] == 2:
        _, p = fisher_exact(table)
        return p, "Fisher's exact"
    else:
        # Chi-square; flag if any expected count < 5
        chi2, p, _, expected = chi2_contingency(table, correction=False)
        test = "Chi-square"
        if (expected < 5).any():
            test = "Chi-square (expected<5)"
        return p, test


# ---------------------------------------------------------------------------
# Summary formatters
# ---------------------------------------------------------------------------

def _fmt_median_iqr(series: pd.Series, decimals: int = 1) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "—"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.{decimals}f} ({q1:.{decimals}f}–{q3:.{decimals}f})"


def _fmt_n_pct(series: pd.Series, cat, n_total: int) -> str:
    n = (series == cat).sum()
    pct = 100 * n / n_total if n_total > 0 else 0
    return f"{n} ({pct:.1f}%)"


def _fmt_pvalue(p) -> str:
    """Format a p-value (or BH-adjusted p-value) for figure display: '<0.001' or 3 decimals."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _add_row(rows: list, variable: str, category: str,
             val_inc: str, val_exc: str, p: float, test: str):
    rows.append({
        "Variable": variable,
        "Category": category,
        "Included": val_inc,
        "Excluded": val_exc,
        "p-value": round(p, 3) if not np.isnan(p) else np.nan,
        "Test": test,
    })


def build_table(included: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_inc = len(included)
    n_exc = len(excluded)

    def cont(col, label, decimals=1):
        p = _mann_whitney(included[col], excluded[col])
        _add_row(rows, label, "median (IQR)",
                 _fmt_median_iqr(included[col], decimals),
                 _fmt_median_iqr(excluded[col], decimals),
                 p, "Mann-Whitney U")

    def cat(col, label, cat_labels: dict):
        p, test = _chi2_or_fisher(included[col], excluded[col])
        _add_row(rows, label, "", "", "", p, test)
        for code, cat_label in cat_labels.items():
            _add_row(rows, "", f"  {cat_label}",
                     _fmt_n_pct(included[col], code, n_inc),
                     _fmt_n_pct(excluded[col], code, n_exc),
                     np.nan, "")

    # ── Patient characteristics ──────────────────────────────────────────
    cont("Age at Diagnosis (years)", "Age at diagnosis (years)")
    cat("Sex (female=1, male=2)", "Sex",
        {1.0: "Female", 2.0: "Male"})
    cat("ECOG PS", "ECOG performance status",
        {0.0: "ECOG 0", 1.0: "ECOG 1", 2.0: "ECOG 2", 3.0: "ECOG 3"})
    cont("BMI", "BMI (kg/m²)")
    cat("Diabetes (yes=1, no=2)", "Diabetes",
        {1.0: "Yes", 2.0: "No"})
    cat("Smoking (yes=1, no=2)", "Tobacco use",
        {1.0: "Yes", 2.0: "No"})
    cat("Alcohol Consumption (0=none, 1=mild, 2=moderate-severe)",
        "Alcohol consumption",
        {0.0: "None", 1.0: "Mild", 2.0: "Moderate–severe", 3.0: "Severe"})
    cat("Chronic Pancreatitis (yes=1, no=2)", "Chronic pancreatitis",
        {1.0: "Yes", 2.0: "No"})
    cat("Family History of Pancreatic Cancer (yes=1, no=2)",
        "Familial pancreatic cancer history",
        {1.0: "Yes", 2.0: "No"})
    cont("Pain at Diagnosis (VAS 0-10)", "Pain at diagnosis (VAS 0–10)", decimals=0)
    cat("Jaundice at Diagnosis (yes=1, no=2)", "Jaundice at diagnosis",
        {1.0: "Yes", 2.0: "No"})
    cont("Asthenia (VAS 0-10)", "Asthenia (VAS 0–10)", decimals=0)
    cont("% Weight Loss at Diagnosis", "Weight loss at diagnosis (%)")

    # ── Tumour characteristics ───────────────────────────────────────────
    cat("Primary Tumor Location (head=1, body=2, tail=3, multiple=4)",
        "Tumour location",
        {1.0: "Head", 2.0: "Body", 3.0: "Tail", 4.0: "Multiple"})
    cont("Maximum Diameter (cm)", "Maximum tumour diameter (cm)")
    cat("pT", "pT stage",
        {1.0: "T1", 2.0: "T2", 3.0: "T3", 4.0: "T4"})
    cat("pN", "pN stage",
        {0.0: "N0", 1.0: "N1", 2.0: "N2"})
    cat("Stage (TNM)", "TNM stage",
        {"IA": "IA", "IB": "IB", "IIA": "IIA", "IIB": "IIB", "III": "III"})
    cat("Differentiation Grade (well=1, moderately=2, poorly=3)",
        "Histological grade",
        {1.0: "Well diff.", 2.0: "Moderately diff.", 3.0: "Poorly diff."})
    cat("Lymphovascular Invasion (yes=1, no=2)", "Lymphovascular invasion",
        {1.0: "Yes", 2.0: "No"})
    cat("Perineural Invasion (yes=1, no=2)", "Perineural invasion",
        {1.0: "Yes", 2.0: "No"})
    cat("Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)", "IPMN",
        {1.0: "Yes", 2.0: "No"})
    cont("No. Resected Lymph Nodes", "Lymph nodes resected (n)", decimals=0)
    cont("No. Affected Lymph Nodes", "Lymph nodes affected (n)", decimals=0)

    # ── Treatment ────────────────────────────────────────────────────────
    cat("Surgery (Whipple=1, Total Pancreatectomy=2, Other=3)", "Surgery type",
        {1.0: "Whipple", 2.0: "Total pancreatectomy", 3.0: "Other"})
    cat("Resection Type (R1=1, R0=2)", "Resection margin",
        {1.0: "R1", 2.0: "R0"})
    cat("Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)", "Adjuvant treatment",
        {1.0: "Gemcitabine", 2.0: "Gem-Cap", 3.0: "FOLFIRINOX"})
    cat("RT Administered (yes=1, no=2)", "Radiotherapy",
        {1.0: "Yes", 2.0: "No"})

    # ── Outcomes ─────────────────────────────────────────────────────────
    cat("Relapse (yes=1, no=2)", "Relapse",
        {1.0: "Yes", 2.0: "No"})
    cont("Overall Survival (months)", "Overall survival (months)")
    cont("Disease-Free Time from Surgery (months)", "Disease-free survival (months)")
    cat("Vital Status (dead=1, alive=2)", "Vital status",
        {1.0: "Deceased", 2.0: "Alive"})

    df = pd.DataFrame(rows)

    # BH correction on all tested variables (rows with a p-value — continuous headers
    # and categorical headers; subcategory rows carry NaN p-values and are excluded)
    mask = df["p-value"].notna()
    pvals = df.loc[mask, "p-value"].values
    if len(pvals) > 0:
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        df.loc[mask, "p-adj (BH)"] = np.round(adj, 3)
    else:
        df["p-adj (BH)"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_table(df: pd.DataFrame, n_inc: int, n_exc: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    xlsx_path = os.path.join(outdir, "bias_assessment_table.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Bias assessment")
        ws = writer.sheets["Bias assessment"]

        # Header row: include sample sizes
        ws["C1"] = f"Included (n={n_inc})"
        ws["D1"] = f"Excluded (n={n_exc})"

        # Column widths
        ws.column_dimensions["A"].width = 45
        ws.column_dimensions["B"].width = 25
        ws.column_dimensions["C"].width = 22
        ws.column_dimensions["D"].width = 22
        ws.column_dimensions["E"].width = 10
        ws.column_dimensions["F"].width = 28
        ws.column_dimensions["G"].width = 14

        # Bold header
        from openpyxl.styles import Font, PatternFill, Alignment
        header_font = Font(bold=True)
        header_fill = PatternFill("solid", fgColor="D9E1F2")
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        # Highlight significant p-values (p < 0.05) in yellow
        sig_fill = PatternFill("solid", fgColor="FFEB9C")
        for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
            for cell in row:
                try:
                    if cell.value is not None and float(cell.value) < 0.05:
                        cell.fill = sig_fill
                except (ValueError, TypeError):
                    pass

    print(f"Saved: {xlsx_path}")


# ---------------------------------------------------------------------------
# Supplementary figure panel: nominally significant variables only
# ---------------------------------------------------------------------------

_SIG_HEADER_GREEN = "#57CC99"
_SIG_HEADER_GRAY = "#BFBFBF"
_SIG_HEADER_LIGHT = "#E6E6E6"
_SIG_BORDER = "#808080"


def plot_nominal_significant_table(
    table: pd.DataFrame,
    n_inc: int,
    n_exc: int,
    outdir: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Render a supplementary-figure table restricted to variables with a nominal
    (unadjusted) p-value below `alpha`, keeping all subcategory rows for the
    selected categorical variables.

    This is a read-only, additive view of the table already produced by
    build_table(): it does not recompute any statistic, test or adjustment.

    Returns the filtered table (same columns as `table`), which is also the
    content saved to "bias_nominal_significant_table.csv".
    """
    os.makedirs(outdir, exist_ok=True)

    table = table.reset_index(drop=True)
    header_idx = table.index[table["p-value"].notna()].tolist()

    # Each header row (continuous variable, or categorical variable header)
    # owns every row up to (but excluding) the next header row.
    groups = []
    for i, idx in enumerate(header_idx):
        end = header_idx[i + 1] if i + 1 < len(header_idx) else len(table)
        groups.append(table.iloc[idx:end])

    selected_groups = [g for g in groups if g.iloc[0]["p-value"] < alpha]
    selected = (
        pd.concat(selected_groups, ignore_index=True)
        if selected_groups else table.iloc[0:0].copy()
    )

    csv_path = os.path.join(outdir, "bias_nominal_significant_table.csv")
    selected.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if not selected_groups:
        print(f"[plot_nominal_significant_table] No variables with nominal p < {alpha}; "
              "skipping figure.")
        return selected

    # --- Figure rendering --------------------------------------------------
    col_labels = ["Variable", "Category", f"Included (n={n_inc})",
                  f"Excluded (n={n_exc})", "p-value", "Test", "p-adj (BH)"]
    col_widths = np.array([0.21, 0.19, 0.15, 0.15, 0.09, 0.13, 0.08])
    col_widths = col_widths / col_widths.sum()
    col_x = np.concatenate([[0.0], np.cumsum(col_widths)])
    header_colors = [_SIG_HEADER_LIGHT, _SIG_HEADER_LIGHT, _SIG_HEADER_GREEN,
                      _SIG_HEADER_GRAY, _SIG_HEADER_LIGHT, _SIG_HEADER_LIGHT, _SIG_HEADER_LIGHT]

    # One (header_row, visual_rows) pair per selected variable. visual_rows are
    # the rows that get their own line in the figure: subcategory rows for a
    # categorical variable, or the single row itself for a continuous one.
    render_groups = []
    for g in selected_groups:
        header = g.iloc[0]
        subrows = g.iloc[1:]
        visual = subrows.reset_index(drop=True) if len(subrows) > 0 else g.iloc[[0]].reset_index(drop=True)
        render_groups.append((header, visual))

    row_h = 1.0
    header_h = 1.3
    total_rows = sum(len(v) for _, v in render_groups)

    fig_w = 12.0
    fig_h = 0.9 * (header_h + total_rows * row_h) / 1.6  # compact inches-per-row scale
    fig, ax = plt.subplots(figsize=(fig_w, max(fig_h, 2.0)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, header_h + total_rows * row_h)
    ax.invert_yaxis()
    ax.axis("off")

    # Header row
    for j, label in enumerate(col_labels):
        ax.add_patch(plt.Rectangle((col_x[j], 0.0), col_widths[j], header_h,
                                    facecolor=header_colors[j], edgecolor=_SIG_BORDER, linewidth=0.8))
        ax.text((col_x[j] + col_x[j + 1]) / 2, header_h / 2, label,
                 ha="center", va="center", fontsize=9, fontweight="bold", fontfamily="sans-serif")

    # Data rows: merged columns (Variable, p-value, Test, p-adj) span the full
    # block of a variable; Category/Included/Excluded get one line per subrow.
    y = header_h
    for header, visual in render_groups:
        n_sub = len(visual)
        block_h = row_h * n_sub

        merged_cols = {
            0: str(header["Variable"]) if pd.notna(header["Variable"]) and str(header["Variable"]).strip() else "",
            4: _fmt_pvalue(header["p-value"]),
            5: str(header["Test"]) if pd.notna(header["Test"]) and str(header["Test"]).strip() else "",
            6: _fmt_pvalue(header["p-adj (BH)"]),
        }
        for j, text in merged_cols.items():
            ax.add_patch(plt.Rectangle((col_x[j], y), col_widths[j], block_h,
                                        facecolor="white", edgecolor=_SIG_BORDER, linewidth=0.8))
            ax.text((col_x[j] + col_x[j + 1]) / 2, y + block_h / 2, text,
                     ha="center", va="center", fontsize=8.5,
                     fontweight="bold" if j == 0 else "normal", fontfamily="sans-serif")

        for k in range(n_sub):
            ry = y + row_h * k
            row = visual.iloc[k]
            for j, key in ((1, "Category"), (2, "Included"), (3, "Excluded")):
                ax.add_patch(plt.Rectangle((col_x[j], ry), col_widths[j], row_h,
                                            facecolor="white", edgecolor=_SIG_BORDER, linewidth=0.6))
                val = row[key]
                text = "" if pd.isna(val) else str(val)
                ax.text((col_x[j] + col_x[j + 1]) / 2, ry + row_h / 2, text,
                         ha="center", va="center", fontsize=8.5, fontfamily="sans-serif")
        y += block_h

    fig.tight_layout()
    svg_path = os.path.join(outdir, "bias_nominal_significant_table.svg")
    png_path = os.path.join(outdir, "bias_nominal_significant_table.png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bias_assessment(
    outdir: str,
    clinical_path: str,
    excluded_path: str,
    included_path: str,
    cohort: str = "clean",
):
    print(f"=== Bias Assessment: Excluded vs Included patients (cohort={cohort}) ===\n")
    included_raw, excluded_raw = load_and_split(
        cohort=cohort,
        clinical_path=clinical_path,
        excluded_path=excluded_path,
        included_path=included_path,
    )

    included = clean_variables(included_raw)
    excluded = clean_variables(excluded_raw)

    print(f"\nBuilding comparison table...")
    table = build_table(included, excluded)

    # Print to console
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 140)
    sig = table[table["p-value"] < 0.05]
    print(f"\nVariables with p < 0.05 (raw): {len(sig)}")
    if len(sig):
        print(sig[["Variable", "p-value", "p-adj (BH)"]].to_string(index=False))
    else:
        print("  None — no significant differences detected.")

    export_table(table, n_inc=len(included), n_exc=len(excluded), outdir=outdir)
    plot_nominal_significant_table(table, n_inc=len(included), n_exc=len(excluded), outdir=outdir)
    print("\nDone.")
    return table


# ---------------------------------------------------------------------------
# Descriptive comparison: total clinical cohort vs final TMA cohort
# ---------------------------------------------------------------------------

COLOR_TOTAL    = "#FFB703"   # amber / yellow  → total clinical cohort
COLOR_INCLUDED = "#57CC99"   # green           → final TMA cohort

# Continuous variables: (raw column name, display label, decimal places)
_CONT: List[Tuple[str, str, int]] = [
    ("Age at Diagnosis (years)",                        "Age (years)",         1),
    ("BMI",                                               "BMI (kg/m²)",         1),
    ("Pain at Diagnosis (VAS 0-10)",                   "Pain VAS (0–10)",     0),
    ("Asthenia (VAS 0-10)",                                "Asthenia VAS (0–10)", 0),
    ("% Weight Loss at Diagnosis",                  "Weight loss (%)",     1),
    ("Maximum Diameter (cm)",                              "Max diameter (cm)",   1),
    ("No. Resected Lymph Nodes",                             "LN resected (n)",     0),
    ("No. Affected Lymph Nodes",                             "LN affected (n)",     0),
    ("Overall Survival (months)",                      "OS (months)",         1),
    ("Disease-Free Time from Surgery (months)",     "DFS (months)",        1),
]

# Categorical variables: (raw column, display label, {code: label} ordered dict)
_CAT: List[Tuple[str, str, Dict]] = [
    ("Sex (female=1, male=2)",                             "Sex",
     {1.0: "Female", 2.0: "Male"}),
    ("ECOG PS",                                           "ECOG PS",
     {0.0: "0", 1.0: "1", 2.0: "2", 3.0: "3"}),
    ("Diabetes (yes=1, no=2)",                              "Diabetes",
     {1.0: "Yes", 2.0: "No"}),
    ("Smoking (yes=1, no=2)",                             "Tobacco use",
     {1.0: "Yes", 2.0: "No"}),
    ("Alcohol Consumption (0=none, 1=mild, 2=moderate-severe)", "Alcohol",
     {0.0: "None", 1.0: "Mild", 2.0: "Mod–severe"}),
    ("Chronic Pancreatitis (yes=1, no=2)",                    "Chronic pancreatitis",
     {1.0: "Yes", 2.0: "No"}),
    ("Family History of Pancreatic Cancer (yes=1, no=2)",   "Familial Pc history",
     {1.0: "Yes", 2.0: "No"}),
    ("Jaundice at Diagnosis (yes=1, no=2)",                "Jaundice",
     {1.0: "Yes", 2.0: "No"}),
    ("Primary Tumor Location (head=1, body=2, tail=3, multiple=4)", "Tumour location",
     {1.0: "Head", 2.0: "Body", 3.0: "Tail", 4.0: "Multiple"}),
    ("pT",                                                "pT stage",
     {1.0: "T1", 2.0: "T2", 3.0: "T3", 4.0: "T4"}),
    ("pN",                                                "pN stage",
     {0.0: "N0", 1.0: "N1", 2.0: "N2"}),
    ("Stage (TNM)",                                     "TNM stage",
     {"IA": "IA", "IB": "IB", "IIA": "IIA", "IIB": "IIB", "III": "III"}),
    ("Differentiation Grade (well=1, moderately=2, poorly=3)", "Histological grade",
     {1.0: "Well", 2.0: "Moderate", 3.0: "Poor"}),
    ("Lymphovascular Invasion (yes=1, no=2)",             "LVI",
     {1.0: "Yes", 2.0: "No"}),
    ("Perineural Invasion (yes=1, no=2)",                "PNI",
     {1.0: "Yes", 2.0: "No"}),
    ("Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)", "IPMN",
     {1.0: "Yes", 2.0: "No"}),
    ("Surgery (Whipple=1, Total Pancreatectomy=2, Other=3)", "Surgery type",
     {1.0: "Whipple", 2.0: "Total pancreatectomy", 3.0: "Other"}),
    ("Resection Type (R1=1, R0=2)",                      "Resection margin",
     {1.0: "R1", 2.0: "R0"}),
    ("Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)", "Adjuvant treatment",
     {1.0: "Gemcitabine", 2.0: "Gem-Cap", 3.0: "FOLFIRINOX"}),
    ("RT Administered (yes=1, no=2)",                        "Radiotherapy",
     {1.0: "Yes", 2.0: "No"}),
    ("Relapse (yes=1, no=2)",                                 "Relapse",
     {1.0: "Yes", 2.0: "No"}),
    ("Vital Status (dead=1, alive=2)",                                   "Vital status",
     {1.0: "Deceased", 2.0: "Alive"}),
]


def build_descriptive_table(
    total_df: pd.DataFrame, included_df: pd.DataFrame
) -> pd.DataFrame:
    """Descriptive table: total cohort vs Analyzed. No statistical tests."""
    rows = []
    n_tot = len(total_df)
    n_inc = len(included_df)

    def cont(col, label, dec):
        tot = _to_numeric(total_df[col]).dropna()
        inc = _to_numeric(included_df[col]).dropna()
        rows.append({
            "Variable": label,
            "Category": "median (IQR)",
            f"Total (n={n_tot})": _fmt_median_iqr(tot, dec),
            f"Analyzed (n={n_inc})": _fmt_median_iqr(inc, dec),
        })

    def cat(col, label, cat_labels):
        rows.append({"Variable": label, "Category": "",
                     f"Total (n={n_tot})": "", f"Analyzed (n={n_inc})": ""})
        for code, cat_label in cat_labels.items():
            rows.append({
                "Variable": "",
                "Category": f"  {cat_label}",
                f"Total (n={n_tot})": _fmt_n_pct(total_df[col], code, n_tot),
                f"Analyzed (n={n_inc})": _fmt_n_pct(included_df[col], code, n_inc),
            })

    for col, label, dec in _CONT:
        cont(col, label, dec)
    for col, label, cat_labels in _CAT:
        cat(col, label, cat_labels)

    return pd.DataFrame(rows)


def _export_descriptive_table(df: pd.DataFrame, n_tot: int, n_inc: int, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, "descriptive_total_vs_analyzed.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Total vs Analyzed")
        ws = writer.sheets["Total vs Analyzed"]
        from openpyxl.styles import Font, PatternFill, Alignment
        header_font = Font(bold=True)
        header_fill = PatternFill("solid", fgColor="D9E1F2")
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions["A"].width = 35
        ws.column_dimensions["B"].width = 22
        ws.column_dimensions["C"].width = 22
        ws.column_dimensions["D"].width = 22
    print(f"Saved: {xlsx_path}")


def plot_continuous_comparison(
    total_df: pd.DataFrame, included_df: pd.DataFrame, outdir: str
) -> None:
    """Grid of boxplots: total cohort (yellow) vs Analyzed (green)."""
    n_vars = len(_CONT)
    n_cols = 5
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    bp_props = dict(patch_artist=True, medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=1), capprops=dict(lw=1),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for i, (col, label, dec) in enumerate(_CONT):
        ax = axes[i]
        tot_vals = _to_numeric(total_df[col]).dropna().values
        inc_vals = _to_numeric(included_df[col]).dropna().values

        bp = ax.boxplot(
            [tot_vals, inc_vals],
            labels=[f"Total\n(n={len(tot_vals)})", f"Analyzed\n(n={len(inc_vals)})"],
            **bp_props,
        )
        bp["boxes"][0].set_facecolor(COLOR_TOTAL)
        bp["boxes"][1].set_facecolor(COLOR_INCLUDED)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_TOTAL,    alpha=0.7, label=f"Total cohort (n={len(total_df)})"),
        mpatches.Patch(facecolor=COLOR_INCLUDED, alpha=0.7, label=f"Analyzed (n={len(included_df)})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9, ncol=2)
    fig.suptitle("Continuous variables: Total cohort vs Analyzed",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    out_path = os.path.join(outdir, "descriptive_continuous_boxplots.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_categorical_comparison(
    total_df: pd.DataFrame, included_df: pd.DataFrame, outdir: str
) -> None:
    """Grid of grouped bar charts: total (yellow) vs analyzed (green), one subplot per variable."""
    n_vars = len(_CAT)
    n_cols = 4
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    n_tot = len(total_df)
    n_inc = len(included_df)

    for i, (col, label, cat_labels) in enumerate(_CAT):
        ax = axes[i]
        categories = list(cat_labels.values())
        codes      = list(cat_labels.keys())
        x = np.arange(len(categories))
        width = 0.35

        pct_tot = [100 * (total_df[col] == c).sum() / n_tot for c in codes]
        pct_inc = [100 * (included_df[col] == c).sum() / n_inc for c in codes]

        bars_tot = ax.bar(x - width / 2, pct_tot, width,
                          color=COLOR_TOTAL,    alpha=0.75, label=f"Total (n={n_tot})")
        bars_inc = ax.bar(x + width / 2, pct_inc, width,
                          color=COLOR_INCLUDED, alpha=0.75, label=f"Analyzed (n={n_inc})")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=7,
                           rotation=30 if len(categories) > 3 else 0, ha="right")
        ax.set_ylabel("%", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_TOTAL,    alpha=0.75, label=f"Total cohort (n={n_tot})"),
        mpatches.Patch(facecolor=COLOR_INCLUDED, alpha=0.75, label=f"Analyzed (n={n_inc})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9, ncol=2)
    fig.suptitle("Categorical variables: Total cohort vs Analyzed",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    out_path = os.path.join(outdir, "descriptive_categorical_bars.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_descriptive_comparison(
    outdir: str,
    clinical_path: str,
    excluded_path: str,
    included_path: str,
    cohort: str = "clean",
) -> None:
    """Descriptive comparison: total clinical cohort vs the requested cohort's included group."""
    print(f"\n=== Descriptive Comparison: Total cohort vs Analyzed (cohort={cohort}) ===\n")
    included_raw, excluded_raw = load_and_split(
        cohort=cohort,
        clinical_path=clinical_path,
        excluded_path=excluded_path,
        included_path=included_path,
    )

    # Reconstruct total cohort by joining included + excluded
    total_raw = pd.concat([included_raw, excluded_raw], ignore_index=True)

    total    = clean_variables(total_raw)
    included = clean_variables(included_raw)

    print("Building descriptive table...")
    table = build_descriptive_table(total, included)
    _export_descriptive_table(table, n_tot=len(total), n_inc=len(included), outdir=outdir)

    print("Generating continuous boxplots...")
    plot_continuous_comparison(total, included, outdir=outdir)

    print("Generating categorical bar charts...")
    plot_categorical_comparison(total, included, outdir=outdir)

    print("Descriptive comparison done.")