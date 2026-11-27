"""main.py"""

"""
End-to-end image-analysis pipeline for a double-IHC tissue microarray cohort.
Loads the per-core exports, optionally corrects the staining-batch effect
using the control TMA as a reference material (BATCH_CORRECTION, see
batch_correction.py), normalizes intensities per TMA, fits one Gaussian
Mixture Model per staining group and marker to get the GATA6/CK17 positivity
thresholds, classifies cells, and aggregates up to patient level for the
batch-effect, reproducibility and clinical-correlation steps that follow.

Every run parameter (paths, the cohort switch, whether to refit the GMMs)
lives in the block right below this docstring, so nothing further down the
file needs touching to rerun on different data.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os
import re
import json
import pandas as pd
from data_preprocessing import load_data_if_exists, drop_cores_with_all_nan_patients, validation_calculations, export_cells_per_core_in_raw_order, run_model_pathologist_validation
from plot_static import (
    plot_intensity_kde_by_staining_group,
    plot_violin,
    plot_intensity_lines,
    plot_intensity_kde,
    plot_overlapping_intensity_kde_per_core,
    plot_binary_correlation_heatmap,
)
from plot_interactive import plot_patient_core_scatter, plot_patient_core_scatter_means
from normalization import normalize_intensities_by_tma_quantiles
from gmm_thresholds import auto_gmm_threshold, plot_gmm_supplementary_figure
from clinical import run_clinical_multivariate_exploration
from batch_effect import run_batch_effect_diagnosis, add_hospital_from_id, print_tma_scale_summary, summarize_scale_change
from batch_correction import load_control_for_calibration, compute_control_factors, apply_batch_correction
from reproducibility import run_reproducibility, run_staining_reproducibility, run_workflow_reproducibility, cross_reference_workflow_heterogeneity
from cell_classification import apply_group_specific_thresholds_5groups, add_core_hscores_to_cells
from summary_tables import generate_metadata_table, generate_clinical_table
from molecular_profile_analysis import run_grade_molecular_correlation
from bias_assessment import run_bias_assessment, run_descriptive_comparison

# ---------------------------------------------------------------------------
# Cohort selection.
#   "complete" -> all patients with valid image material; included/excluded
#                 for bias assessment follows internal_documentation/
#                 OPTIPANC_ExcludedPatients.xlsx.
#   "clean"    -> final publication roster (internal_documentation/
#                 Patient_TMA_Core_Clean.xlsx); everyone else is excluded.
# Drives OUTPUT_ROOT, which patients the GMMs/classification/bias assessment
# see, and gates the complete-cohort-only reproducibility and inter-observer
# validation steps further down.
# ---------------------------------------------------------------------------
COHORT = "complete"   # "complete" | "clean"

OUTPUT_ROOT = f"output/{COHORT}_cohort"

# Column used to group cells for GMM fitting / threshold assignment.
GROUP_COL = "Staining"

# FIT_GMM:
#   True  -> refit the 5-group x 2-marker GMMs, save metrics/params/thresholds/figure.
#   False -> reuse the thresholds/parameters already saved on disk, no refit.
FIT_GMM = False

# ---------------------------------------------------------------------------
# Staining-batch correction (batch_correction.py).
#   True  -> ratio-based correction with the control TMA as reference material
#            (Yu et al., Genome Biology 2024; 25:254): one factor per staining
#            batch and intensity column, on optical density, applied to the raw
#            intensities BEFORE the per-TMA q01-q99 normalization. The control
#            TMA (WORKFLOW_REPRO_FOLDER) is corrected with the same factors in
#            the workflow-reproducibility step.
#   False -> no correction (pipeline as without this step).
# Outputs go to the same OUTPUT_ROOT either way: a corrected run overwrites an
# uncorrected one. The factors are saved to OUTPUT_ROOT/batch_correction/.
#
# IMPORTANT: the factors are ALWAYS computed from the FULL control TMA (every
# core present in all batches, all control patients), whatever COHORT is. The
# control TMA is a technical calibrator, not part of the study cohort, so the
# clean-roster filter is never applied to it for calibration (it still is for
# the workflow-reproducibility metrics under COHORT="clean"). The factors are
# therefore identical for "clean" and "complete".
#
# REFERENCE_BATCH: staining batch (Patients_list.xlsx "Batch" column) every
# other batch is rescaled to; it gets factor 1 and its TMAs are not modified.
# ---------------------------------------------------------------------------
BATCH_CORRECTION = True
REFERENCE_BATCH = 1


# Input paths. Change these to point at different data; nothing else in this
# file needs touching.
base_folder = "data/raw/Low_dilatation_opt1_staining_retrained"
processed_folder = "data/processed/Low_dilatation_opt1_staining_retrained"
CLINICAL_EXCEL_PATH = "data/raw/Clinical_data/OptiPanc Base DEF 09042026_sindupli.xlsx"
EXCLUDED_ROSTER_PATH = "internal_documentation/OPTIPANC_ExcludedPatients.xlsx"
CLEAN_ROSTER_PATH = "internal_documentation/Patient_TMA_Core_Clean.xlsx"
PATHOLOGIST_VALIDATION_PATH = "internal_documentation/Validation_comparation_5groups_trial.xlsx"
# Round 2: Pathologist1 alone, H12O-OPT11/OPT15 (control-TMA replicates,
# staining batches 1/5) + HUPH-OPT24. Negatives entered as 0; "±" staining
# (intensity < 1+) entered as 0 (H12O-OPT11 10,C CK17).
PATHOLOGIST_VALIDATION_ROUND2_PATH = "internal_documentation/Validation_pathologist1_set2.xlsx"

# Reproducibility inputs (complete-cohort only for scanning/staining; both
# cohorts for the control-TMA workflow check further down).
SCANNING_REPRO_FOLDER = "data/raw/Reproducibility_opt1_staining_retrained/Scanning"
STAINING_REPRO_FOLDER = "data/raw/Reproducibility_opt1_staining_retrained/Staining"
STAINING_BATCH_ORDER = ["Batch1", "Batch2", "Batch4", "Batch5"]
WORKFLOW_REPRO_FOLDER = "data/raw/Reproducibility_opt1_staining_retrained/Workflow"
WORKFLOW_BATCH_ORDER = ["H12O-OPT11", "H12O-OPT12", "H12O-OPT14", "H12O-OPT15"]

# Reference k/threshold per staining group and marker, recorded once a GMM
# fit has been reviewed, so routine reruns don't need a manual re-check every
# time: each run's obtained values are compared against this and the
# pipeline stops before classification if they disagree. One entry per
# COHORT and per BATCH_CORRECTION setting (key "<cohort>" without correction,
# "<cohort>_batchcorr" with it), since the fit depends on which patients/cores
# go into it and on whether their intensities were corrected. If the input
# data or the cohort definition changes enough to need a new fit, refit,
# review the new thresholds and figure, and paste the new values in here as
# the next reference.
EXPECTED_BY_COHORT = {
    "clean": {
        "DAB++":       {"GATA6": (4, 0.4210), "CK17": (5, 0.7066)},
        "Low_dab":     {"GATA6": (4, 0.6385), "CK17": (5, 0.7428)},
        "Low_mag+dab": {"GATA6": (4, 0.5547), "CK17": (5, 0.6793)},
        "Mag++":       {"GATA6": (4, 0.5497), "CK17": (4, 0.6060)},
        "Mag+DAB":     {"GATA6": (4, 0.5109), "CK17": (5, 0.6851)},
    },
    "complete": {
        "DAB++":       {"GATA6": (4, 0.4214), "CK17": (5, 0.7072)},
        "Low_dab":     {"GATA6": (4, 0.6263), "CK17": (5, 0.7276)},
        "Low_mag+dab": {"GATA6": (4, 0.5521), "CK17": (5, 0.7374)},
        "Mag++":       {"GATA6": (4, 0.5474), "CK17": (4, 0.5740)},
        "Mag+DAB":     {"GATA6": (4, 0.5091), "CK17": (5, 0.6636)},
    },
    "clean_batchcorr": {
    "DAB++":       {"GATA6": (4, 0.3879), "CK17": (5, 0.7061)},
    "Low_dab":     {"GATA6": (4, 0.6336), "CK17": (5, 0.7428)},
    "Low_mag+dab": {"GATA6": (4, 0.5302), "CK17": (5, 0.6826)},
    "Mag++":       {"GATA6": (4, 0.5041), "CK17": (4, 0.5942)},
    "Mag+DAB":     {"GATA6": (4, 0.4693), "CK17": (5, 0.6767)},
    },
    "complete_batchcorr": {
        "DAB++":       {"GATA6": (4, 0.3944), "CK17": (5, 0.7047)},
        "Low_dab":     {"GATA6": (4, 0.6191), "CK17": (5, 0.7293)},
        "Low_mag+dab": {"GATA6": (4, 0.5060), "CK17": (5, 0.7280)},
        "Mag++":       {"GATA6": (4, 0.5127), "CK17": (4, 0.5620)},
        "Mag+DAB":     {"GATA6": (4, 0.4654), "CK17": (5, 0.6682)},
    },
}
EXPECTED_KEY = f"{COHORT}_batchcorr" if BATCH_CORRECTION else COHORT

# Load and process the data if not already processed
data = load_data_if_exists(base_folder, processed_folder)

# Drop cores with no Patient (controls)
data = drop_cores_with_all_nan_patients(data)

# Clean-cohort roster, loaded unconditionally so it's also available below for
# the control-TMA filter in workflow reproducibility (both "complete" and
# "clean" runs need clean_ids in scope — only "clean" actually filters on it).
clean_roster = pd.read_excel(CLEAN_ROSTER_PATH)
clean_ids = set(
    clean_roster["Patient ID"].astype(str).str.replace(r"\s+", "", regex=True)
)


def _normalize_visiopharm_core(value: str) -> str:
    """Convert a roster 'Visiopharm Core' code (e.g. "1H") to the image data's
    'Core' format (e.g. "01, H"): zero-padded row number + ", " + column letter(s).
    """
    match = re.match(r"^(\d+)([A-Za-z]+)$", str(value).strip())
    num, letter = match.groups()
    return f"{int(num):02d}, {letter}"


# Exact (Patient ID, TMA, Core) keys the roster marks as clean. Not every
# core of a "clean" patient is itself clean (a patient can have 3 cores in
# the raw data but only 2 listed here), so filtering on Patient ID alone
# would over-include cores.
clean_core_keys = set(
    zip(
        clean_roster["Patient ID"].astype(str).str.replace(r"\s+", "", regex=True),
        clean_roster["TMA"].astype(str).str.strip(),
        clean_roster["Visiopharm Core"].map(_normalize_visiopharm_core),
    )
)

# Cohort filter: for "clean", restrict to the final publication roster before
# anything downstream (metadata table, plots, batch effect, normalization,
# GMM fit, classification, bias assessment) sees the data. Matched on the
# exact (Patient ID, TMA, Core) triple, not just Patient ID, so only the
# cores the roster actually lists as clean survive. Nothing here is
# hardcoded; the resulting counts are printed so they can be checked against
# the roster by hand.
if COHORT == "clean":
    data_ids_norm = data["ID"].astype(str).str.replace(r"\s+", "", regex=True)
    data_keys = list(
        zip(data_ids_norm, data["TMA"].astype(str).str.strip(), data["Core"].astype(str).str.strip())
    )
    data = data[[key in clean_core_keys for key in data_keys]].copy()
    print(
        f"[COHORT=clean] Roster: {clean_roster['Patient ID'].nunique()} patients "
        f"({len(clean_roster)} rows / {len(clean_core_keys)} unique cores). "
        f"Filtered image data to "
        f"{data['ID'].nunique()} patients / "
        f"{data[['TMA', 'Core']].drop_duplicates().shape[0]} cores."
    )

# Generate TMA metadata summary table (before intensity dropna to capture all cores)
print("Generating TMA metadata summary table...")
generate_metadata_table(data, outdir=f"{OUTPUT_ROOT}/summary_tables")

# The pipeline tracks two per-cell markers, each read from the Visiopharm
# export as a mean staining intensity in a specific cell compartment: GATA6
# in the nucleus (column "DAB Intensity - Nuclei") and CK17 in the cytoplasm
# (column "Mag Intensity - Cytoplasm"). Drop any cell missing either reading.
data = data.dropna(subset=['DAB Intensity - Nuclei', 'Mag Intensity - Cytoplasm'])

# Perform the plots
# "Nuclei"/"Cytoplasm" above are the compartments the two columns are read
# from; switch either column to its "Complete Cell" counterpart if the
# whole-cell intensity is wanted instead for that marker.
plot_violin(data, output_root=OUTPUT_ROOT)
plot_intensity_lines(data, output_root=OUTPUT_ROOT)
plot_intensity_kde(
    data,
    suffix="",
    outpath=f"{OUTPUT_ROOT}/distributions/general_tmas/distributions_per_TMA_raw.svg",
)
plot_overlapping_intensity_kde_per_core(data, output_root=OUTPUT_ROOT)
scale_raw = print_tma_scale_summary(
    df=data,
    tma_col="TMA",
    intensity_cols={
        "GATA6": "DAB Intensity - Nuclei",
        "CK17": "Mag Intensity - Cytoplasm",
    },
    label="RAW",
    scale_max=255,
    outdir=f"{OUTPUT_ROOT}/normalization",
)
plot_patient_core_scatter(data, output_root=OUTPUT_ROOT)
plot_patient_core_scatter_means(data, output_root=OUTPUT_ROOT)


## Batch effect diagnosis on raw intensities (PRE-thresholding)
print("Running batch effect diagnosis (PRE, raw intensities)...")
data = add_hospital_from_id(data, id_col="ID", out_col="Hospital", tma_col="TMA")
run_batch_effect_diagnosis(
    mode="PRE",
    raw_df=data,
    validation_results=None,
    outdir=f"{OUTPUT_ROOT}/batch_effect_raw",
    batch_cols=["TMA", "Hospital", "Staining Date", "Scan Date"],
    sample_col_for_pvca="Patient",
)

## Staining-batch correction (optional, see BATCH_CORRECTION at the top).
## After the PRE diagnosis (which always describes the uncorrected raw data)
## and before the per-TMA normalization, which would otherwise absorb the
## between-batch difference measured on the control TMA. Factors always come
## from the full control TMA, never cohort-filtered (see batch_correction.py).
batch_factors = None
if BATCH_CORRECTION:
    print(f"Batch correction: computing factors from the full control TMA "
          f"(reference batch = {REFERENCE_BATCH})...")
    control_calibration = load_control_for_calibration(WORKFLOW_REPRO_FOLDER)
    batch_factors = compute_control_factors(
        control_calibration,
        reference_batch=REFERENCE_BATCH,
        outdir=f"{OUTPUT_ROOT}/batch_correction",
    )
    del control_calibration
    data = apply_batch_correction(data, batch_factors)
    plot_intensity_kde(
        data,
        suffix="",
        outpath=f"{OUTPUT_ROOT}/distributions/general_tmas/distributions_per_TMA_raw_batchcorrected.svg",
    )
    print("Batch correction applied.")

## Normalization by TMA quantiles (q01-q99 scaling)
print("Normalizing intensities per TMA (q01–q99, clipped)...")
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
    clip=True,   # IMPORTANT: for global GMM stability
    outdir=f"{OUTPUT_ROOT}/normalization",
)
print("Normalization complete.")

# Visiopharm also reports a second, independent intensity reading per marker
# and compartment under a "Phenotyping" suffix. Project those onto the same
# per-TMA q01-q99 scale already fit above for the primary GATA6 nuclear
# ("DAB Intensity - Nuclei") and CK17 cytoplasmic ("Mag Intensity - Cytoplasm")
# columns, so the GMM thresholds (fit on the primary _norm columns) apply to
# the Phenotyping columns too.
data_norm = data_norm.merge(
    q_table[["TMA", "GATA6_q01", "GATA6_denom", "CK17_q01", "CK17_denom"]],
    on="TMA", how="left"
)
data_norm["DAB Intensity - Nuclei Phenotyping_norm"] = (
    (data_norm["DAB Intensity - Nuclei Phenotyping"] - data_norm["GATA6_q01"])
    / data_norm["GATA6_denom"]
).clip(0, 1)
data_norm["Mag Intensity - Cytoplasm Phenotyping_norm"] = (
    (data_norm["Mag Intensity - Cytoplasm Phenotyping"] - data_norm["CK17_q01"])
    / data_norm["CK17_denom"]
).clip(0, 1)
data_norm.drop(columns=["GATA6_q01", "GATA6_denom", "CK17_q01", "CK17_denom"], inplace=True)
print("Phenotyping columns projected onto normalized scale.")

# Re-run the plots to see the normalized data
plot_intensity_kde(
    data_norm,
    suffix="_norm",
    outpath=f"{OUTPUT_ROOT}/distributions/general_tmas/distributions_per_TMA_norm.svg",
    xlim=(0, 1),
)
scale_norm = print_tma_scale_summary(
    df=data_norm,
    tma_col="TMA",
    intensity_cols={
        "GATA6": "DAB Intensity - Nuclei_norm",
        "CK17": "Mag Intensity - Cytoplasm_norm",
    },
    label="NORMALIZED",
    scale_max=1,
    outdir=f"{OUTPUT_ROOT}/normalization",
)
summarize_scale_change(scale_raw, scale_norm, outdir=f"{OUTPUT_ROOT}/normalization")

# ---------------------------------------------------------------------------
# GMM grouping, by Staining group.
# ---------------------------------------------------------------------------
groups_present = sorted(data_norm[GROUP_COL].dropna().unique().tolist())
print(f"Grouping by '{GROUP_COL}' → {len(groups_present)} groups: {groups_present}")

# Compare GATA6/CK17 distributions across the 5 staining groups before fitting any GMM
plot_intensity_kde_by_staining_group(
    data_norm,
    ['DAB++', 'Low_dab', 'Low_mag+dab', 'Mag++', 'Mag+DAB'],
    suffix="_norm",
    outpath=f"{OUTPUT_ROOT}/distributions/per_staining_group/distributions_per_group_norm.svg",
    xlim=(0, 1),
)

# GMM thresholding (evaluation, fitting and plotting) — one GATA6 + one CK17 GMM
# per group of GROUP_COL, controlled by FIT_GMM (see top of file): True refits
# everything and saves metrics/params/thresholds/figure; False reuses what is
# already saved on disk (no refit).
gmm_outdir = f"{OUTPUT_ROOT}/gmm"
marker_cols_raw = {
    "GATA6": "DAB Intensity - Nuclei_norm",
    "CK17": "Mag Intensity - Cytoplasm_norm",
}

if FIT_GMM:
    print("FIT_GMM=True: fitting GMMs per group/marker...")
    metrics_rows = []
    fits = {}
    for gval in groups_present:
        sub = data_norm[data_norm[GROUP_COL] == gval]
        fits[gval] = {}
        for marker, col in marker_cols_raw.items():
            _thr, _chosen_k, metrics_df, fit_params = auto_gmm_threshold(
                sub, col,
                output_dir=gmm_outdir, max_clusters=6,
                group_label=str(gval), marker_label=marker,
            )
            fits[gval][marker] = fit_params
            m = metrics_df.copy()
            m["group"] = str(gval)
            m["marker"] = marker
            m["elbow_k"] = fit_params["elbow_k"]
            m["chosen_k"] = fit_params["chosen_k"]
            metrics_rows.append(m)

    metrics_all = pd.concat(metrics_rows, ignore_index=True)[
        ["group", "marker", "k", "ssq", "silhouette", "bic", "elbow_k", "chosen_k"]
    ]
    os.makedirs(gmm_outdir, exist_ok=True)
    metrics_all.to_csv(os.path.join(gmm_outdir, "gmm_evaluation_metrics.csv"), index=False)

    fit_param_rows = []
    thresholds = {}
    k_obtained = {}
    for gval, by_marker in fits.items():
        thresholds[gval] = {}
        k_obtained[gval] = {}
        for marker, fp in by_marker.items():
            thr_rounded = round(float(fp["threshold"]), 4)
            thresholds[gval][marker] = thr_rounded
            k_obtained[gval][marker] = fp["chosen_k"]
            for i in range(len(fp["means"])):
                fit_param_rows.append({
                    "group": gval,
                    "marker": marker,
                    "chosen_k": fp["chosen_k"],
                    "component": i,
                    "mean": float(fp["means"][i]),
                    "sd": float(fp["stds"][i]),
                    "weight": float(fp["weights"][i]),
                    "threshold": thr_rounded,
                })
    pd.DataFrame(fit_param_rows).to_csv(
        os.path.join(gmm_outdir, "gmm_fit_parameters.csv"), index=False
    )
    with open(os.path.join(gmm_outdir, "thresholds_opt1.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[THRESHOLDS] saved → {gmm_outdir}/thresholds_opt1.json "
          "(4-decimal rounded; this is what gets used for classification)")

    plot_gmm_supplementary_figure(
        metrics_all=metrics_all,
        fits=fits,
        data=data_norm,
        group_col=GROUP_COL,
        marker_cols=marker_cols_raw,
        outdir=gmm_outdir,
    )
else:
    print("FIT_GMM=False: loading thresholds_opt1.json and "
          "gmm_fit_parameters.csv (no refit)...")
    with open(os.path.join(gmm_outdir, "thresholds_opt1.json")) as f:
        thresholds = json.load(f)
    fit_params_path = os.path.join(gmm_outdir, "gmm_fit_parameters.csv")
    fit_params_df = pd.read_csv(fit_params_path)
    k_obtained = {}
    for (gval, marker), sub in fit_params_df.groupby(["group", "marker"]):
        k_obtained.setdefault(str(gval), {})[marker] = int(sub["chosen_k"].iloc[0])

assert thresholds, (
    "thresholds empty: refit the GMMs (FIT_GMM=True) before classification."
)

# GMM reproducibility check. Compares the k/threshold obtained above (fresh
# fit or reloaded from disk) against EXPECTED_BY_COHORT (set at the top of
# this file), so an unintended drift between runs gets caught before
# classification instead of silently changing the results.
expected = EXPECTED_BY_COHORT.get(EXPECTED_KEY, {})

if not expected:
    print(f"\n[GMM REPRODUCIBILITY CHECK] SKIPPED — no reference fit recorded yet "
          f"for {EXPECTED_KEY!r}. Review the thresholds/plots from this run, then "
          f"paste them into EXPECTED_BY_COHORT[{EXPECTED_KEY!r}] in main.py to enable the "
          "check on future reruns.\n")
else:
    check_rows = []
    all_match = True
    for gval, by_marker in expected.items():
        for marker, (k_exp, thr_exp) in by_marker.items():
            k_obt = k_obtained.get(gval, {}).get(marker)
            thr_obt_raw = thresholds.get(gval, {}).get(marker)
            thr_obt = round(float(thr_obt_raw), 4) if thr_obt_raw is not None else None
            abs_diff = abs(thr_obt - thr_exp) if thr_obt is not None else float("nan")
            match = (k_obt == k_exp) and (thr_obt is not None) and (abs_diff < 1e-6)
            all_match = all_match and match
            check_rows.append({
                "group": gval, "marker": marker,
                "k_expected": k_exp, "k_obtained": k_obt,
                "thr_expected": thr_exp, "thr_obtained": thr_obt,
                "abs_diff": abs_diff, "match": match,
            })

    check_df = pd.DataFrame(check_rows)
    os.makedirs(gmm_outdir, exist_ok=True)
    check_path = os.path.join(gmm_outdir, "gmm_reproducibility_check.csv")
    check_df.to_csv(check_path, index=False)
    print(f"\n[GMM REPRODUCIBILITY CHECK] saved → {check_path}")
    print(check_df.to_string(index=False))

    if not all_match:
        print("\n" + "!" * 78)
        print("!!! GMM REPRODUCIBILITY CHECK FAILED")
        print("!!! Obtained thresholds/k do NOT match the expected reference values.")
        print(f"!!! See {check_path} for the row(s) that differ.")
        print("!!! STOPPING before classification — review before re-running.")
        print("!" * 78 + "\n")
        raise RuntimeError(
            "GMM reproducibility check failed: obtained thresholds/k do not match "
            f"the expected reference values. See {check_path}."
        )
    print("[GMM REPRODUCIBILITY CHECK] OK — all thresholds/k match the reference fit.\n")

# Cell classification using the per-group GMM thresholds
data_norm = apply_group_specific_thresholds_5groups(
    df=data_norm,
    staining_col=GROUP_COL,
    group_thresholds=thresholds,
    dab_intensity_col="DAB Intensity - Nuclei Phenotyping_norm",
    ck17_intensity_col="Mag Intensity - Cytoplasm Phenotyping_norm",
    out_suffix="",   # leave empty to match Visiopharm-like column names
)
data_norm, core_endpoints = add_core_hscores_to_cells(
    cell_df=data_norm,
    group_cols=("TMA", "Hospital", "Core", "Patient", "ID"),
    out_suffix="",  # pon "_norm" si generas columnas con suffix
)
print("Cell-level gating columns added and saved.")
plot_binary_correlation_heatmap(data_norm, output_root=OUTPUT_ROOT)

# Export cells per core in raw order for final validation.
# out_root redirects the export to OUTPUT_ROOT (instead of the shared processed
# folder). These are the files reimported into Visiopharm to compare classification.
export_cells_per_core_in_raw_order(
    cell_df=data_norm,
    processed_folder=processed_folder,
    out_root=OUTPUT_ROOT,
)
print("Per-core exports in raw row order saved.")

# Make calculations for final validation
print("Calculating final validation metrics...")
validation_results = validation_calculations(
    data_norm,
    f"{OUTPUT_ROOT}/csv",
    thresholds=thresholds,
    group_col=GROUP_COL,
)
print("Validation results saved.")

## Batch effect diagnosis on POST-thresholding endpoints
print("Running batch effect diagnosis (POST, endpoints)...")
validation_results = add_hospital_from_id(validation_results, id_col="ID", out_col="Hospital", tma_col="TMA")
run_batch_effect_diagnosis(
    mode="POST",
    raw_df=None,
    validation_results=validation_results,
    outdir=f"{OUTPUT_ROOT}/batch_effect_post",
    batch_cols=["TMA", "Hospital", "Staining Date", "Scan Date"],
    sample_col_for_pvca="Patient",
)

# ---------------------------------------------------------------------------
# Scanning and staining reproducibility: raw intensity profiles across
# repeated scans or staining batches, independent of GROUP_COL and the
# thresholds above. Complete-cohort only.
# ---------------------------------------------------------------------------
if COHORT == "complete":
    print("Scanning reproducibility...")
    df_reprod = run_reproducibility(
        base_folder=SCANNING_REPRO_FOLDER,
        outdir=f"{OUTPUT_ROOT}/reproducibility/scanning",
        outname="reproducibility_profiles",
    )
    print("Staining reproducibility...")
    df_staining = run_staining_reproducibility(
        base_folder=STAINING_REPRO_FOLDER,
        outdir=f"{OUTPUT_ROOT}/reproducibility/staining",
        outname="staining_reproducibility_profiles",
        batch_order=STAINING_BATCH_ORDER,
    )

# ---------------------------------------------------------------------------
# Control-TMA workflow reproducibility: classifies the control TMA (the same
# tissue, re-stained across 4 batches) with this cohort's thresholds, then
# exports per-batch results for reimport into Visiopharm to check how
# consistently it gets classified across batches. Runs for both cohorts so
# clean and complete can be compared directly: under "clean" the control TMA
# is restricted to the roster patients first, under "complete" it uses
# everyone on it. With BATCH_CORRECTION the control TMA is corrected with the
# same factors as the cohort (computed above from the full control TMA). Note
# the control TMA is also the calibrator, so its batch means are aligned by
# construction: this step then tests per-core classification consistency.
# ---------------------------------------------------------------------------
print(f"Workflow reproducibility (control TMA, cohort={COHORT})...")
run_workflow_reproducibility(
    base_folder=WORKFLOW_REPRO_FOLDER,
    outdir=f"{OUTPUT_ROOT}/reproducibility/workflow",
    thresholds=thresholds,
    batch_order=WORKFLOW_BATCH_ORDER,
    group_col=GROUP_COL,
    clean_patient_ids=clean_ids if COHORT == "clean" else None,
    batch_factors=batch_factors,
)
print("Cross-referencing workflow discordant cores with heterogeneity metrics...")
cross_reference_workflow_heterogeneity(
    workflow_outdir=f"{OUTPUT_ROOT}/reproducibility/workflow",
    outdir=f"{OUTPUT_ROOT}/reproducibility/workflow",
)


## Inter-observer agreement: pathologist vs model classification on the
## patients already scored in PATHOLOGIST_VALIDATION_PATH (round 1, two
## pathologists) and PATHOLOGIST_VALIDATION_ROUND2_PATH (round 2,
## Pathologist1 only). Each round is reported separately. Complete-cohort
## only (not part of the clean-cohort publication pipeline). Runs after
## workflow reproducibility so the "-model" columns of both Excels can be
## checked against this run's CSVs (control-TMA replicates live there).
if COHORT == "complete":
    model_csv_globs = [
        f"{OUTPUT_ROOT}/csv/*/*_validation.csv",
        f"{OUTPUT_ROOT}/reproducibility/workflow/validation/*/*_validation.csv",
    ]
    run_model_pathologist_validation(
        excel_path=PATHOLOGIST_VALIDATION_PATH,
        outdir=f"{OUTPUT_ROOT}/validation",
        model_csv_globs=model_csv_globs,
        # Round-1 Excel uses the pre-batch TMA names.
        tma_aliases={
            "H12O-OPT1": "H12O-OPT12", "H12O-OPT2": "H12O-OPT21",
            "HPRIN-OPT1": "HPRIN-OPT11", "HRYC-OPTIPANC": "HRYC-OPT11",
            "HUIS-OPT1": "HUIS-OPT11", "HULP-OPT1": "HULP-OPT11",
            "HULP-OPT2": "HULP-OPT21", "HULP-OPT3": "HULP-OPT31",
        },
    )
    run_model_pathologist_validation(
        excel_path=PATHOLOGIST_VALIDATION_ROUND2_PATH,
        outdir=f"{OUTPUT_ROOT}/validation",
        out_name="model_pathologist_validation_round2.xlsx",
        model_csv_globs=model_csv_globs,
    )

## Bias assessment: excluded vs included patients
print(f"Running bias assessment (cohort={COHORT})...")
run_bias_assessment(
    outdir=f"{OUTPUT_ROOT}/bias_assessment",
    cohort=COHORT,
    clinical_path=CLINICAL_EXCEL_PATH,
    excluded_path=EXCLUDED_ROSTER_PATH,
    included_path=CLEAN_ROSTER_PATH,
)

## Descriptive comparison: total clinical cohort vs this cohort's included group
print(f"Running descriptive comparison (cohort={COHORT})...")
run_descriptive_comparison(
    outdir=f"{OUTPUT_ROOT}/bias_assessment",
    cohort=COHORT,
    clinical_path=CLINICAL_EXCEL_PATH,
    excluded_path=EXCLUDED_ROSTER_PATH,
    included_path=CLEAN_ROSTER_PATH,
)

## Merge and explore clinical data
print("Running clinical correlation exploratory analysis...")
outputs = run_clinical_multivariate_exploration(
    validation_results=validation_results,
    clinical_path=CLINICAL_EXCEL_PATH,
    outdir=f"{OUTPUT_ROOT}/clinical",
    id_col="ID",
    hue_cat="Stage (TNM)",  # optional: if None, automatic selection
)
merged_df = outputs["merged"]
generate_clinical_table(merged_df, outdir=f"{OUTPUT_ROOT}/summary_tables")

## Grade vs molecular correlation: Spearman rho between the 4 populations
## (+ % GATA6+ / % CK17+ totals) and pathologist differentiation grade.
print("Running grade vs molecular correlation...")
run_grade_molecular_correlation(
    merged_df=merged_df,
    outdir=f"{OUTPUT_ROOT}/molecular_profile",
)
