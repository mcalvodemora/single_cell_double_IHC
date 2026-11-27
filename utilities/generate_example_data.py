"""generate_example_data.py

Builds a small, fully synthetic dataset under example_data/, shaped exactly
like what the pipeline (src/data_preprocessing.py, src/clinical.py,
src/reproducibility.py, src/bias_assessment.py) reads, so the published code
can be run end to end without any real patient data.

Every value here is randomly generated (fixed seed) from the column-name
constants already used elsewhere in the code -- nothing is copied from real
files. Patient IDs, TMA names and all free-text fields use an unmistakable
"SYN" prefix so nobody could confuse this for real data. See
example_data/README.md for the "this is synthetic" notice shown alongside
the data itself.

Run from the repository root: python utilities/generate_example_data.py
"""

import os
import shutil
import uuid

import numpy as np
import pandas as pd

rng = np.random.RandomState(0)

ROOT = "example_data"
BASE_FOLDER = os.path.join(ROOT, "raw", "Synthetic_TMA_cohort")
CLINICAL_DIR = os.path.join(ROOT, "raw", "Clinical_data")
REPRO_BASE = os.path.join(ROOT, "raw", "Synthetic_reproducibility")
INTERNAL_DIR = os.path.join(ROOT, "internal_documentation")

STAINING_GROUPS = ["DAB++", "Low_dab", "Low_mag+dab", "Mag++", "Mag+DAB"]


def fresh_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def per_object_df(n_cells):
    dab_cell = np.clip(rng.normal(110, 35, n_cells), 0, 255)
    dab_nuclei = np.clip(dab_cell + rng.normal(0, 10, n_cells), 0, 255)
    mag_cell = np.clip(rng.normal(230, 40, n_cells), 0, 255).astype(int)
    mag_cyto = np.clip(mag_cell + rng.normal(0, 10, n_cells), 0, 255).astype(int)
    return pd.DataFrame({
        "Name": ["placeholder"] * n_cells,  # overwritten by caller per-core
        "DAB Intensity - Cell": dab_cell,
        "DAB Intensity - Nuclei": dab_nuclei,
        "DAB Intensity - Nuclei Phenotyping": np.clip(dab_nuclei + rng.normal(0, 5, n_cells), 0, 255),
        "Diameter per cell": rng.uniform(5, 20, n_cells),
        "Mag Intensity - Cell": mag_cell,
        "Mag Intensity - Cytoplasm": mag_cyto,
        "Mag Intensity - Cytoplasm Phenotyping": np.clip(mag_cyto + rng.normal(0, 5, n_cells), 0, 255).astype(int),
        "Object Info - Object ID": [str(uuid.uuid4()) for _ in range(n_cells)],
    })


def per_image_df():
    return pd.DataFrame({
        "Name": ["placeholder"],
        "DAB Intensity - Nuclei/Core": [float(np.clip(rng.normal(110, 20), 0, 255))],
        "Last APP": ["Synthetic_classification_APP_v1"],
        "Mag Intensity - Boundary/Core": [float(np.clip(rng.normal(230, 20), 0, 255))],
    })


def write_core(outdir, core_name, n_cells):
    obj_dir = os.path.join(outdir, "per_object")
    img_dir = os.path.join(outdir, "per_image")
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    obj = per_object_df(n_cells)
    obj["Name"] = f"Grid {core_name}"
    obj.to_csv(os.path.join(obj_dir, f"Grid {core_name}_objects.tsv"), sep="\t", index=False)

    img = per_image_df()
    img["Name"] = f"Grid {core_name}"
    img.to_csv(os.path.join(img_dir, f"Grid {core_name}_image.tsv"), sep="\t", index=False)


def grid_code(row_idx, col_idx):
    cols = "ABCDEFGHIJ"
    return f"({row_idx:02d}, {cols[col_idx]})"


def build_main_cohort():
    fresh_dir(BASE_FOLDER)
    cores_per_group = 2
    cells_per_core = 100
    tmas = ["SYN-TMA1", "SYN-TMA2"]

    all_patient_rows = []
    patient_counter = 1

    for tma_idx, tma in enumerate(tmas):
        tma_dir = os.path.join(BASE_FOLDER, tma)
        patient_core_map = {}  # patient_id -> list of "1A" style codes
        # Each TMA is its own batch -- scanned/stained on a different date --
        # so there's an actual TMA/batch axis for the batch-effect diagnosis
        # to check, instead of everything looking like one run.
        batch_num = tma_idx + 1
        scan_date = pd.Timestamp("2026-01-15") + pd.Timedelta(days=5 * tma_idx)
        staining_date = pd.Timestamp("2026-01-10") + pd.Timedelta(days=5 * tma_idx)

        for g_idx, group in enumerate(STAINING_GROUPS):
            group_dir = os.path.join(tma_dir, group)
            row_num = g_idx + 1
            core_codes_padded = []
            for c_idx in range(cores_per_group):
                code = grid_code(row_num, c_idx)
                write_core(group_dir, code, cells_per_core)
                # compact code matching Patients_list.xlsx "Core" convention, e.g. "1A"
                compact = f"{row_num}{'ABCDEFGHIJ'[c_idx]}"
                core_codes_padded.append(compact)

            # assign each core in this group to a (possibly new) synthetic patient
            for compact in core_codes_padded:
                patient_id = f"SYNPAT{patient_counter:03d}"
                patient_counter += 1
                patient_core_map[patient_id] = [compact]

        core_str_by_patient = {pid: ",".join(codes) for pid, codes in patient_core_map.items()}
        patients_list_rows = []
        for pid, core_str in core_str_by_patient.items():
            patients_list_rows.append({
                "Patient ID": pid,
                "NAP": f"SYN-NAP-{pid}",
                "TMA": tma,
                "Core": core_str,
                "Scan Date": scan_date,
                "Scan Time": "10:00:00",
                "Scan Profile": "BF_20X_SYN",
                "Staining Date": staining_date,
                "GATA6 Lot": f"SYNLOT{batch_num:03d}",
                "CK17 Lot": batch_num,
                "GATA6 Staining Protocol": "Synthetic_Protocol_A",
                "CK17 Staining Protocol": "Synthetic_Protocol_A",
                "Batch": batch_num,
            })
            all_patient_rows.append(pid)
        pd.DataFrame(patients_list_rows).to_excel(
            os.path.join(tma_dir, "Patients_list.xlsx"), index=False
        )

    return all_patient_rows, tmas


def build_clinical_and_rosters(main_patient_ids, tmas):
    fresh_dir(CLINICAL_DIR)
    fresh_dir(INTERNAL_DIR)

    excluded_ids = [f"SYNPAT{900 + i}" for i in range(3)]
    all_ids = main_patient_ids + excluded_ids

    rows = []
    for pid in all_ids:
        grade = rng.choice([1, 2, 3])
        rows.append({
            "ID": pid,
            "NHC": float(rng.randint(100000, 999999)),
            "Pathology Accession Number": f"SYN-ACC-{pid}",
            "Date of Birth": "1950-01-01",
            "Date of Diagnosis": "2024-01-01",
            "Age at Diagnosis (years)": float(rng.randint(45, 85)),
            "Sex (female=1, male=2)": float(rng.choice([1, 2])),
            "ECOG PS": float(rng.choice([0, 1, 2, 3])),
            "Diabetes (yes=1, no=2)": float(rng.choice([1, 2])),
            "Smoking (yes=1, no=2)": float(rng.choice([1, 2])),
            "Weight (kg)": float(rng.randint(55, 95)),
            "Height (cm)": float(rng.randint(155, 190)),
            "BMI": str(round(rng.uniform(19, 32), 1)),
            "Alcohol Consumption (0=none, 1=mild, 2=moderate-severe)": float(rng.choice([0, 1, 2])),
            "Chronic Pancreatitis (yes=1, no=2)": float(rng.choice([1, 2])),
            "Family History of Pancreatic Cancer (yes=1, no=2)": float(rng.choice([1, 2])),
            "Familial Pancreatic Cancer Mutation Type": rng.choice([None, "BRCA2", "PALB2"]),
            "Pain at Diagnosis (VAS 0-10)": str(rng.randint(0, 10)),
            "Jaundice at Diagnosis (yes=1, no=2)": float(rng.choice([1, 2])),
            "% Weight Loss at Diagnosis": str(round(rng.uniform(0, 15), 1)),
            "Asthenia (VAS 0-10)": str(rng.randint(0, 10)),
            "Primary Tumor Location (head=1, body=2, tail=3, multiple=4)": float(rng.choice([1, 2, 3, 4])),
            "Differentiation Grade (well=1, moderately=2, poorly=3)": float(grade),
            "Lymphovascular Invasion (yes=1, no=2)": float(rng.choice([1, 2])),
            "Perineural Invasion (yes=1, no=2)": float(rng.choice([1, 2])),
            "Resection Type (R1=1, R0=2)": float(rng.choice([1, 2])),
            "pT": float(rng.choice([1, 2, 3, 4])),
            "Maximum Diameter (cm)": str(round(rng.uniform(1, 6), 1)),
            "pN": float(rng.choice([0, 1, 2])),
            "No. Affected Lymph Nodes": float(rng.randint(0, 5)),
            "No. Resected Lymph Nodes": float(rng.randint(5, 25)),
            "Stage Group (I=1, IIa=2, IIb=3, III-IV=4)": str(rng.choice([1, 2, 3, 4])),
            "Stage (TNM)": "T2N0M0",
            "CA19.9 at Diagnosis": str(round(rng.uniform(5, 500), 1)),
            "Surgery (Whipple=1, Total Pancreatectomy=2, Other=3)": float(rng.choice([1, 2, 3])),
            "Presence of Intraductal Papillary Mucinous Neoplasm (IPMN)": rng.choice(["yes", "no"]),
            "CA19.9 after Surgery": str(round(rng.uniform(5, 100), 1)),
            "Treatment (Gemcitabine=1, Gem-Cap=2, FOLFIRINOX=3)": str(rng.choice([1, 2, 3])),
            "Adjuvant Therapy Start Date (after Surgery)": pd.Timestamp("2024-03-01"),
            "Adjuvant Therapy: Weeks to Start after Surgery": float(rng.randint(2, 10)),
            "No. Months of Treatment": float(rng.randint(1, 6)),
            "% Dose Administered": float(rng.randint(50, 100)),
            "RT Administered (yes=1, no=2)": float(rng.choice([1, 2])),
            "Relapse (yes=1, no=2)": float(rng.choice([1, 2])),
            "Date of Relapse": pd.Timestamp("2024-09-01"),
            "Disease-Free Time from Surgery (months)": float(rng.randint(3, 24)),
            "Relapse Type (local=1, distant=2)": str(rng.choice([1, 2])),
            "Site of Metastasis at Relapse (1=none, 2=liver, 3=lung, 4=peritoneum, 5=bone, 6=two or more of the above, 7=other)": str(rng.choice([1, 2, 3])),
            "Vital Status (dead=1, alive=2)": float(rng.choice([1, 2])),
            "Date of Last Follow-up/Death": "2026-01-01",
            "Overall Survival (months)": float(rng.randint(6, 36)),
            "Cause of Death": rng.choice([None, "progression", "other"]),
            "Molecular Study Available (yes=1, no=2)": rng.choice(["yes", "no"]),
            "Second Tumors Unrelated to Disease Course (yes=1, no=2)": rng.choice(["yes", "no"]),
        })
    clinical_df = pd.DataFrame(rows)
    clinical_df.to_excel(os.path.join(CLINICAL_DIR, "Synthetic_Clinical_Data.xlsx"), index=False)

    # Excluded roster: the 3 clinical-only patients never present in any TMA
    excluded_df = pd.DataFrame({
        "Patient ID": excluded_ids,
        "Hospital": ["SYN"] * len(excluded_ids),
        "Exclusion Type": ["N original -> N TMA"] * len(excluded_ids),
        "Reason": ["Synthetic placeholder exclusion"] * len(excluded_ids),
    })
    excluded_df.to_excel(os.path.join(INTERNAL_DIR, "ExcludedPatients_synthetic.xlsx"), index=False)

    # Clean roster: all main-cohort patients except the last one on each TMA
    clean_rows = []
    for tma in tmas:
        tma_dir = os.path.join(BASE_FOLDER, tma)
        patients_list = pd.read_excel(os.path.join(tma_dir, "Patients_list.xlsx"))
        for _, prow in patients_list.iloc[:-1].iterrows():
            for compact_core in str(prow["Core"]).split(","):
                clean_rows.append({
                    "Patient ID": prow["Patient ID"],
                    "NAP": prow["NAP"],
                    "TMA": tma,
                    "Visiopharm Core": compact_core,
                })
    pd.DataFrame(clean_rows).to_excel(
        os.path.join(INTERNAL_DIR, "Patient_TMA_Core_Clean_synthetic.xlsx"), index=False
    )

    # Pathologist validation file (complete-cohort only), a handful of rows
    val_rows = []
    for tma in tmas:
        tma_dir = os.path.join(BASE_FOLDER, tma)
        patients_list = pd.read_excel(os.path.join(tma_dir, "Patients_list.xlsx"))
        for _, prow in patients_list.iloc[:2].iterrows():
            core = str(prow["Core"]).split(",")[0]
            val_rows.append({
                "TMA": tma, "Core": core, "Patient": prow["Patient ID"],
                "GATA6%-Pathologist1": int(rng.randint(0, 100)), "GATA6score-Pathologist1": int(rng.randint(0, 3)),
                "CK17%-Pathologist1": int(rng.randint(0, 100)), "CK17score-Pathologist1": int(rng.randint(0, 3)),
                "GATA6%-Pathologist2": float(rng.randint(0, 100)), "GATA6score-Pathologist2": float(rng.randint(0, 3)),
                "CK17%-Pathologist2": float(rng.randint(0, 100)), "CK17score-Pathologist2": float(rng.randint(0, 3)),
                "GATA6%-model": float(rng.randint(0, 100)), "GATA6score-model": str(rng.randint(0, 3)),
                "CK17%-model": float(rng.randint(0, 100)), "CK17score-model": str(rng.randint(0, 3)),
            })
    pd.DataFrame(val_rows).to_excel(
        os.path.join(INTERNAL_DIR, "Validation_comparation_synthetic.xlsx"), index=False
    )


def build_reproducibility(tmas, main_patient_ids):
    fresh_dir(REPRO_BASE)

    # Workflow (control-TMA) reproducibility: runs for both cohorts. Each
    # batch folder is loaded the same way as a main-cohort TMA, so it needs
    # its own Patients_list.xlsx -- the same control-TMA cores/patients,
    # re-stained identically across all 4 batches.
    workflow_dir = os.path.join(REPRO_BASE, "Workflow")
    control_batches = ["SYN-CTRL-B1", "SYN-CTRL-B2", "SYN-CTRL-B3", "SYN-CTRL-B4"]
    control_patients = main_patient_ids[:len(STAINING_GROUPS)]
    for batch in control_batches:
        patients_list_rows = []
        for g_idx, (group, patient_id) in enumerate(zip(STAINING_GROUPS, control_patients)):
            group_dir = os.path.join(workflow_dir, batch, group)
            row_num = g_idx + 1
            code = grid_code(row_num, 0)
            write_core(group_dir, code, 50)
            patients_list_rows.append({
                "Patient ID": patient_id,
                "NAP": f"SYN-NAP-{patient_id}",
                "TMA": batch,
                "Core": f"{row_num}A",
                "Scan Date": pd.Timestamp("2026-01-15"),
                "Scan Time": "10:00:00",
                "Scan Profile": "BF_20X_SYN",
                "Staining Date": pd.Timestamp("2026-01-10"),
                "GATA6 Lot": "SYNLOT001",
                "CK17 Lot": 1,
                "GATA6 Staining Protocol": "Synthetic_Protocol_A",
                "CK17 Staining Protocol": "Synthetic_Protocol_A",
                "Batch": control_batches.index(batch) + 1,
            })
        pd.DataFrame(patients_list_rows).to_excel(
            os.path.join(workflow_dir, batch, "Patients_list.xlsx"), index=False
        )

    # Scanning reproducibility (complete-cohort only): the same TMA, scanned
    # on two different dates, so there's an actual repeat measurement to
    # check agreement on -- one date alone wouldn't test anything. Flat
    # per-core exports under <date>/<TMA>/, with TMA/date read back from
    # "Study level 5"/"4".
    scan_dates = ["2026-01-15", "2026-01-20"]
    for c_idx in range(2):
        code = grid_code(1, c_idx)
        # Same underlying core, small scan-to-scan noise around a shared mean.
        true_ck17 = float(np.clip(rng.normal(230, 20), 0, 255))
        true_gata6 = float(np.clip(rng.normal(110, 20), 0, 255))
        for scan_date in scan_dates:
            scanning_dir = os.path.join(REPRO_BASE, "Scanning", scan_date, tmas[0])
            os.makedirs(scanning_dir, exist_ok=True)
            df = pd.DataFrame({
                "Study level 4": [scan_date],
                "Study level 5": [tmas[0]],
                "Name": [f"Grid {code}"],
                "Mean Intensity CK17/Core": [float(np.clip(true_ck17 + rng.normal(0, 5), 0, 255))],
                "Mean Intensity GATA6/Core": [float(np.clip(true_gata6 + rng.normal(0, 5), 0, 255))],
            })
            df.to_csv(os.path.join(scanning_dir, f"Grid {code}_objects.tsv"), sep="\t", index=False)

    # Staining reproducibility (complete-cohort only): flat per-core exports
    # under <Batch>/, "Study level 5" holds the batch label.
    staining_dir = os.path.join(REPRO_BASE, "Staining")
    for batch in ["Batch1", "Batch2"]:
        batch_dir = os.path.join(staining_dir, batch)
        os.makedirs(batch_dir, exist_ok=True)
        for c_idx in range(2):
            code = grid_code(1, c_idx)
            df = pd.DataFrame({
                "Study level 4": ["2026-01-10"],
                "Study level 5": [batch],
                "Name": [f"Grid {code}"],
                "Mean Intensity CK17/Core": [float(np.clip(rng.normal(230, 20), 0, 255))],
                "Mean Intensity GATA6/Core": [float(np.clip(rng.normal(110, 20), 0, 255))],
            })
            df.to_csv(os.path.join(batch_dir, f"Grid {code}_objects.tsv"), sep="\t", index=False)


def write_readme():
    content = """# SYNTHETIC DATA -- NOT REAL PATIENTS

Everything under `example_data/` is randomly generated (see
`utilities/generate_example_data.py`, fixed seed). No real patient, image or
clinical measurement went into producing it. It exists only to let someone
run `src/main.py` end to end without access to the real cohort, and to let
the "Expected data layout" section of the top-level README be checked
against a real run instead of taken on faith.

All IDs, TMA names and lot/batch labels use a "SYN" prefix on purpose, so
nothing here could be mistaken for real data if copied elsewhere. The numeric
values (intensities, clinical variables, survival times) have no biological
meaning -- they are random draws shaped to stay in plausible ranges so the
pipeline's numeric steps (GMM fitting, normalization, statistical tests)
don't choke on degenerate input, nothing more.

See the "Try it with synthetic data" section of the top-level
[README.md](../README.md) for exactly which `src/main.py` constants to point
here (including one -- `EXPECTED_BY_COHORT` -- that needs clearing rather
than repointing, since it holds the real cohort's reference thresholds).
"""
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(content)


def main():
    fresh_dir(ROOT)
    main_patient_ids, tmas = build_main_cohort()
    build_clinical_and_rosters(main_patient_ids, tmas)
    build_reproducibility(tmas, main_patient_ids)
    write_readme()
    print(f"Synthetic example data written to {ROOT}/")


if __name__ == "__main__":
    main()
