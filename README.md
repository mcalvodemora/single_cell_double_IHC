# Data-driven pipeline for clinical exploration of single-cell molecular signature and differentiation states with double tumor markers applied to pancreatic cancer

Image-analysis and clinical-correlation pipeline for double-IHC (GATA6/CK17)
tissue microarrays (TMAs) from pancreatic cancer, supporting the accompanying
manuscript. Cells are classified GATA6+/− and CK17+/− using per-staining-group
Gaussian Mixture Model (GMM) thresholds, aggregated to core/patient level,
checked for batch effects and reproducibility, and correlated with clinical
data.

This repository contains **code and synthetic data only**. Patient clinical data and
whole-slide/TMA image-derived measurements are not included — see
[Data availability](#data-availability) below.

## Requirements

- Python 3.10, see [environment.yml](environment.yml) for exact package
  versions (create the environment with
  `conda env create -f environment.yml`, then `conda activate panc-dualmarker`).

## Running the pipeline

The whole pipeline is one script, run from the repository root:

```bash
python src/main.py
```

It is a plain script (not a package/CLI) by design: the constants at the very
top of [src/main.py](src/main.py) cover every parameter needed to point the
pipeline at a new run of the same kind of data — same TMA/staining-group
layout, same clinical variables (see
[Adapting this to different data](#adapting-this-to-different-data) below for
what that excludes). The parameters are:

| Constant | Meaning |
|---|---|
| `COHORT` | `"clean"` restricts everything to the final publication roster (`CLEAN_ROSTER_PATH`); `"complete"` uses all patients with valid image material and additionally runs the complete-cohort-only reproducibility/inter-observer validation steps. |
| `OUTPUT_ROOT` | Derived from `COHORT` (`output/clean_cohort` or `output/complete_cohort`); everything the run produces is written under here. |
| `GROUP_COL` | Column used to group cells before fitting the per-group GMMs (`"Staining"`). |
| `FIT_GMM` | `True` refits the GMMs and saves metrics/thresholds/figure; `False` reuses what is already saved on disk under `OUTPUT_ROOT/gmm/` (no refit). |
| `BATCH_CORRECTION` | `True` corrects the staining-batch effect with the control TMA as reference material before normalization (see [Staining-batch correction](#staining-batch-correction)); `False` skips it. Same `OUTPUT_ROOT` either way (a run overwrites the other). |
| `REFERENCE_BATCH` | Staining batch (`Batch` column of `Patients_list.xlsx`) the other batches are rescaled to (factor 1, left untouched). |
| `base_folder` / `processed_folder` | Raw TMA export and its processed-cache location. |
| `CLINICAL_EXCEL_PATH` | Clinical database Excel. |
| `CLEAN_ROSTER_PATH` / `EXCLUDED_ROSTER_PATH` | Patient rosters used by the `COHORT` filter and the bias assessment (`"clean"` / `"complete"` respectively). |
| `PATHOLOGIST_VALIDATION_PATH` | Inter-observer validation spreadsheet (`COHORT="complete"` only). |
| `SCANNING_REPRO_FOLDER` / `STAINING_REPRO_FOLDER` / `STAINING_BATCH_ORDER` | Scanning/staining reproducibility inputs and the batch order to plot them in (`COHORT="complete"` only). |
| `WORKFLOW_REPRO_FOLDER` / `WORKFLOW_BATCH_ORDER` | Control-TMA (same tissue, re-stained across batches) reproducibility input and batch order (both cohorts). |
| `EXPECTED_BY_COHORT` | Keyed `"<cohort>"` (no batch correction) or `"<cohort>_batchcorr"` (with it). Reference `k`/threshold per staining group and marker, recorded once a GMM fit has been reviewed, so that routine reruns don't require re-fitting or manually re-checking the GMMs every time: each run's obtained values are automatically compared against this reference, and the pipeline stops before classification if they disagree. On the first run for a given cohort there is no entry yet, so the check is skipped with a warning — review the fit, then paste the printed values in as the new reference for future reruns. |

## Staining-batch correction

Optional (`BATCH_CORRECTION`), implemented in
[src/batch_correction.py](src/batch_correction.py). The cohort TMAs were
stained in several rounds (`Batch` in `Patients_list.xlsx`), and staining
intensity drifts between rounds. The control TMA (`WORKFLOW_REPRO_FOLDER`,
the same tissue stained in every round together with that round's cohort
TMAs) is used as a reference material for a ratio-based correction (Yu et
al., *Assessing and mitigating batch effects in large-scale omics studies*,
Genome Biology 2024; 25:254). Patient and batch are fully confounded in this
cohort (each patient sits in one TMA and one batch), so the batch effect is
estimated only on the control TMA, never on the patients themselves.

1. Per-cell optical density on the control TMA, `OD = -log10(I/255)` (the raw
   scale is inverted: 0 = strongly stained, 255 = white; OD, unlike pixel
   intensity, is approximately proportional to the amount of chromogen).
2. Mean OD per control core and batch, keeping only the cores present in
   every batch (paired comparison).
3. Batch level = mean of those per-core means;
   `factor_b = level(REFERENCE_BATCH) / level(b)`, one per batch and per
   intensity column (GATA6/CK17, primary and Phenotyping columns).
4. Every cell of every cohort TMA is corrected with its batch's factor on the
   OD scale and converted back to intensity, `I_corr = 255 · 10^(−OD · factor_b)`.

The correction runs after the PRE batch-effect diagnosis (which always
describes the uncorrected raw data) and **before** the per-TMA q01–q99
normalization, which would otherwise absorb the between-batch difference the
control TMA measures. On the intensity scale the correction is a power curve,
`(I/255)^factor`, which the normalization (a linear min–max per TMA) does not
undo. The factors are saved to `OUTPUT_ROOT/batch_correction/correction_factors.csv`,
and the same factors are applied to the control TMA in the
workflow-reproducibility step.

**The factors are always computed from the full control TMA** (all cores
present in every batch), regardless of `COHORT`: the control TMA is a
technical calibrator, not part of the study cohort, so the clean-roster filter
is never applied to it for calibration and the factors are identical for
`"clean"` and `"complete"`.

Comparing the POST batch-effect diagnosis (`OUTPUT_ROOT/batch_effect_post/`)
of a run with `BATCH_CORRECTION=False` against one with `True` measures the
effect of the correction, since nothing else differs between the two runs.

## Try it with synthetic data

This repository contains no real data (see [Data availability](#data-availability)
below), so there's nothing to run the pipeline against out of the box. A small,
fully synthetic dataset shaped like the real one — random values, fake
IDs, see [example_data/README.md](example_data/README.md) — is included under
`example_data/` for exactly this: to confirm the published code runs end to end.
To try it, in `src/main.py`'s config block:

- point `base_folder`, `processed_folder`, `CLINICAL_EXCEL_PATH`,
  `EXCLUDED_ROSTER_PATH`, `CLEAN_ROSTER_PATH`, `PATHOLOGIST_VALIDATION_PATH`,
  `SCANNING_REPRO_FOLDER` and `STAINING_REPRO_FOLDER` at the matching files/
  folders under `example_data/` instead of the real `data/` /
  `internal_documentation/` ones (same names, see
  [Expected data layout](#expected-data-layout) below);
- set `STAINING_BATCH_ORDER = ["Batch1", "Batch2"]` and `WORKFLOW_BATCH_ORDER
  = ["SYN-CTRL-B1", "SYN-CTRL-B2", "SYN-CTRL-B3", "SYN-CTRL-B4"]` (the
  synthetic control-TMA only has those batches, not the real ones);
- clear `EXPECTED_BY_COHORT = {}` — its numbers are the real cohort's
  reference thresholds, so the reproducibility check would (correctly) fail
  against synthetic data otherwise.

Then run `python src/main.py` as usual. The example data is regenerated by
`python utilities/generate_example_data.py` (fixed seed).

## Expected data layout

Not included in this repository. This describes what the code actually reads
(`data_preprocessing.py`), not a specific folder name — the root itself is
whatever `base_folder` in `main.py`'s config block points to:

```
<base_folder>/
  <TMA>/                              one folder per tissue microarray
    Patients_list.xlsx                core-to-patient metadata (see below)
    <staining_group>/                 DAB++ | Low_dab | Low_mag+dab | Mag++ | Mag+DAB
      per_object/*.tsv                 one row per cell
      per_image/*.tsv                  one row per core
```

- `<TMA>` is used verbatim as the "TMA" value throughout the pipeline.
- The 5 `<staining_group>` names are fixed (the code groups GMM fitting and
  one of the diagnostic plots by these exact 5 labels); everything else
  (`<base_folder>`, `<TMA>` names) is free-form.
- Each `per_object`/`per_image` TSV is a native Visiopharm per-core export:
  its `Name` column must end in the core identifier in parentheses, e.g.
  `"... (1, A)"`. `per_object` (cell-level) is distinguished from `per_image`
  (core-level) by the presence of a `Diameter per cell` column.
- `Patients_list.xlsx` has one row per patient, with columns `Patient ID`,
  `Core` (comma-separated core codes for that patient on this TMA, e.g.
  `"1A, 5A"`), `Scan Date`, `Scan Time`, `Staining Date`, `GATA6 Lot`,
  `CK17 Lot`, and optionally `Scan Profile`, `GATA6 Staining Protocol`,
  `CK17 Staining Protocol`, `Batch`.

Reproducibility inputs live under their own root folders (`SCANNING_REPRO_FOLDER`,
`STAINING_REPRO_FOLDER`, `WORKFLOW_REPRO_FOLDER`), but not all in the same shape:

- `WORKFLOW_REPRO_FOLDER` (control TMA, same tissue re-stained across
  batches; both cohorts) follows the exact same `<batch>/<staining_group>/
  {per_object,per_image}/*.tsv` + `Patients_list.xlsx` layout as above —
  each batch is loaded like its own TMA.
- `SCANNING_REPRO_FOLDER` (`<date>/<TMA>/*.tsv`) and `STAINING_REPRO_FOLDER`
  (`<Batch>/*.tsv`) are flat, one-row-per-core exports instead — no
  staining-group subfolders, no `per_object`/`per_image` split, no
  `Patients_list.xlsx`. Columns: `Study level 4` (date), `Study level 5`
  (TMA name or batch label), `Name`, `Mean Intensity GATA6/Core`, `Mean
  Intensity CK17/Core`. Both are `COHORT="complete"` only.

Clinical data (`CLINICAL_EXCEL_PATH`) is a single Excel file, one row per
patient.

`internal_documentation/` holds the patient-roster spreadsheets referenced in
the parameter table above (`CLEAN_ROSTER_PATH`, `EXCLUDED_ROSTER_PATH`,
`PATHOLOGIST_VALIDATION_PATH`), each a simple patient/core listing.

`data/`, `output/` and `internal_documentation/` are git-ignored.

## Adapting this to different data

The constants in `main.py` are run parameters: which cohort, where this run's
data lives, whether to refit the GMMs. They're enough to rerun the pipeline on
a new batch of data shaped like the layout above — same markers, same staining
groups, same clinical variables.

They are not enough to adapt the method itself to a different marker pair,
staining panel, or set of clinical variables. That requires editing the
module that owns the relevant definition directly, near the top of the file:

| Module | What it defines |
|---|---|
| `clinical.py` | Which clinical columns are read and how they're typed/encoded (`NUMERIC_COLS`, `CATEGORICAL_COLS`, `INT_ENCODED_CAT_COLS`, `DATE_COLS`, `META_COLS`). |
| `patient_aggregation.py` | The 4 GATA6/CK17 population definitions and the grade column carried through to the patient level (`POPULATION_COLS`, `GRADE_COL`). |
| `bias_assessment.py` | Which clinical variables are compared between included and excluded patients (`_CONT`, `_CAT`). |
| `batch_effect.py`, `normalization.py`, `gmm_thresholds.py` | The two marker intensity columns and the 5 staining-group names/order (`DEFAULT_PRE_INTENSITY_COLS`, `DEFAULT_INTENSITY_COLS`, `GMM_GROUP_ORDER`). |
| `batch_correction.py` | Which raw intensity columns get the staining-batch correction (`DEFAULT_CORRECTION_COLS`). |

## Data availability

The clinical and image-derived patient data used in this study are not
publicly available due to patient privacy restrictions, but are available
from the corresponding author upon reasonable request, subject to the
relevant institutional and ethics approvals.

## License

MIT — see [LICENSE](LICENSE).
