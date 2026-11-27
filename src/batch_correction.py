# batch_correction.py
"""
Staining-batch correction using the control TMA as a reference material.

Method: ratio-based batch-effect correction with a concurrently profiled
reference material (Yu et al., "Assessing and mitigating batch effects in
large-scale omics studies", Genome Biology 2024; 25:254, section
"Location-scale methods" and "Multiomics Quartet project"). The control TMA
(the same tissue, stained in every staining round together with the cohort
TMAs of that round) is the reference: whatever changes in it between rounds
is technical by construction, so it measures the batch effect without
touching the patients' biology. Patient and batch are fully confounded in
this cohort (each patient sits in one TMA and one batch), which rules out
methods that estimate batch from the study samples themselves (ComBat, SVA,
RUV); the ratio-based method is the one Yu et al. single out for exactly
that confounded scenario.

How the factors are computed (compute_control_factors):
    1. Per-cell optical density on the control TMA: OD = -log10(I / 255).
       Bright-field pixel intensity is not proportional to the amount of
       chromogen, OD approximately is (Beer-Lambert), so the ratio is taken
       on OD. Note the raw scale is inverted: 0 = strongly stained,
       255 = white.
    2. Mean OD per control core, per batch.
    3. Only cores present in every batch are kept (paired comparison: the
       same tissue in each round).
    4. Batch level = mean of those per-core means.
    5. factor_b = level(reference batch) / level(b). The reference batch gets
       factor 1 and its TMAs are left untouched (cf. the "reference batch"
       idea of M-ComBat in Yu et al.).
One factor per batch and per intensity column ("feature by feature"), the
same for every staining group.

How they're applied (apply_batch_correction): every cell of every TMA gets
the factor of ITS BATCH (not of its TMA), on the OD scale, and is converted
back to intensity: I_corr = 255 * 10^(-OD * factor), i.e.
I_corr/255 = (I/255)^factor. That is a power curve, not a linear rescale,
which is why the per-TMA q01-q99 normalization that runs afterwards (a linear
min-max) does not undo it: the two steps fix different things (staining-round
loss of chromogen vs slide-specific scale). A plain ratio on raw intensity
would be erased entirely by that normalization.

IMPORTANT - the factors are ALWAYS computed from the full control TMA (every
core present in all batches), never from a cohort-filtered subset: the
control TMA is a technical calibrator, not part of the study cohort, so the
factors must not depend on COHORT. Callers must not apply the clean-roster
filter before compute_control_factors(); the factors are therefore identical
for COHORT="clean" and COHORT="complete".

Must run BEFORE the per-TMA q01-q99 normalization (normalization.py): that
normalization rescales each TMA on its own and would absorb the absolute
between-batch difference the control TMA is meant to measure.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from data_preprocessing import load_data_if_exists, drop_cores_with_all_nan_patients

# Every raw intensity column that reaches classification: the primary columns
# (q01-q99 normalization + GMM fit) and the Phenotyping columns (projected
# onto the same normalized scale and used to classify cells).
DEFAULT_CORRECTION_COLS: List[str] = [
    "DAB Intensity - Nuclei",
    "DAB Intensity - Nuclei Phenotyping",
    "Mag Intensity - Cytoplasm",
    "Mag Intensity - Cytoplasm Phenotyping",
]

_SCALE_MAX = 255.0


def to_od(intensity: pd.Series) -> pd.Series:
    """Pixel intensity (0-255, 0 = most stained) -> optical density."""
    return -np.log10(np.clip(intensity.astype(float), 1.0, _SCALE_MAX) / _SCALE_MAX)


def from_od(od: pd.Series) -> pd.Series:
    """Optical density -> pixel intensity, clipped to [1, 255]."""
    return np.clip(_SCALE_MAX * np.power(10.0, -od), 1.0, _SCALE_MAX)


def load_control_for_calibration(
    base_folder: str,
    processed_folder: Optional[str] = None,
) -> pd.DataFrame:
    """Load the full control TMA (all batches, all patients) for calibration.

    Same loading/cleaning as run_workflow_reproducibility() (reproducibility.py),
    including its default processed-cache location, but deliberately WITHOUT
    the clean-roster filter (see module docstring).
    """
    if processed_folder is None:
        processed_folder = os.path.join(
            os.path.dirname(os.path.abspath(base_folder)),
            "processed",
            os.path.basename(base_folder),
        )
    control = load_data_if_exists(base_folder, processed_folder)
    control = drop_cores_with_all_nan_patients(control)
    control = control.dropna(subset=["DAB Intensity - Nuclei", "Mag Intensity - Cytoplasm"])
    return control


def compute_control_factors(
    control_df: pd.DataFrame,
    reference_batch,
    intensity_cols: List[str] = DEFAULT_CORRECTION_COLS,
    batch_col: str = "Batch",
    core_col: str = "Core",
    outdir: Optional[str] = None,
) -> pd.DataFrame:
    """Per-batch, per-column multiplicative OD factors from the control TMA.

    Args:
        control_df: Cell-level control-TMA data (load_control_for_calibration).
        reference_batch: Batch value every other batch is rescaled to (factor 1).
        intensity_cols: Raw intensity columns to calibrate.
        batch_col: Batch column (from Patients_list.xlsx).
        core_col: Core column; cores are matched across batches by this.
        outdir: If given, the factor table is saved there as
            correction_factors.csv (supplementary table).

    Returns:
        Long DataFrame: batch_col, column, control_mean_OD, factor,
        n_common_cores.
    """
    if control_df[batch_col].isna().any():
        raise ValueError(f"Control TMA has cells with no '{batch_col}' value.")

    ctrl = control_df[[batch_col, core_col] + list(intensity_cols)].copy()
    for col in intensity_cols:
        ctrl[col] = to_od(ctrl[col])

    core_means = ctrl.groupby([batch_col, core_col])[list(intensity_cols)].mean().reset_index()
    n_batches = core_means[batch_col].nunique()
    batches_per_core = core_means.groupby(core_col)[batch_col].nunique()
    common_cores = batches_per_core[batches_per_core == n_batches].index
    if len(common_cores) == 0:
        raise ValueError("No control core is present in every batch: cannot calibrate.")
    core_means = core_means[core_means[core_col].isin(common_cores)]

    levels = core_means.groupby(batch_col)[list(intensity_cols)].mean()
    if reference_batch not in levels.index:
        raise ValueError(
            f"reference_batch={reference_batch!r} not found in the control TMA "
            f"batches {sorted(levels.index.tolist())}."
        )
    factors = levels.loc[reference_batch] / levels

    rows = []
    for batch in levels.index:
        for col in intensity_cols:
            rows.append({
                batch_col: batch,
                "column": col,
                "control_mean_OD": float(levels.loc[batch, col]),
                "factor": float(factors.loc[batch, col]),
                "n_common_cores": int(len(common_cores)),
            })
    factor_df = pd.DataFrame(rows)

    print(f"[BATCH CORRECTION] Factors from the control TMA "
          f"({len(common_cores)} cores present in all {n_batches} batches; "
          f"reference batch = {reference_batch}):")
    print(factor_df.pivot(index=batch_col, columns="column", values="factor").round(3).to_string())

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "correction_factors.csv")
        factor_df.to_csv(out_path, index=False)
        print(f"[BATCH CORRECTION] saved → {out_path}")
    return factor_df


def apply_batch_correction(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    batch_col: str = "Batch",
) -> pd.DataFrame:
    """Correct every intensity column in `factors` with the factor of each cell's batch.

    Returns a copy; the corrected values replace the original columns, so
    everything downstream (normalization, GMM, classification) uses them
    transparently. Fails loudly if any cell has no batch or a batch with no
    control-TMA factor, rather than leaving it silently uncorrected.
    """
    df = df.copy()
    if df[batch_col].isna().any():
        raise ValueError(f"{int(df[batch_col].isna().sum())} cells have no '{batch_col}' value: cannot correct them.")
    missing = set(df[batch_col].unique()) - set(factors[batch_col].unique())
    if missing:
        raise ValueError(f"No control-TMA factor for batch(es) {sorted(missing)}.")

    for col in factors["column"].unique():
        factor_by_batch = factors[factors["column"] == col].set_index(batch_col)["factor"]
        df[col] = from_od(to_od(df[col]) * df[batch_col].map(factor_by_batch))
    return df
