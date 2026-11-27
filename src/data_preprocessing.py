"""data_preprocessing.py"""

"""
Loads the raw per-core TSV/Excel exports from Visiopharm for each TMA, merges
cell-level (per_object) and core-level (per_image) measurements with the
patient/core metadata in Patients_list.xlsx, and caches the result as one CSV
per TMA so later runs can skip reprocessing. Also provides the cohort-roster
filtering, cell-export, and validation helpers used by main.py.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

from typing import Optional
import pandas as pd
import numpy as np
import os
import re
import glob
import openpyxl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.neighbors import KernelDensity


def load_data_if_exists(base_folder, processed_folder):
    """
    Loads and processes TMA data. If a processed CSV file exists for a TMA, it loads the file directly.
    Otherwise, it processes the raw data..

    Args:
        base_folder (str): Path to the root directory containing raw data (e.g., 'data/raw').
        processed_folder (str): Path to the folder where processed data will be saved (e.g., 'data/processed').

    Returns:
        pd.DataFrame: A concatenated DataFrame containing data from all TMAs, either loaded from CSV files
                      or newly processed.
    """
    tma_folders = os.listdir(base_folder)
    processed_data = []

    for tma in tma_folders:
        # Avoid reading folder named "Glands"
        if tma == "Glands":
            continue

        processed_tma_folder = os.path.join(processed_folder, tma)
        processed_tma_file = os.path.join(processed_tma_folder, f"{tma}_processed.csv")

        if os.path.exists(processed_tma_file):
            # If the processed file exists, load it
            print(f"Processed file found for {tma}. Loading {processed_tma_file}.")
            processed_data.append(pd.read_csv(processed_tma_file))
        else:
            # If the processed file doesn't exist, load and process the raw data
            print(f"Processed file not found for {tma}. Processing {tma}.")
            tma_data = load_data(os.path.join(base_folder, tma), processed_folder=processed_folder)
            processed_data.append(tma_data)

    data = pd.concat(processed_data, ignore_index=True)
    column_types = {
        "Core": "str",
        "Staining": "str",
        "DAB Intensity - Cell": "float",
        "Mag Intensity - Cell": "float",
        "Mag Intensity - Cytoplasm": "float",
        "DAB Intensity - Nucleus": "float",
        "TMA": "str",
    }

    # Adjust columns to the specified types
    for col, dtype in column_types.items():
        if col in data.columns:
            data[col] = data[col].astype(dtype)

    return data


def load_data(base_folder, processed_folder):
    """
    Loads and processes TSV files from the 'per_object' and 'per_image' subdirectories across multiple stain groups
    into a single DataFrame for a specific TMA, and saves it in the 'data\\processed\\tma' directory.

    Args:
        base_folder (str): Path to the root data directory for one TMA (e.g., 'data/raw/<tma>').
        processed_folder (str): Path to the folder where processed data will be saved.

    Returns:
        pd.DataFrame: A concatenated DataFrame with processed data from the TMA.
    """
    os.makedirs(
        processed_folder, exist_ok=True
    )  # Create the 'processed' directory if it doesn't exist.

    tma = os.path.basename(base_folder)  # Get the TMA name from the folder path
    tma_processed_folder = os.path.join(processed_folder, tma)
    os.makedirs(
        tma_processed_folder, exist_ok=True
    )  # Create the 'processed\\tma' directory if it doesn't exist.

    # Search for the excel patients file
    excel_path = os.path.join(base_folder, "Patients_list.xlsx")
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"The excel file wasn't found {excel_path}")

    data_list = []
    image_data_list = []

    for staining in os.listdir(base_folder):
        staining_path = os.path.join(base_folder, staining)
        if not os.path.isdir(staining_path):
            continue  # Ignore individual files as excel

        # per_object files
        staining_object_path = os.path.join(staining_path, "per_object")
        if not os.path.exists(staining_object_path):
            continue
        files = glob.glob(os.path.join(staining_object_path, "*.tsv"))
        if not files:
            continue  # Skip if no TSV files found

        for file in files:
            df = pd.read_csv(file, sep="\t")
            if df.empty:
                raise ValueError(f"The file {file} is empty or contains no valid data.")

            ## Mantain same Object Info order as the original file for future reimport to Visiopharm if needed
            # Create a directory to save the order of the Object IDs for each core in the processed folder, in case we need to reimport the data to Visiopharm and mantain the same order of cells.
            # The file will be named as {core}_raw_ID_order.csv and will contain two columns: "Object Info - Object ID" and "RawOrder" (the order of the Object IDs in the original file, starting from 0)
            object_order_dir = os.path.join(tma_processed_folder, "object_raw_ID_order")
            os.makedirs(object_order_dir, exist_ok=True)
            # Obtain core name from the file name (is cotained between () in the file name, ex. "1, A" in "Low_dab_(1, A).tsv")
            file_name = os.path.basename(file).split(".")[0]  # Remove extension
            # Create a DataFrame with the Object Info  and the RawOrder (the order of the Object IDs in the original file, starting from 0)
            objinfo_cols = [col for col in df.columns if col.startswith("Object Info")]
            order_df = df[objinfo_cols].copy()
            order_df.insert(1, "RawOrder", range(len(order_df)))
            # Save the order DataFrame to a CSV file in the object_raw_ID_order directory
            order_file_path = os.path.join(object_order_dir, f"{file_name}.csv")
            order_df.to_csv(order_file_path, index=False)

            df["TMA"] = tma
            # Staining group is known directly from the folder this TSV came from
            # (more robust than guessing it from a "Study level N" column, since that
            # column index isn't guaranteed to be consistent across files/exports).
            df["Staining"] = staining
            data_list.append(df)

        # per_image files
        staining_image_path = os.path.join(staining_path, "per_image")
        if not os.path.exists(staining_image_path):
            continue
        image_files = glob.glob(os.path.join(staining_image_path, "*.tsv"))
        if not image_files:
            continue
        for image_file in image_files:
            image_df = pd.read_csv(image_file, sep="\t")
            if image_df.empty:
                raise ValueError(
                    f"The file {image_file} is empty or contains no valid data."
                )
            image_df["TMA"] = tma
            image_df["Staining"] = staining
            image_data_list.append(image_df)

    # Concatenate per_object
    if data_list:
        # Concatenate the DataFrames for this TMA
        tma_df = pd.concat(data_list, ignore_index=True)

        # Process the concatenated DataFrame
        tma_df = process_data(tma_df, excel_path)

    # Concatenate per_image
    if image_data_list:
        # Concatenate the DataFrames for this TMA
        tma_image_df = pd.concat(image_data_list, ignore_index=True)

        # Process the concatenated DataFrame
        tma_image_df = process_data(tma_image_df, excel_path)

    # Merge the per_image data with the per_object data and repeat the values in the image Dataframe for all rows in the core - Broadcasting
    if "tma_df" in locals() and "tma_image_df" in locals():
        tma_df = pd.merge(
            tma_df, tma_image_df, on=["Core", "TMA", "Staining"], how="outer"
        )
        tma_df.to_csv(
            os.path.join(tma_processed_folder, f"{tma}_processed.csv"), index=False
        )

    return tma_df


def process_data(df, excel_path):
    """
    Processes the loaded DataFrame by performing the following operations:
    - Removes specific columns.
    - Renames columns.
    - Modifies the 'core' column by extracting a specific part of the string.
    - Assigns patient identifiers based on the 'Core' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        patient_excel_path (str): Path to the Excel with patient-core correspondence.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    # "Staining" was already set in load_data() from the folder name each TSV
    # came from (DAB++/Low_dab/Low_mag+dab/Mag++/Mag+DAB). Guessing it instead
    # from a "Study level N" column isn't reliable: that column's index isn't
    # guaranteed to stay consistent across files/exports and can silently
    # produce a NaN Staining value.
    study_level_cols = [col for col in df.columns if col.startswith("Study level")]
    df.rename(columns={"Name": "Core"}, inplace=True)

    columns_to_remove = [
        "Image",
        "LayerData",
        "Object Info - ROIName",
        "Object Info - ROIType",
        "Object Info - LabelType",
        "Object Info - LabelName",
        "Object Info - Envelope left",
        "Object Info - Envelope top",
        "Object Info - Envelope right",
        "Object Info - Envelope bottom",
        "Core ID",
        "Original Image",
        "Grid ID",
        "Core Area",
    ]
    df.drop(columns=columns_to_remove, errors="ignore", inplace=True)
    df.drop(columns=study_level_cols, errors="ignore", inplace=True)

    # Only maintain "(x,x)", which is the core identifier
    df["Core"] = df["Core"].str.extract(r"\(([^)]+)\)$")

    # Patient-core correspondence, built only from the object files.
    # Patients_list.xlsx columns are already in English (Patient ID, Core, Scan
    # Date, Scan Time, Scan Profile, Staining Date, GATA6 Lot, CK17 Lot, GATA6
    # Staining Protocol, CK17 Staining Protocol, Batch), so nothing needs
    # translating here. There's no NAP column; the patient identifier is the
    # hospital ID.
    patient_df = pd.read_excel(excel_path)

    core_to_hospitalID = {}
    core_to_scan_date = {}
    core_to_scan_time = {}
    core_to_scan_profile = {}
    core_to_staining_date = {}
    core_to_gata6_lot = {}
    core_to_ck17_lot = {}
    core_to_gata6_protocol = {}
    core_to_ck17_protocol = {}
    core_to_batch = {}

    for _, row in patient_df.iterrows():
        cores = str(row["Core"]).replace(" ", "").split(",")
        hospitalID = row["Patient ID"]
        for core in cores:
            number = core[:-1]
            letter = core[-1]
            if number.isdigit():
                number = number.zfill(2)
            formatted_core = f"{number}, {letter}"
            core_to_hospitalID[formatted_core] = hospitalID
            core_to_scan_date[formatted_core] = row["Scan Date"]
            core_to_scan_time[formatted_core] = row["Scan Time"]
            core_to_scan_profile[formatted_core] = row.get("Scan Profile", None)
            core_to_staining_date[formatted_core] = row["Staining Date"]
            core_to_gata6_lot[formatted_core] = row["GATA6 Lot"]
            core_to_ck17_lot[formatted_core] = row["CK17 Lot"]
            core_to_gata6_protocol[formatted_core] = row.get("GATA6 Staining Protocol", None)
            core_to_ck17_protocol[formatted_core] = row.get("CK17 Staining Protocol", None)
            core_to_batch[formatted_core] = row.get("Batch", None)

    # Normalize core column (in casi is ex. 1 instead of 01) to match with the core_to_hospitalID dictionary
    def normalize_core(core_str):
        try:
            number, letter = core_str.replace(" ", "").split(",")
            if not number.startswith("0") and number.isdigit():
                number = number.zfill(2)
            return f"{number}, {letter}"
        except (ValueError, AttributeError):
            return core_str

    df["Core"] = df["Core"].apply(normalize_core)

    # Create Patient column only on object files
    # Object files have a column called 'Diameter per cell' which is not present in image files
    if "Diameter per cell" in df.columns:
        df["ID"] = df["Core"].map(core_to_hospitalID)
        df["Patient"] = df["ID"]
        df["Scan Date"] = df["Core"].map(core_to_scan_date)
        df["Scan Time"] = df["Core"].map(core_to_scan_time)
        df["Scan Profile"] = df["Core"].map(core_to_scan_profile)
        df["Staining Date"] = df["Core"].map(core_to_staining_date)
        df["Lot GATA6"] = df["Core"].map(core_to_gata6_lot)
        df["Lot CK17"] = df["Core"].map(core_to_ck17_lot)
        df["Protocol GATA6"] = df["Core"].map(core_to_gata6_protocol)
        df["Protocol CK17"] = df["Core"].map(core_to_ck17_protocol)
        df["Batch"] = df["Core"].map(core_to_batch)

    # Dtypes
    for col in ["Scan Date", "Staining Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Scan Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce").dt.time
            
    # Check for null values
    # df[df.isnull().any(axis=1)]  # Uncomment this line to see rows with NaN values if needed.

    return df


def drop_cores_with_all_nan_patients(df):
    """
    Removes all rows belonging to cores (Core + TMA combinations) where
    the 'Patient' column is entirely NaN.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with cores with all Patient=NaN removed.
    """

    # Identify Core-TMA combinations where all Patient values are NaN
    cores_all_nan = (
        df.groupby(["Core", "TMA"])["Patient"]
        .apply(lambda x: x.isna().all())
        .reset_index()
    )

    # Keep only Core-TMA combinations where Patient is not all NaN
    cores_to_keep = cores_all_nan[cores_all_nan["Patient"] == False][["Core", "TMA"]]

    # Merge to retain only valid Core-TMA combinations
    df_filtered = df.merge(cores_to_keep, on=["Core", "TMA"], how="inner")

    return df_filtered

def export_cells_per_core_in_raw_order(
    cell_df: pd.DataFrame,
    processed_folder: str,
    tma_col: str = "TMA",
    staining_col: str = "Staining",
    core_col: str = "Core",
    object_id_col: str = "Object Info - Object ID",
    order_subdir: str = "object_raw_ID_order",
    out_root: Optional[str] = None,
):
    """
    Export per-core cell tables in the exact same row order as the original Visiopharm TSV exports.

    Expected:
      - order files (object_raw_ID_order/*.csv) contain:
          * Object Info - Object ID
          * RawOrder
          * (optionally) any other columns starting with "Object Info"
      - cell_df contains per-cell measures including:
          * Object Info - Object ID
          * DAB+ (#): per-cell GATA6 positivity flag (1 positive, 0 negative)
          * CK17+ (#): per-cell CK17 positivity flag (1 positive, 0 negative)

    Output:
      - final_cells_raw_order/*.txt per core, tab-delimited, ordered by RawOrder, containing
        exactly 2 columns:
          * GATA6_positive  (renamed from DAB+ (#))
          * CK17_positive   (renamed from CK17+ (#))
        All Object Info columns are excluded from the output.

    Output location:
      - By default (out_root=None) each core file is written under
        <processed_folder>/<tma>/final_cells_raw_order/, alongside the order files.
      - If out_root is given, order files are still READ from processed_folder but the
        outputs are WRITTEN to <out_root>/<tma>/final_cells_raw_order/. Use this to keep
        the three GMM experiments from overwriting each other's per-core exports.
    """

    required_cols = {tma_col, staining_col, core_col, object_id_col}
    missing = required_cols - set(cell_df.columns)
    if missing:
        raise ValueError(f"cell_df missing required columns: {sorted(missing)}")

    # Iterate per TMA folder in processed_folder
    for tma in os.listdir(processed_folder):
        tma_dir = os.path.join(processed_folder, tma)
        if not os.path.isdir(tma_dir):
            continue

        order_dir = os.path.join(tma_dir, order_subdir)
        if not os.path.isdir(order_dir):
            continue

        order_files = sorted(glob.glob(os.path.join(order_dir, "*.csv")))
        if not order_files:
            continue

        out_tma_dir = tma_dir if out_root is None else os.path.join(out_root, tma)
        out_dir = os.path.join(out_tma_dir, "final_cells_raw_order")
        os.makedirs(out_dir, exist_ok=True)

        # Pre-filter by TMA
        df_tma = cell_df[cell_df[tma_col].astype(str) == str(tma)].copy()

        # Normalize ObjectID dtype once at TMA level (avoid mismatches)
        df_tma[object_id_col] = df_tma[object_id_col].astype(str).str.strip()

        for order_path in order_files:
            base = os.path.splitext(os.path.basename(order_path))[0]
            order_df = pd.read_csv(order_path)

            if object_id_col not in order_df.columns or "RawOrder" not in order_df.columns:
                raise ValueError(f"Order file malformed: {order_path}")

            # --------------------------------------------------
            # Extract valid core from parentheses: "(04, E)" or "(2, A)" etc.
            # Works for both:
            #   "H120-... (04, E)_objects"
            #   "Grid (2, A)_objects"
            # --------------------------------------------------
            all_matches = re.findall(r"\(([^)]+)\)", base)

            core_from_file = None
            for candidate in all_matches:
                candidate = candidate.strip()
                if re.match(r"^\d{1,2},\s*[A-Za-z]$", candidate):
                    core_from_file = candidate
                    break

            if core_from_file is None:
                print(f"[WARN] Cannot parse valid core from: {base}")
                continue

            # --------------------------------------------------
            # Normalize core to format "0x, X"
            # --------------------------------------------------
            number, letter = core_from_file.replace(" ", "").split(",")
            if number.isdigit():
                number = number.zfill(2)
            core_from_file = f"{number}, {letter.upper()}"

            # --------------------------------------------------
            # Prepare order df: keep RawOrder + all Object Info columns
            # --------------------------------------------------
            order_df[object_id_col] = order_df[object_id_col].astype(str).str.strip()

            objinfo_cols = [c for c in order_df.columns if c.startswith("Object Info")]
            if object_id_col not in objinfo_cols:
                objinfo_cols = [object_id_col] + objinfo_cols

            order_small = order_df[objinfo_cols + ["RawOrder"]].copy()
            order_small["RawOrder"] = pd.to_numeric(order_small["RawOrder"], errors="coerce")
            order_small = order_small.dropna(subset=["RawOrder"])
            order_small["RawOrder"] = order_small["RawOrder"].astype(int)

            # --------------------------------------------------
            # Filter by TMA + Core
            # --------------------------------------------------
            sub = df_tma[df_tma[core_col].astype(str) == str(core_from_file)].copy()

            # Keep only IDs that appear in the order file
            sub = sub[sub[object_id_col].isin(order_small[object_id_col])].copy()

            if sub.empty:
                print(
                    f"[WARN] No matching cells for TMA={tma} core={core_from_file} ({base}). "
                    f"Possible causes: core mismatch, ObjectID dtype/spacing mismatch, or truly empty core."
                )
                continue

            # --------------------------------------------------
            # Keep only the requested cell-level columns
            # --------------------------------------------------
            keep_from_cells = [object_id_col, "DAB+ (#)", "CK17+ (#)"]
            missing_cells_cols = [c for c in keep_from_cells if c not in sub.columns]
            if missing_cells_cols:
                raise ValueError(f"Missing columns in cell_df for {base}: {missing_cells_cols}")

            sub_small = sub[keep_from_cells].copy()
            sub_small[object_id_col] = sub_small[object_id_col].astype(str).str.strip()

            # --------------------------------------------------
            # Merge: Object Info (from order file) + DAB/CK17 (from cell_df), keep order
            # --------------------------------------------------
            merged = order_small.merge(
                sub_small,
                on=object_id_col,
                how="left",
                validate="one_to_one",
            )

            merged = merged.sort_values("RawOrder").drop(columns=["RawOrder"])

            # Drop all Object Info columns; rename the two marker columns for Visiopharm reimport
            objinfo_to_drop = [c for c in merged.columns if c.startswith("Object Info")]
            merged = merged.drop(columns=objinfo_to_drop)
            merged = merged.rename(columns={"DAB+ (#)": "GATA6_positive", "CK17+ (#)": "CK17_positive"})

            out_path = os.path.join(out_dir, f"{base}.txt")
            merged.to_csv(out_path, index=False, sep="\t")

        print(f"[EXPORT] {tma} done → saved in final_cells_raw_order")
        
def validation_calculations(df, output_dir, thresholds, group_col="Staining"):
    """
    Compute per-core summary statistics from classified cells and save one
    validation CSV per TMA.

    For each core this aggregates the per-cell classification into:
    - percent positive cells per marker (GATA6, CK17);
    - H-score per marker: the standard semi-quantitative IHC score, combining
      the fraction of cells in each intensity tier (negative, 1+, 2+, 3+) into
      one weighted value, already computed per cell by add_core_hscores_to_cells()
      and just broadcast here;
    - counts and percentages of the four GATA6/CK17 combinations (GATA6+CK17+,
      GATA6+CK17-, GATA6-CK17+, GATA6-CK17-);
    - a heterogeneity metric (Shannon entropy over those four populations) and,
      separately, metrics based on the raw intensity values themselves
      (covariance area, spatial variance between cells).

    Also saves, per TMA, a per-core scatter plot of the normalized GATA6/CK17
    intensities with the classification thresholds marked, and a stacked bar
    plot of the four population percentages across cores.

    Args:
        df (pd.DataFrame): Classified cells. Must already contain the per-cell
            classification columns produced by apply_group_specific_thresholds()
            (DAB+ (#), CK17+ (#), and the Negative/1+/2+/3+ DAB/CK17 (#) tier
            counts), plus the per-core H-Score DAB / H-Score CK17 broadcast by
            add_core_hscores_to_cells().
        output_dir (str): Directory where the output CSV files will be saved.
        thresholds (dict): Normalized (0-1 scale) GATA6/CK17 thresholds, one
            entry per staining group, e.g.
            {"DAB++": {"GATA6": float, "CK17": float}, "Low_dab": {...}, ...}.
    Returns:
        pd.DataFrame: Concatenated validation results (one row per core) across all TMAs.
    """
    THRESHOLDS = thresholds

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Iterate over each TMA
    for tma, df_tma in df.groupby("TMA"):

        # Specific output directory for the TMA
        tma_dir = os.path.join(output_dir, str(tma))
        os.makedirs(tma_dir, exist_ok=True)

        # Group by core (and patient, if you want to keep it explicit)
        grouped = df_tma.groupby(["Core", "Patient"])

        results = []

        # Iterate over each core in a TMA and patient
        for (core, patient), group in grouped:

            # Obtain hospital ID
            first_row = group.iloc[0]
            codigo_hospital = first_row["ID"] if "ID" in first_row else np.nan
            # Identify the group this core belongs to, using the same grouping
            # column used to fit and assign the GMM thresholds (GROUP_COL in
            # main.py, normally "Staining"). Falls back to "Staining" if that
            # column isn't present.
            if group_col in group.columns:
                core_group = group[group_col].iloc[0]
            elif "Staining" in group.columns:
                core_group = group["Staining"].iloc[0]
            else:
                core_group = "Unknown"

            if core_group not in THRESHOLDS:
                raise KeyError(
                    f"No threshold for {group_col}='{core_group}' (TMA={tma}, core={core}). "
                    f"Check that the pasted `thresholds` dict has an entry for every group "
                    f"of '{group_col}'. Available keys: {sorted(THRESHOLDS.keys())}"
                )

            thr_gata6 = THRESHOLDS[core_group]["GATA6"]
            thr_ck17 = THRESHOLDS[core_group]["CK17"]

            total_cells = len(group)

            ## % of positivity per marker
            # DAB+ (#) and CK17+ (#) columns should contain 1 for positive cells and 0 for negative cells
            # But there are 1, 2 or more -> convert any row with value >=1 to 1
            group["DAB+ (#)"] = group["DAB+ (#)"].apply(lambda x: 1 if x >= 1 else 0)
            group["CK17+ (#)"] = group["CK17+ (#)"].apply(lambda x: 1 if x >= 1 else 0)
            # Percentage of positive cells over the total cells in the core
            pct_dab = (
                group["DAB+ (#)"].sum() / total_cells * 100 if total_cells > 0 else 0
            )
            pct_ck17 = (
                group["CK17+ (#)"].sum() / total_cells * 100 if total_cells > 0 else 0
            )

            ## Counts per intensity tier (per-cell 0/1 flags summed to core counts)
            count_dab_neg  = int(group["Negative DAB (#)"].sum())
            count_dab_1    = int(group["1+ DAB (#)"].sum())
            count_dab_2    = int(group["2+ DAB (#)"].sum())
            count_dab_3    = int(group["3+ DAB (#)"].sum())
            count_ck17_neg = int(group["Negative CK17 (#)"].sum())
            count_ck17_1   = int(group["1+ CK17 (#)"].sum())
            count_ck17_2   = int(group["2+ CK17 (#)"].sum())
            count_ck17_3   = int(group["3+ CK17 (#)"].sum())

            # Dominant intensity tier (the bin with the most cells)
            dab_tier_counts = pd.Series({
                "Negative": count_dab_neg, "1+": count_dab_1,
                "2+": count_dab_2,         "3+": count_dab_3,
            })
            max_dab = dab_tier_counts.idxmax()

            ck17_tier_counts = pd.Series({
                "Negative": count_ck17_neg, "1+": count_ck17_1,
                "2+": count_ck17_2,         "3+": count_ck17_3,
            })
            max_ck17 = ck17_tier_counts.idxmax()

            # H-score broadcasted by add_core_hscores_to_cells
            first_row = group.iloc[0]
            hscore_dab  = first_row["H-Score DAB"]
            hscore_ck17 = first_row["H-Score CK17"]

            # Population counts: DAB+CK17+, DAB+CK17-, DAB-CK17+, DAB-CK17-
            dab_pos = group["DAB+ (#)"] == 1
            ck17_pos = group["CK17+ (#)"] == 1

            count_dab_pos_ck17_pos = ((dab_pos) & (ck17_pos)).sum()
            count_dab_pos_ck17_neg = ((dab_pos) & (~ck17_pos)).sum()
            count_dab_neg_ck17_pos = ((~dab_pos) & (ck17_pos)).sum()
            count_dab_neg_ck17_neg = ((~dab_pos) & (~ck17_pos)).sum()

            # Percentages of the populations over the total cells in the core
            pct_dab_pos_ck17_pos = (
                count_dab_pos_ck17_pos / total_cells * 100 if total_cells > 0 else 0
            )
            pct_dab_pos_ck17_neg = (
                count_dab_pos_ck17_neg / total_cells * 100 if total_cells > 0 else 0
            )
            pct_dab_neg_ck17_pos = (
                count_dab_neg_ck17_pos / total_cells * 100 if total_cells > 0 else 0
            )
            pct_dab_neg_ck17_neg = (
                count_dab_neg_ck17_neg / total_cells * 100 if total_cells > 0 else 0
            )

            ## Heterogeneity metric: Shannon entropy per core (H = - sum(p_i * log2(p_i)) for each population i) H=[0,2]
            proportions = (
                np.array(
                    [
                        pct_dab_pos_ck17_pos,
                        pct_dab_pos_ck17_neg,
                        pct_dab_neg_ck17_pos,
                        pct_dab_neg_ck17_neg,
                    ]
                )
                / 100
            )
            # Filter out zero proportions to avoid log2(0)
            nonzero = proportions[proportions > 0]
            shannon_bits = (
                float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) > 0 else 0.0
            )
            shannon_norm = shannon_bits / np.log2(
                4
            )  # Normalize to [0,1] by dividing by log2(4)=2

            ## Metrics of heterogeneity based on raw intensity values (not populations)
            # Create a 2D array with the DAB Intensity - Nuclei and Mag Intensity - Cytoplasm values
            gata6_vals = group["DAB Intensity - Nuclei"].to_numpy()
            ck17_vals = group["Mag Intensity - Cytoplasm"].to_numpy()
            XY = np.column_stack([gata6_vals, ck17_vals])
            n_cells = XY.shape[0]
            # 1. Covariance matrix of the two intensities and its determinant
            if n_cells >= 2:
                cov = np.cov(
                    XY, rowvar=False
                )  # rowvar=False to have columns as variables
                det_cov = float(
                    np.linalg.det(cov)
                )  # determinant of the covariance matrix (is the )
                det_cov = max(det_cov, 0.0)  # avoid negative due to numerical errors
                cov_area = np.sqrt(
                    det_cov
                )  # square root of the determinant (interpretable as "area")
                # Obtain for interpretation var(x), var(y), cov(x,y)
                var_gata6 = cov[0, 0]
                var_ck17 = cov[1, 1]
                cov_gata6_ck17 = cov[0, 1]
                # Standard deviations
                std_gata6 = np.sqrt(var_gata6)
                std_ck17 = np.sqrt(var_ck17)
            else:
                det_cov = 0.0
                cov_area = 0.0
                var_gata6 = 0.0
                var_ck17 = 0.0
                cov_gata6_ck17 = 0.0
                std_gata6 = 0.0
                std_ck17 = 0.0

            # 2. Spatial variance: Average distance between points (cells) in the 2D intensity space
            if n_cells >= 2:
                dists = pdist(XY, metric="euclidean")
                spatial_variance = float(np.mean(dists))
            else:
                spatial_variance = 0.0
            # Spatial variance relative to the maximum possible distance in the intensity space (255,255)
            max_distance = np.sqrt((255**2) + (255**2))
            spatial_variance_relative = (
                spatial_variance / max_distance if max_distance > 0 else 0.0
            )

            # When saving, round floats to 4 decimals; counts stay integers (untouched)
            count_dab_neg = round(count_dab_neg, 0)
            pct_dab = round(pct_dab, 4)
            pct_ck17 = round(pct_ck17, 4)
            hscore_dab = round(hscore_dab, 4)
            hscore_ck17 = round(hscore_ck17, 4)
            pct_dab_pos_ck17_pos = round(pct_dab_pos_ck17_pos, 4)
            pct_dab_pos_ck17_neg = round(pct_dab_pos_ck17_neg, 4)
            pct_dab_neg_ck17_pos = round(pct_dab_neg_ck17_pos, 4)
            pct_dab_neg_ck17_neg = round(pct_dab_neg_ck17_neg, 4)
            shannon_bits = round(shannon_bits, 4)
            shannon_norm = round(shannon_norm, 4)
            variance_gata6 = round(var_gata6, 4)
            variance_ck17 = round(var_ck17, 4)
            covariance_gata6_ck17 = round(cov_gata6_ck17, 4)
            covariance_determinant = round(det_cov, 4)
            covariance_area = round(cov_area, 4)
            spatial_variance = round(spatial_variance, 4)
            spatial_variance_relative = round(spatial_variance_relative, 4)
            standard_deviation_gata6 = round(std_gata6, 4)
            standard_deviation_ck17 = round(std_ck17, 4)
            results.append(
                {
                    "TMA": tma,
                    "Core": core,
                    "Patient": patient,
                    "ID": codigo_hospital,
                    "Staining Date": first_row.get("Staining Date", np.nan),
                    "Scan Date": first_row.get("Scan Date", np.nan),
                    "% positive cell GATA6": pct_dab,
                    "% positive cell CK17": pct_ck17,
                    "Intensity based on H-score GATA6 (max 1+,2+,3+)": max_dab,
                    "Intensity based on H-score CK17 (max 1+,2+,3+)": max_ck17,
                    "Count GATA6-": count_dab_neg,
                    "Count GATA61+": count_dab_1,
                    "Count GATA62+": count_dab_2,
                    "Count GATA63+": count_dab_3,
                    "Count CK17-": count_ck17_neg,
                    "Count CK171+": count_ck17_1,
                    "Count CK172+": count_ck17_2,
                    "Count CK173+": count_ck17_3,
                    "H-Score GATA6": hscore_dab,
                    "H-Score CK17": hscore_ck17,
                    "Count GATA6+CK17+": count_dab_pos_ck17_pos,
                    "Count GATA6+CK17-": count_dab_pos_ck17_neg,
                    "Count GATA6-CK17+": count_dab_neg_ck17_pos,
                    "Count GATA6-CK17-": count_dab_neg_ck17_neg,
                    "% GATA6+CK17+": pct_dab_pos_ck17_pos,
                    "% GATA6+CK17-": pct_dab_pos_ck17_neg,
                    "% GATA6-CK17+": pct_dab_neg_ck17_pos,
                    "% GATA6-CK17-": pct_dab_neg_ck17_neg,
                    "Shannon entropy (bits)": shannon_bits,
                    "Shannon entropy (normalized)": shannon_norm,
                    "Variance GATA6": variance_gata6,
                    "Variance CK17": variance_ck17,
                    "Standard Deviation GATA6": standard_deviation_gata6,
                    "Standard Deviation CK17": standard_deviation_ck17,
                    "Covariance GATA6-CK17": covariance_gata6_ck17,
                    "Covariance determinant": covariance_determinant,
                    "Covariance area": covariance_area,
                    "Spatial variance": spatial_variance,
                    "Spatial variance (relative)": spatial_variance_relative,
                }
            )

            # Plot per core: scatter of normalized Phenotyping intensities with classification thresholds
            x_col = "DAB Intensity - Nuclei Phenotyping_norm"
            y_col = "Mag Intensity - Cytoplasm Phenotyping_norm"
            plt.figure(figsize=(8, 6))
            plt.scatter(group[x_col], group[y_col], alpha=0.5)
            ax_lim = (0, 1)
            display_lim = (-0.01, 1.05)
            plt.xlim(*display_lim)
            plt.ylim(*display_lim)
            ticks = [0, 0.25, 0.5, 0.75, 1.0]
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.axvline(x=thr_gata6, color="r", linestyle="--", label="GATA6 Threshold")
            plt.axhline(y=thr_ck17, color="b", linestyle="--", label="CK17 Threshold")
            plt.title(
                f"Intensity per cell - Core {core} in TMA {tma} - Group: {core_group}"
            )
            plt.xlabel("GATA6 Intensity - Nuclei Phenotyping (normalized)")
            plt.ylabel("CK17 Intensity - Cytoplasm Phenotyping (normalized)")
            plt.legend()
            offset = 0.01
            plt.text(thr_gata6 + offset, ax_lim[0] + offset, f"GATA6 Thr: {thr_gata6:.3f}", color="r")
            plt.text(ax_lim[0] + offset, thr_ck17 + offset, f"CK17 Thr: {thr_ck17:.3f}", color="b")
            plt.grid()

            plot_file = f"{tma}_Core_{core.replace(',','_').replace(' ','')}_intensity_scatter.svg"
            scatter_dir = os.path.join(os.path.dirname(output_dir), "scatter", str(tma))
            os.makedirs(scatter_dir, exist_ok=True)
            plt.savefig(os.path.join(scatter_dir, plot_file), dpi=150, bbox_inches="tight")
            plt.close()

        # Save results to a CSV file per TMA
        results_df = pd.DataFrame(results)
        # Clasify covariance are based on TMA metrics
        if not results_df.empty:
            Q1, Q3 = np.percentile(results_df["Covariance area"], [25, 75])
            IQR = Q3 - Q1

            def _label_local(x):
                if x <= Q1:
                    return "Low"
                if x <= Q3:
                    return "Medium"
                if x <= Q3 + 1.5 * IQR:
                    return "High"
                return "VeryHigh"

            results_df["Heterogeneity_local"] = results_df["Covariance area"].apply(
                _label_local
            )
        # Save for global results
        all_results.append(results_df)
        output_file = os.path.join(tma_dir, f"{tma}_validation.csv")
        results_df.to_csv(output_file, index=False)

        # Bar plot of populations per core (1 line per patient-core)
        results_df["PatientCore"] = (
            results_df["Patient"].astype(str) + "-" + results_df["Core"].astype(str)
        )
        # Sort by PatientCore (cores of the same patient together)
        plot_df = results_df.sort_values("PatientCore")
        pop_cols = ["% GATA6+CK17+", "% GATA6+CK17-", "% GATA6-CK17+", "% GATA6-CK17-"]
        # PDAC subtype labels for the legend
        subtype_labels = {
            "% GATA6+CK17+": "GATA6+CK17+ (Hybrid)",
            "% GATA6+CK17-": "GATA6+CK17- (Classical)",
            "% GATA6-CK17+": "GATA6-CK17+ (Basal)",
            "% GATA6-CK17-": "GATA6-CK17- (Undifferentiated)",
        }
        # Define a color map for the population columns
        color_map = {
            "% GATA6-CK17-": "#808080",  # grey
            "% GATA6-CK17+": "#d62728",  # red
            "% GATA6+CK17-": "#2ca02c",  # green
            "% GATA6+CK17+": "#98df8a",  # light green
        }
        labels = plot_df["PatientCore"].to_list()
        y_pos = np.arange(len(labels))
        plt.figure(figsize=(10, max(4, 0.35 * len(labels))))
        pop_vals = plot_df[pop_cols].fillna(0).values

        left = np.zeros(len(labels))
        for i, col in enumerate(pop_cols):
            vals = pop_vals[:, i]
            plt.barh(y_pos, vals, left=left, label=subtype_labels[col], color=color_map[col])
            left += vals  # acumulado para el siguiente tramo

        # Limit x axis to max 100 or max sum of row values if higher than 100
        row_sums = pop_vals.sum(axis=1)
        xlim_max = max(100, float(np.ceil(row_sums.max())))
        plt.xlim(0, xlim_max)

        plt.yticks(y_pos, labels)
        plt.xlabel("Percentage of cells")
        plt.title(f"Global cell composition by core (TMA {tma})")
        plt.legend(title="Phenotype (PDAC subtype)", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        stacked_file = f"{tma}_composition_stacked_by_core.svg"
        pop_comp_dir = os.path.join(os.path.dirname(output_dir), "population_composition")
        os.makedirs(pop_comp_dir, exist_ok=True)
        plt.savefig(os.path.join(pop_comp_dir, stacked_file), dpi=150, bbox_inches="tight")
        plt.close()

    # Classify covariance are based on global metrics (not by TMA)
    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)

        Q1g, Q3g = np.percentile(all_df["Covariance area"], [25, 75])
        IQRg = Q3g - Q1g

        def _label_global(x):
            if x <= Q1g:
                return "Low"
            if x <= Q3g:
                return "Medium"
            if x <= Q3g + 1.5 * IQRg:
                return "High"
            return "VeryHigh"

        all_df["Heterogeneity_global"] = all_df["Covariance area"].apply(_label_global)
        # Add the heterogeneity global column to each TMA csv file matching by TMA and Core
        for tma, df_tma in all_df.groupby("TMA"):
            tma_dir = os.path.join(output_dir, str(tma))
            tma_file = os.path.join(tma_dir, f"{tma}_validation.csv")
            if os.path.isfile(tma_file):
                tma_df_existing = pd.read_csv(tma_file)
                tma_df_merged = pd.merge(
                    tma_df_existing,
                    df_tma[["Core", "Heterogeneity_global"]],
                    on="Core",
                    how="left",
                )
                tma_df_merged.to_csv(tma_file, index=False)
    return all_df


def _check_model_columns_against_csvs(
    val_df: pd.DataFrame,
    model_csv_globs: list,
    tma_aliases: Optional[dict] = None,
) -> None:
    """Warn if the "-model" columns of a validation Excel no longer match the
    pipeline's per-core <TMA>_validation.csv files (i.e. the Excel was filled
    from an older run). Matching is on TMA + Core; % values are compared at
    the Excel's 1-decimal precision. Only prints; never modifies anything.
    """
    paths = sorted({p for pattern in model_csv_globs for p in glob.glob(pattern)})
    if not paths:
        print("  Model check skipped: no pipeline validation CSVs found.")
        return
    csv_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # A TMA can appear in more than one output folder (e.g. the control TMA in
    # both the cohort and the workflow-reproducibility outputs).
    csv_df = csv_df.drop_duplicates(["TMA", "Core"], keep="first").set_index(["TMA", "Core"])
    checks = {
        "GATA6%-model": "% positive cell GATA6",
        "CK17%-model": "% positive cell CK17",
        "GATA6score-model": "Intensity based on H-score GATA6 (max 1+,2+,3+)",
        "CK17score-model": "Intensity based on H-score CK17 (max 1+,2+,3+)",
    }
    missing, stale = [], []
    for _, row in val_df.iterrows():
        tma = str(row["TMA"]).strip()
        key = ((tma_aliases or {}).get(tma, tma), str(row["Core"]).strip())
        if key not in csv_df.index:
            missing.append(key)
            continue
        ref = csv_df.loc[key]
        for excel_col, csv_col in checks.items():
            if excel_col not in val_df.columns:
                continue
            new, old = ref[csv_col], row[excel_col]
            if "%" in excel_col:
                differs = abs(round(float(new), 1) - float(old)) > 0.05
            else:
                differs = str(new) != str(old)
            if differs:
                stale.append((key, excel_col, old, new))
    if missing:
        print(f"  WARNING: {len(missing)} cores not found in the pipeline CSVs: {missing[:5]}")
    if stale:
        print(f"  WARNING: {len(stale)} '-model' values differ from the current pipeline "
              f"output (Excel filled from an older run?). First: {stale[:5]}")
    else:
        print(f"  Model check: '-model' columns match the current pipeline output "
              f"({len(val_df) - len(missing)} cores).")


def run_model_pathologist_validation(
    excel_path: str,
    outdir: str,
    out_name: str = "model_pathologist_validation.xlsx",
    model_csv_globs: Optional[list] = None,
    tma_aliases: Optional[dict] = None,
) -> None:
    """
    Compare model output against manual pathologist annotations.

    Reads an Excel file with annotations from one or more pathologists
    (columns "...-Pathologist1", "...-Pathologist2", ...) and the model
    predictions, then computes, per marker (GATA6, CK17):

      With 2 pathologists:
        - Pearson correlation between pathologists (% and score)
        - Pearson correlation between model and mean of pathologists (% only)
        - MAE between pathologists, each pathologist vs model, and mean vs model
      With any number of pathologists:
        - Pearson correlation between each pathologist and the model (% only)

    Every metric is reported with its n (cores with both values present).
    Blank cells (e.g. cores a pathologist considered non-evaluable) are left
    out pairwise, metric by metric, and listed in the "Excluded" sheet.
    Pathologist and model % / pathologist score columns must be numeric or
    blank: negatives are entered as 0 (% and score), never as text.

    Expected Excel columns (per pathologist N and marker M in GATA6, CK17):
        "TMA", "Core", "Patient",
        "M%-PathologistN", "MscoreN-PathologistN",
        "M%-model", "Mscore-model"

    Results are printed to console and saved as an XLSX to outdir.

    Args:
        excel_path: Path to the pathologist-vs-model comparison spreadsheet.
            Fill "GATA6%-model" and "CK17%-model" with the model outputs
            (from validation_calculations CSV) before calling.
        outdir: Directory where the results XLSX will be saved.
        out_name: Output file name (one per scoring round).
        model_csv_globs: Optional glob patterns for the pipeline's
            <TMA>_validation.csv files. When given, the "-model" columns of
            the Excel are checked against them and a warning is printed if
            they are out of date. Results are always computed from the Excel.
        tma_aliases: Maps TMA names used in the Excel to pipeline TMA names
            (e.g. {"H12O-OPT1": "H12O-OPT12"}), only for that check.
    """
    val_df = pd.read_excel(excel_path)
    markers = ["GATA6", "CK17"]
    pathologists = sorted(
        {m.group(1) for c in val_df.columns if (m := re.fullmatch(r"(?:GATA6|CK17)%-(Pathologist\d+)", str(c)))},
        key=lambda p: int(p.replace("Pathologist", "")),
    )
    if not pathologists:
        raise ValueError(f"No '<marker>%-PathologistN' columns found in {excel_path}")
    print(f"Pathologists found: {', '.join(pathologists)} ({len(val_df)} cores)")

    # Numeric columns must hold numbers or blanks; text (e.g. "Negative", "?")
    # has to be resolved in the Excel, not silently dropped here.
    numeric_cols = [c for c in val_df.columns
                    if re.fullmatch(r"(GATA6|CK17)(%|score)-Pathologist\d+", str(c))
                    or c in ("GATA6%-model", "CK17%-model")]
    for col in numeric_cols:
        as_num = pd.to_numeric(val_df[col], errors="coerce")
        bad = val_df[col].notna() & as_num.isna()
        if bad.any():
            raise ValueError(
                f"Non-numeric values in '{col}': "
                f"{val_df.loc[bad, ['TMA', 'Core', col]].values.tolist()}"
            )
        val_df[col] = as_num

    if model_csv_globs:
        _check_model_columns_against_csvs(val_df, model_csv_globs, tma_aliases)

    def n_pairs(a: pd.Series, b: pd.Series) -> int:
        return int((a.notna() & b.notna()).sum())

    def pearson(a: pd.Series, b: pd.Series) -> float:
        return a.corr(b)

    def mae(a: pd.Series, b: pd.Series) -> float:
        return (a - b).abs().mean()

    corr_rows, mae_rows, excluded = [], [], []
    # Markers each pathologist actually scored (a round can be GATA6-only).
    scored = {
        mk: [p for p in pathologists
             if f"{mk}%-{p}" in val_df.columns and val_df[f"{mk}%-{p}"].notna().any()]
        for mk in markers
    }

    for mk in markers:
        paths = scored[mk]
        if not paths:
            print(f"\n{mk}: not scored by any pathologist, skipped.")
            continue
        model_col = f"{mk}%-model"
        for p in paths:
            blank = val_df[f"{mk}%-{p}"].isna()
            for _, r in val_df[blank].iterrows():
                excluded.append({"Marker": mk, "Pathologist": p, "TMA": r["TMA"],
                                 "Core": r["Core"], "Patient": r["Patient"],
                                 "Reason": f"blank {mk}%-{p} (not scored / non-evaluable)"})

        print(f"\n{mk}:")
        if len(paths) >= 2:
            p1, p2 = paths[0], paths[1]
            a_pct, b_pct = val_df[f"{mk}%-{p1}"], val_df[f"{mk}%-{p2}"]
            a_sc, b_sc = val_df[f"{mk}score-{p1}"], val_df[f"{mk}score-{p2}"]
            # Mean of pathologists: row-wise mean skipping blanks (as before).
            avg_col = f"{mk}_pct_avg"
            val_df[avg_col] = val_df[[f"{mk}%-{p}" for p in (p1, p2)]].mean(axis=1)
            avg = val_df[avg_col]
            rows = [
                (f"Inter-pathologist {mk} %", a_pct, b_pct),
                (f"Inter-pathologist {mk} score", a_sc, b_sc),
                (f"Model vs mean-pathologist {mk} %", val_df[model_col], avg),
            ]
            for label, x, y in rows:
                corr_rows.append({"Comparison": label, "Pearson r": round(pearson(x, y), 2),
                                  "n": n_pairs(x, y)})
                print(f"  r   {label:42s} {pearson(x, y):.2f}  (n={n_pairs(x, y)})")
            for label, x, y in [(f"Inter-pathologist {mk}", a_pct, b_pct)]:
                mae_rows.append({"Comparison": label, "MAE": round(mae(x, y), 2), "n": n_pairs(x, y)})
                print(f"  MAE {label:42s} {mae(x, y):.2f}  (n={n_pairs(x, y)})")
        for p in paths:
            x, y = val_df[f"{mk}%-{p}"], val_df[model_col]
            corr_rows.append({"Comparison": f"{p} vs Model {mk} %", "Pearson r": round(pearson(x, y), 2),
                              "n": n_pairs(x, y)})
            mae_rows.append({"Comparison": f"{p} vs Model {mk}", "MAE": round(mae(x, y), 2),
                             "n": n_pairs(x, y)})
            print(f"  r   {p + ' vs Model ' + mk + ' %':42s} {pearson(x, y):.2f}  (n={n_pairs(x, y)})")
            print(f"  MAE {p + ' vs Model ' + mk:42s} {mae(x, y):.2f}  (n={n_pairs(x, y)})")
        if len(paths) >= 2:
            x, y = val_df[f"{mk}_pct_avg"], val_df[model_col]
            label = f"Mean Pathologists vs Model {mk}"
            mae_rows.append({"Comparison": label, "MAE": round(mae(x, y), 2), "n": n_pairs(x, y)})
            print(f"  MAE {label:42s} {mae(x, y):.2f}  (n={n_pairs(x, y)})")

    # Same row order as the original two-pathologist report: grouped by
    # comparison type, then pathologist, then marker (GATA6 before CK17),
    # then % before score.
    kinds = ["Inter-pathologist", "Model vs mean", "Pathologist", "Mean Pathologists"]

    def _rank(row):
        c = row["Comparison"]
        kind = next(i for i, k in enumerate(kinds) if c.startswith(k))
        return (kind, c.split()[0] if kind == 2 else "", "CK17" in c, "score" in c)

    corr_rows, mae_rows = sorted(corr_rows, key=_rank), sorted(mae_rows, key=_rank)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, out_name)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(corr_rows).to_excel(writer, sheet_name="Correlations", index=False)
        pd.DataFrame(mae_rows).to_excel(writer, sheet_name="MAE", index=False)
        pd.DataFrame(excluded, columns=["Marker", "Pathologist", "TMA", "Core", "Patient", "Reason"]).to_excel(
            writer, sheet_name="Excluded", index=False)
        val_df.to_excel(writer, sheet_name="Raw data", index=False)
    print(f"\n  Results saved → {out_path}")
