""" Script plot_static.py """

"""
Static plotting utilities: raw/normalized intensity distributions per TMA,
per staining group and per core, plus the binary correlation heatmap between
the two markers' positive-cell calls.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde


def plot_intensity_kde_by_staining_group(
    data,
    staining_values,
    outpath: str,
    suffix: str = "",
    xlim=None,
):
    """
    Same layout as plot_intensity_kde, but one row per Staining group (instead of
    per TMA), to compare how different the GATA6/CK17 distributions are across the
    5 raw staining groups before fitting a GMM per group.
    """
    intensity_columns = {
        "GATA6": f"DAB Intensity - Nuclei{suffix}",
        "CK17": f"Mag Intensity - Cytoplasm{suffix}",
    }

    if xlim is None:
        xlim = (0, 260) if suffix == "" else (0, 1)

    fig, axes = plt.subplots(len(staining_values), 2, figsize=(14, 6 * len(staining_values)))
    if len(staining_values) == 1:
        axes = [axes]

    for i, sv in enumerate(staining_values):
        group_subset = data[data["Staining"] == sv]

        for j, (marker, intensity_col) in enumerate(intensity_columns.items()):
            ax = axes[i][j]
            color = "tab:orange" if marker == "GATA6" else "tab:blue"

            sns.kdeplot(
                group_subset[intensity_col].dropna(),
                ax=ax,
                fill=True,
                alpha=0.6,
                linewidth=1.2,
                color=color,
            )

            ax.set_title(f"{sv} - {marker} Intensity{(' (normalized)' if suffix else '')} (n={len(group_subset)})")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Density")
            ax.set_xlim(*xlim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close(fig)


def plot_violin(data, output_root):
    """
    Creates violin plots for the distribution of intensity values for different TMAs.
    It generates two subplots per TMA: one for DAB intensity in nuclei and one for Mag intensity in cytoplasm.
    Each subplot displays the intensity distribution by core, with individual cell intensities as points.

    Args:
        data (pd.DataFrame): DataFrame containing the processed data with columns such as 'Core',
                              the per-cell GATA6 nuclear intensity ('DAB Intensity - Nuclei'),
                              the per-cell CK17 cytoplasmic intensity ('Mag Intensity - Cytoplasm'),
                              and 'TMA'.
        output_root (str): Root output directory; pass the run's OUTPUT_ROOT so different runs
            don't overwrite each other's outputs.

    Returns:
        None: This function saves the plot as an SVG file.
    """

    # Define the different types of intensities and their corresponding columns
    intensity_columns = {
        'GATA6': 'DAB Intensity - Nuclei',
        'CK17': 'Mag Intensity - Cytoplasm'
    }

    # Get the unique TMAs from the dataset
    tmas = data['TMA'].unique()

    # Number of subplots needed (one per TMA)
    n_tmas = len(tmas)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_tmas, 2, figsize=(14, 6 * n_tmas))

    # Ensure axes is a 2D list even if there's only one subplot
    if n_tmas == 1:
        axes = [axes]

    # Iterate over each TMA and create the plots
    for i, tma in enumerate(tmas):
        # Filter data for the current TMA
        tma_data = data[data['TMA'] == tma]

        # Create a plot for each type of intensity (DAB and Mag)
        for j, (staining, intensity_col) in enumerate(intensity_columns.items()):
            ax = axes[i][j]  # Select the appropriate subplot for the current TMA and intensity type
            # Create the violin plot for the intensity
            sns.violinplot(x='Core', y=intensity_col, data=tma_data, ax=ax)

            # Customize the plot
            ax.set_title(f"{tma} - {staining} Intensity")
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Core")
            ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels if necessary
            ax.set_ylim(0, 260)

    # Adjust layout to prevent overlapping subplots
    plt.tight_layout()
    output_path = f"{output_root}/raw_intensities_violin/violin_plot.svg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_intensity_lines(data, output_root):
    """
    Creates a plot with one line per TMA, and each line representing the intensity distribution
    for each core. The x-axis represents intensity values, and the y-axis represents cores.
    Furtheremore, it calculates the mean and the standard deviation values for each stain pero TMA and plots those values as vertical lines in the plots.

    Args:
        data (pd.DataFrame): DataFrame containing the processed data with columns such as 'Core',
                              the per-cell GATA6 nuclear intensity ('DAB Intensity - Nuclei'),
                              the per-cell CK17 cytoplasmic intensity ('Mag Intensity - Cytoplasm'),
                              and 'TMA'.

    Returns:
        None: This function saves the plot as an SVG file.
    """

    # Define the different types of intensities and their corresponding columns
    intensity_columns = {
        'GATA6': 'DAB Intensity - Nuclei',
        'CK17': 'Mag Intensity - Cytoplasm'
    }

    # Get the unique TMAs from the dataset
    tmas = data['TMA'].unique()

    # Create a figure with subplots (one for each TMA, and two columns for DAB and Mag)
    fig, axes = plt.subplots(len(tmas), 2, figsize=(14, 8 * len(tmas)))

    if len(tmas) == 1:
        axes = [axes]

    # Define colors for the different statistics lines
    colors = {
        'Mean': 'black',
        '1σ': 'red',
        '2σ': 'blue',
        '3σ': 'green'
    }

    # Iterate over each TMA and create the plots
    for i, tma in enumerate(tmas):
        tma_data = data[data['TMA'] == tma]

        for j, (staining, intensity_col) in enumerate(intensity_columns.items()):
            ax = axes[i][j]

            # Plot each core's intensity distribution
            for core in tma_data['Core'].unique():
                core_data = tma_data[tma_data['Core'] == core]
                ax.plot(core_data[intensity_col], [core] * len(core_data), 'o', alpha=0.6, markersize=3)

            # Calculate statistics
            mean_val = tma_data[intensity_col].mean()
            std_val = tma_data[intensity_col].std()

            # Plot the vertical lines for statistics with different colors
            ax.axvline(mean_val, color=colors['Mean'], linestyle='solid', alpha=0.9, linewidth=1.5, label=f'Mean = {mean_val:.2f}')
            ax.plot([], [], color='gray', label=f'std = {std_val:.2f}')

            ax.axvline(mean_val + std_val, color=colors['1σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'+1σ = {mean_val + std_val:.2f}')
            ax.axvline(mean_val - std_val, color=colors['1σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'-1σ = {mean_val - std_val:.2f}')

            ax.axvline(mean_val + 2 * std_val, color=colors['2σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'+2σ = {mean_val + 2 * std_val:.2f}')
            ax.axvline(mean_val - 2 * std_val, color=colors['2σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'-2σ = {mean_val - 2 * std_val:.2f}')

            ax.axvline(mean_val + 3 * std_val, color=colors['3σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'+3σ = {mean_val + 3 * std_val:.2f}')
            ax.axvline(mean_val - 3 * std_val, color=colors['3σ'], linestyle='dashed', alpha=0.7, linewidth=1, label=f'-3σ = {mean_val - 3 * std_val:.2f}')

            # Set plot titles and labels
            ax.set_title(f"{tma} - {staining} Intensity")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Core")
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8,pad=6)
            ax.set_xlim(0, 260)

            # Add a legend for the statistics lines
            ax.legend(fontsize='small')

    plt.tight_layout()
    output_path = f"{output_root}/raw_intensities_lines/tmas_intensity_lines.svg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_intensity_kde(
    data,
    suffix: str = "",
    outpath: str = "output/distributions/distributions_all_TMAs.svg",
    xlim=None,
):
    """
    Creates a plot with one row per TMA, and each row contains two columns (GATA6 and CK17),
    with KDE plots for the overall intensity distribution for each TMA.

    This function supports plotting either raw intensities or normalized intensities by
    providing a suffix that is appended to the intensity column names.

    Args:
        data (pd.DataFrame):
            DataFrame containing at least the columns:
                - 'TMA'
                - the per-cell GATA6 nuclear intensity, 'DAB Intensity - Nuclei{suffix}'
                - the per-cell CK17 cytoplasmic intensity, 'Mag Intensity - Cytoplasm{suffix}'
        suffix (str):
            Suffix appended to intensity columns. Example:
                suffix=""      -> uses raw columns
                suffix="_norm" -> uses normalized columns (e.g., "..._norm")
        outpath (str):
            Output path where the SVG figure is saved.
        xlim (tuple | None):
            Optional x-axis limits (xmin, xmax). If None:
                - raw: uses (0, 260)
                - normalized (suffix != ""): uses (0, 1)

    Returns:
        None. Saves the plot as an SVG file.
    """

    # Define the different types of intensities and their corresponding columns
    intensity_columns = {
        "GATA6": f"DAB Intensity - Nuclei{suffix}",
        "CK17": f"Mag Intensity - Cytoplasm{suffix}",
    }

    # Sanity check: required columns
    required = ["TMA"] + list(intensity_columns.values())
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for plot_intensity_kde: {missing}")

    # Decide default x-axis range
    if xlim is None:
        xlim = (0, 260) if suffix == "" else (0, 1)

    # Get the unique TMAs from the dataset (stable order)
    tmas = sorted(data["TMA"].dropna().unique().tolist())

    # Precompute global Y limits per marker (so your eye is not fooled)
    ymax_dict = {}

    x_grid = np.linspace(xlim[0], xlim[1], 1000)

    for staining, intensity_col in intensity_columns.items():
        ymax = 0.0
        for tma in tmas:
            vals = data.loc[data["TMA"] == tma, intensity_col].dropna()
            if len(vals) > 10:
                kde = gaussian_kde(vals)
                y = kde(x_grid)
                ymax = max(float(ymax), float(y.max()))
        ymax_dict[staining] = ymax * 1.05 if ymax > 0 else 1.0

    # Create a figure with subplots (one for each TMA, and two columns for GATA6 and CK17)
    fig, axes = plt.subplots(len(tmas), 2, figsize=(14, 6 * len(tmas)))

    if len(tmas) == 1:
        axes = [axes]

    # Iterate over each TMA and create the plots
    for i, tma in enumerate(tmas):
        tma_data = data[data["TMA"] == tma]

        for j, (staining, intensity_col) in enumerate(intensity_columns.items()):
            ax = axes[i][j]

            # Set consistent colors (as you had)
            color = "tab:orange" if staining == "GATA6" else "tab:blue"

            sns.kdeplot(
                tma_data[intensity_col],
                ax=ax,
                fill=True,
                alpha=0.6,
                linewidth=1.2,
                color=color,
            )

            ax.set_title(f"{tma} - {staining} Intensity{(' (normalized)' if suffix else '')}")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Density")
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8, pad=6)

            ax.set_xlim(*xlim)
            ax.set_ylim(0, ymax_dict[staining])

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def plot_overlapping_intensity_kde_per_core(data, output_root):
    """
    Creates a figure for each TMA with one subplot per core showing overlapping KDEs
    of DAB and Mag intensity values. Each subplot contains both curves in different colors.

    Args:
        data (pd.DataFrame): DataFrame with columns including 'TMA', 'Core', the per-cell GATA6
                             nuclear intensity ('DAB Intensity - Nuclei'), and the per-cell CK17
                             cytoplasmic intensity ('Mag Intensity - Cytoplasm').
        output_root (str): Root output directory; pass the run's OUTPUT_ROOT so different runs
            don't overwrite each other's outputs.

    Returns:
        None: Saves the figures in the specified directory.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np
    import os

    per_core_dir = f"{output_root}/distributions/per_core"
    os.makedirs(per_core_dir, exist_ok=True)

    tmas = data['TMA'].unique()

    for tma in tmas:
        tma_data = data[data['TMA'] == tma]
        cores = tma_data['Core'].unique()
        n_cores = len(cores)
        n_cols = 5
        n_rows = int(np.ceil(n_cores / n_cols))

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(f"TMA {tma} - Overlapping GATA6/CK17 Intensity KDE per Core", fontsize=16)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        for idx, core in enumerate(cores):
            core_data = tma_data[tma_data['Core'] == core]
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

            # KDE plots for GATA6 and CK17
            sns.kdeplot(core_data['DAB Intensity - Nuclei'], ax=ax, label='GATA6', fill=True, alpha=0.4, color='tab:blue')
            sns.kdeplot(core_data['Mag Intensity - Cytoplasm'], ax=ax, label='CK17', fill=True, alpha=0.4, color='tab:orange')

            ax.set_xlim(0, 260)
            ax.set_title(f"Core {core}")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Density")
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            if idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{per_core_dir}/distribution_{tma}_overlapping_per_core.svg"
        plt.savefig(filename)
        plt.close(fig)
        
def plot_binary_correlation_heatmap(data, output_root):
    """
    Creates a binary correlation heatmap (phi coefficient) between DAB+ and CK17+ per core for each TMA.
    Args:
        data (pd.DataFrame): DataFrame containing the processed data with columns such as the
            per-core GATA6-positive cell count ('DAB+ (#)'), the per-core CK17-positive cell
            count ('CK17+ (#)'), Core, and TMA.
        output_root (str): Root output directory. This plot is classification-derived, so pass the
            run's OUTPUT_ROOT to keep different cohort runs from overwriting each other.

    Returns:
        None: This function saves the heatmap plot as an SVG file.
    """

    # marker columns for DAB+ and CK17+
    marker_cols = ['DAB+ (#)', 'CK17+ (#)']
    # Get the unique TMAs from the dataset
    tmas = data['TMA'].unique()

    # Iterate over each TMA and calculate the phi coefficient for each core
    for tma in tmas:
        # Filter data for the current TMA
        tma_data = data[data['TMA'] == tma]

        # Get the unique cores in this TMA
        cores = tma_data['Core'].unique()

        correlations = []
        valid_cores = []

        for core in cores:
            core_data = tma_data[tma_data['Core'] == core]

            # Avoid errors if not enough data points or if columns have less than 2 unique values
            if len(core_data) < 2:
                continue
            if core_data[marker_cols[0]].nunique() < 2 or core_data[marker_cols[1]].nunique() < 2:
                continue

            # Calculate the phi coefficient for the current core
            phi, _ = pearsonr(core_data[marker_cols[0]], core_data[marker_cols[1]])
            correlations.append(phi)
            valid_cores.append(core)

        if not correlations:
            print(f"No valid data for TMA {tma}")
            continue

        # Creeate heatmap DataFrame
        correlation_df = pd.DataFrame([correlations], columns=valid_cores)

        plt.figure(figsize=(14, 6))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', cbar=True,
                    vmin=-1, vmax=1, xticklabels=valid_cores, yticklabels=[], fmt=".2f",
                    linewidths=0.5, annot_kws={"size": 8, "rotation": 90})
        plt.xticks(rotation=90)
        plt.title(f"Binary Correlation Heatmap (GATA6+ vs CK17+) for TMA {tma}")
        plt.xlabel('Core')
        plt.ylabel('Correlation (Phi Coefficient)')

        output_path = f"{output_root}/correlation/binary_correlation_heatmap_{tma}.svg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()