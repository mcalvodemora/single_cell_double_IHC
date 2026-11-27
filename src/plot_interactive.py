""" Script plot_interactive.py """

"""
Interactive (Plotly) scatter plots of DAB/Mag intensity per core and per
patient, one panel per TMA.

Confocal Microscopy Unit - CNIO : BioImage & Data Analysis
Developer: María Calvo de Mora
"""

import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

def plot_patient_core_scatter(data, output_root):
    """
    Creates an interactive scatter plot for each TMA where the x-axis represents the DAB intensity
    and the y-axis represents the Mag intensity. Each core is represented by a different color, and
    the legend allows toggling visibility of cores grouped by patient. Additionally, it fits a linear regression per patient.
    The regression line represented in the plot aims to illustrate the general trend of the relationship between DAB intensity and Mag intensity for each patient by adjusting a line that better fits the data points.
    If the slope of the line is positive, it means that as DAB intensity increases, Mag intensity also tends to increase. If the slope is negative, the opposite is true: as DAB increases, Mag decreases.
    If the slope is 0 (horizontal line) there is not a strong relationship between DAB and Mag intensity for that patient.

    Furthermore, it represents the mean DAB and mean Mag value as a bigger dot for each core.

    Args:
        data (pd.DataFrame): DataFrame containing the processed data with columns such as 'Core',
                              the per-cell GATA6 nuclear intensity ('DAB Intensity - Nuclei'),
                              the per-cell CK17 cytoplasmic intensity ('Mag Intensity - Cytoplasm'),
                              and 'TMA'.
        output_root (str): Root output directory; pass the run's OUTPUT_ROOT so different runs
            don't overwrite each other's outputs.

    Returns:
        None: This function directly saves the interactive plot.
    """

    data = data.sort_values(['Patient', 'Core'])
    tmas = data['TMA'].unique()
    n_rows = len(tmas)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=tmas)

    # Unique colors per core
    unique_cores = data['Core'].unique()
    colors = px.colors.qualitative.Dark24
    colors2 = px.colors.qualitative.Prism
    color_map = {core: colors[i % len(colors)] for i, core in enumerate(unique_cores)}
    color_map2 = {core: colors2[i % len(colors2)] for i, core in enumerate(unique_cores)}

    for i, tma in enumerate(tmas):
        tma_data = data[data['TMA'] == tma]
        patients = tma_data['Patient'].unique()

        for patient in patients:
            patient_data = tma_data[tma_data['Patient'] == patient]

            for core in patient_data['Core'].unique():
                core_data = patient_data[patient_data['Core'] == core]
                staining = core_data['Staining'].unique()[0]
                cell_count = core_data.shape[0]
                fig.add_trace(
                    go.Scatter(
                        x=core_data['DAB Intensity - Nuclei'],
                        y=core_data['Mag Intensity - Cytoplasm'],
                        mode='markers',
                        marker=dict(color=color_map[core], size=5),
                        name=f"{core} (Staining: {staining}, Cell Count: {cell_count:.2f})", 
                        legendgroup=f"{tma}-{patient}",
                        legendgrouptitle_text=f"{tma}-{patient}",
                        showlegend=True,
                        hoverinfo='text',
                        text=[
                            f"Patient: {patient}<br>Core: {core}<br>GATA6: {x:.2f}<br>CK17: {y:.2f}<br>Staining: {staining}<br>Cell Count: {cell_count:.2f}"
                            for x, y in zip(core_data['DAB Intensity - Nuclei'], core_data['Mag Intensity - Cytoplasm'])
                        ],
                        visible='legendonly'
                    ),
                    row=i + 1, col=1
                )

                # Calculate the mean DAB and Mag intensity for each core
                mean_dab = core_data['DAB Intensity - Nuclei'].mean()
                mean_mag = core_data['Mag Intensity - Cytoplasm'].mean()
                std_dab = core_data['DAB Intensity - Nuclei'].std()
                std_mag = core_data['Mag Intensity - Cytoplasm'].std()

                # Add a larger point for the mean DAB and Mag values
                fig.add_trace(
                    go.Scatter(
                        x=[mean_dab],
                        y=[mean_mag],
                        mode='markers',
                        marker=dict(color=color_map2[core], size=15, symbol='circle'),
                        name=f"{core} (μ GATA6= {mean_dab:.2f}, μ CK17= {mean_mag:.2f}, σ GATA6= {std_dab:.2f}, σ CK17= {std_mag:.2f})", 
                        legendgroup=f"{tma}-{patient}",
                        showlegend=True,
                        hoverinfo='text',
                        text=[f"Mean GATA6: {mean_dab:.2f}<br>Mean CK17: {mean_mag:.2f}"],
                        visible='legendonly'
                    ),
                    row=i + 1, col=1
                )

            # Linear regression per patient
            x = patient_data['DAB Intensity - Nuclei'].values.reshape(-1, 1)
            y = patient_data['Mag Intensity - Cytoplasm'].values
            if len(x) > 1:
                reg = LinearRegression().fit(x, y)
                x_range = np.linspace(x.min(), x.max(), 100)
                y_pred = reg.predict(x_range.reshape(-1, 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        line=dict(dash='dot', width=2, color='black'),
                        name=f"{patient} (reg)",
                        legendgroup=f"{tma}-{patient}",
                        showlegend=True,
                        visible='legendonly'
                    ),
                    row=i + 1, col=1
                )

    # Layout general
    fig.update_layout(
        title="GATA6 vs CK17 Intensity per patient and TMA",
        height=500 * n_rows,
        legend_title="TMA / Patient / Core",
        font=dict(size=12),
        margin=dict(t=80),
        title_x=0.5,
    )

    # Axis for each subplot
    for j in range(n_rows):
        fig.update_xaxes(title_text="GATA6 Intensity", range=[0, 300], row=j + 1, col=1)
        fig.update_yaxes(title_text="CK17 Intensity", range=[0, 300], row=j + 1, col=1)

    import os
    out_path = f"{output_root}/single_cell_interactive/tmas_scatter_interactive_by_patient.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, full_html=True, include_plotlyjs='cdn', config={"responsive": True})

def plot_patient_core_scatter_means(data, output_root):
    """
    Creates an interactive scatter plot for each TMA where the x-axis represents the GATA6 intensity
    and the y-axis represents the CK17 intensity. Each core is represented by a different color, and
    the legend allows toggling visibility of cores grouped by patient.
    Furthermore, it represents the mean GATA6 and mean CK17 value as a bigger dot for each core.

    Args:
        data (pd.DataFrame): DataFrame containing the processed data with columns such as 'Core',
                              the per-cell GATA6 nuclear intensity ('DAB Intensity - Nuclei'),
                              the per-cell CK17 cytoplasmic intensity ('Mag Intensity - Cytoplasm'),
                              and 'TMA'.
        output_root (str): Root output directory; pass the run's OUTPUT_ROOT so different runs
            don't overwrite each other's outputs.

    Returns:
        None: This function directly saves the interactive plot.
    """

    data = data.sort_values(['Patient', 'Core'])
    tmas = data['TMA'].unique()
    n_rows = len(tmas)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=tmas)

    # Unique colors per core
    unique_cores = data['Core'].unique()
    colors2 = px.colors.qualitative.Alphabet
    color_map2 = {core: colors2[i % len(colors2)] for i, core in enumerate(unique_cores)}

    for i, tma in enumerate(tmas):
        tma_data = data[data['TMA'] == tma]
        patients = tma_data['Patient'].unique()

        for patient in patients:
            patient_data = tma_data[tma_data['Patient'] == patient]

            for core in patient_data['Core'].unique():
                core_data = patient_data[patient_data['Core'] == core]
                staining = core_data['Staining'].unique()[0]
                cell_count = core_data.shape[0]

                # Calculate the mean DAB and Mag intensity for each core
                mean_dab = core_data['DAB Intensity - Nuclei'].mean()
                mean_mag = core_data['Mag Intensity - Cytoplasm'].mean()
                std_dab = core_data['DAB Intensity - Nuclei'].std()
                std_mag = core_data['Mag Intensity - Cytoplasm'].std()

                fig.add_trace(
                    go.Scatter(
                        x=[mean_dab],
                        y=[mean_mag],
                        mode='markers',
                        marker=dict(color=color_map2[core], size=15, symbol='circle'),
                        name=f"{core} (μ GATA6= {mean_dab:.2f}, μ CK17= {mean_mag:.2f}, σ GATA6= {std_dab:.2f}, σ CK17= {std_mag:.2f}, Staining: {staining}, Cell Count: {cell_count:.2f})", 
                        legendgroup=f"{tma}-{patient}",
                        legendgrouptitle_text=f"{tma}-{patient}",
                        showlegend=True,
                        hoverinfo='text',
                        text=[f"Patient: {patient}<br>Core: {core}<br>Mean GATA6: {mean_dab:.2f}<br>Mean CK17: {mean_mag:.2f}<br>Staining: {staining}<br>Cell Count: {cell_count:.2f}"],
                    ),
                    row=i + 1, col=1
                )

    # Layout general
    fig.update_layout(
        title="GATA6 vs CK17 Intensity Mean per patient and TMA",
        height=500 * n_rows,
        legend_title="TMA / Patient / Core",
        font=dict(size=12),
        margin=dict(t=80),
        title_x=0.5,
    )

    # Axis for each subplot
    for j in range(n_rows):
        fig.update_xaxes(title_text="GATA6 Intensity", range=[0, 300], row=j + 1, col=1)
        fig.update_yaxes(title_text="CK17 Intensity", range=[0, 300], row=j + 1, col=1)

    import os
    out_path = f"{output_root}/single_cell_interactive/tmas_scatter_interactive_by_patient_onlyMeans.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, full_html=True, include_plotlyjs='cdn', config={"responsive": True})
