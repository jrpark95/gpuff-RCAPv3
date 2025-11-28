"""
Interactive 3D Cloudshine Visualization - Plane Source Method
==============================================================
Creates interactive HTML visualization using Plotly for plane source cloudshine.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Constants from CUDA code
MU_AIR = 0.01    # Air attenuation coefficient [m^-1]
K_BUILD = 1.4    # Buildup correction factor
PI = np.pi

class PlaneSourceCalculator:
    """Calculate cloudshine using plane source method with 10 slabs."""

    def __init__(self, puff_x, puff_y, puff_z, sigma_y, sigma_z, activity, h_mix=1000.0):
        self.puff_x = puff_x
        self.puff_y = puff_y
        self.puff_z = puff_z
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.activity = activity
        self.h_mix = h_mix

        # Dose conversion factors
        self.dcf_sic = 1e-5
        self.dcf_pn = self.dcf_sic / 241.2

        # Generate 10 slab heights
        self.slab_heights = self._generate_slab_heights()

    def _generate_slab_heights(self):
        """Generate 10 slab heights using quantile coefficients."""
        c = np.array([0.127, 0.385, 0.674, 1.037, 1.645])
        z_slabs = []

        for ci in c:
            z_slabs.append(self.puff_z + ci * self.sigma_z)
            z_slabs.append(self.puff_z - ci * self.sigma_z)

        z_slabs = np.array(z_slabs)
        z_slabs = np.abs(z_slabs)  # Ground reflection

        if self.h_mix > 0:
            z_slabs = np.where(z_slabs > self.h_mix,
                             2.0 * self.h_mix - z_slabs,
                             z_slabs)

        return np.sort(z_slabs)

    def calculate_dose_rate(self, x, y, z=1.0):
        """Calculate dose rate at receptor position."""
        dx = x - self.puff_x
        dy = y - self.puff_y
        r = np.sqrt(dx**2 + dy**2)

        # Lateral Gaussian factor
        lateral = np.exp(-0.5 * (r / self.sigma_y)**2) / (2.0 * PI * self.sigma_y**2)

        # Buildup sum for all slabs
        sum_buildup = 0.0
        for z_s in self.slab_heights:
            dz = abs(z_s - z)
            buildup = (1.0 + K_BUILD * MU_AIR * dz) * np.exp(-MU_AIR * dz)
            sum_buildup += buildup

        # Total dose rate (mrem/hr)
        dose_rate = 0.1 * self.dcf_pn * lateral * sum_buildup * self.activity * 3.6e6
        return dose_rate

def create_interactive_plane_source(puff_params, grid_size=60):
    """Create interactive 3D visualization for plane source method."""

    print("Creating interactive plane source visualization...")

    # Initialize calculator
    calc = PlaneSourceCalculator(
        puff_params['x'], puff_params['y'], puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity'], puff_params.get('h_mix', 1000.0)
    )

    # Create ground grid
    extent = max(5 * puff_params['sigma_y'], 2000)
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Calculate dose field
    print(f"Calculating dose field ({grid_size}x{grid_size} grid)...")
    Z = np.zeros_like(X)
    for i in range(grid_size):
        if i % 20 == 0:
            print(f"  Progress: {i}/{grid_size}")
        for j in range(grid_size):
            Z[i, j] = calc.calculate_dose_rate(X[i, j], Y[i, j])

    print("Building interactive visualization...")

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.6, 0.4],
        specs=[[{'type': 'surface', 'rowspan': 2}, {'type': 'scatter'}],
               [None, {'type': 'bar'}]],
        subplot_titles=('3D Dose Surface (Plane Source Method)',
                       'Cross-Sections',
                       'Slab Contributions')
    )

    # 1. 3D Surface Plot
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='Dose Field',
            showscale=True,
            colorbar=dict(
                title=dict(text='Dose Rate<br>(mrem/hr)', side='right'),
                x=0.45,
                len=0.6,
                thickness=20
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="white",
                    project=dict(z=True)
                )
            ),
            hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Dose: %{z:.3e} mrem/hr<extra></extra>'
        ),
        row=1, col=1
    )

    # Add puff marker
    fig.add_trace(
        go.Scatter3d(
            x=[puff_params['x']],
            y=[puff_params['y']],
            z=[Z.max() * 1.1],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=['Puff Center'],
            textposition='top center',
            name='Puff Center',
            showlegend=False,
            hovertemplate='Puff Center<br>Height: %{z:.0f}m<extra></extra>'
        ),
        row=1, col=1
    )

    # Add sigma circles on ground
    theta = np.linspace(0, 2*np.pi, 100)
    for n_sigma, color in [(1, 'orange'), (2, 'yellow'), (3, 'cyan')]:
        circle_x = puff_params['x'] + n_sigma * puff_params['sigma_y'] * np.cos(theta)
        circle_y = puff_params['y'] + n_sigma * puff_params['sigma_y'] * np.sin(theta)
        circle_z = np.zeros_like(circle_x)

        fig.add_trace(
            go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{n_sigma}σ boundary',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # 2. Cross-sections
    center_idx = grid_size // 2

    # X cross-section
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Z[center_idx, :],
            mode='lines',
            name='X-axis (Y=0)',
            line=dict(color='blue', width=2),
            hovertemplate='X: %{x:.0f}m<br>Dose: %{y:.3e} mrem/hr<extra></extra>'
        ),
        row=1, col=2
    )

    # Y cross-section
    fig.add_trace(
        go.Scatter(
            x=y,
            y=Z[:, center_idx],
            mode='lines',
            name='Y-axis (X=0)',
            line=dict(color='green', width=2),
            hovertemplate='Y: %{x:.0f}m<br>Dose: %{y:.3e} mrem/hr<extra></extra>'
        ),
        row=1, col=2
    )

    # Add vertical line at puff center using scatter
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[min(Z[center_idx, :]), max(Z[center_idx, :])],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # 3. Slab contributions bar chart
    slab_contributions = []
    lateral_center = 1.0 / (2.0 * PI * calc.sigma_y**2)  # At center (r=0)

    for i, z_s in enumerate(calc.slab_heights):
        dz = abs(z_s - 1.0)  # Distance from receptor at 1m height
        buildup = (1.0 + K_BUILD * MU_AIR * dz) * np.exp(-MU_AIR * dz)
        contribution = 0.1 * calc.dcf_pn * lateral_center * buildup * calc.activity * 3.6e6
        slab_contributions.append(contribution)

    fig.add_trace(
        go.Bar(
            x=[f'Slab {i+1}<br>{h:.0f}m' for i, h in enumerate(calc.slab_heights)],
            y=slab_contributions,
            marker=dict(
                color=slab_contributions,
                colorscale='Plasma',
                showscale=False
            ),
            text=[f'{c:.2e}' for c in slab_contributions],
            textposition='outside',
            hovertemplate='%{x}<br>Contribution: %{y:.3e} mrem/hr<extra></extra>',
            name='Slab Dose'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Plane Source Cloudshine - Interactive 3D Visualization</b><br>' +
                 f'<sub>Puff: σ_y={puff_params["sigma_y"]:.0f}m, σ_z={puff_params["sigma_z"]:.0f}m, ' +
                 f'Height={puff_params["z"]:.0f}m, Activity={puff_params["activity"]:.2e} Ci</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Dose Rate (mrem/hr)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        xaxis2=dict(title='Distance (m)'),
        yaxis2=dict(title='Dose Rate (mrem/hr)', type='log'),
        xaxis3=dict(title='Slab', tickangle=-45),
        yaxis3=dict(title='Dose Contribution (mrem/hr)'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02
        ),
        height=900,
        width=1400
    )

    return fig

def create_slab_visualization(puff_params):
    """Create interactive visualization of the 10-slab structure."""

    calc = PlaneSourceCalculator(
        puff_params['x'], puff_params['y'], puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity']
    )

    fig = go.Figure()

    # Create mesh for each slab
    x_range = 3 * puff_params['sigma_y']
    grid_points = 40
    x = np.linspace(-x_range, x_range, grid_points)
    y = np.linspace(-x_range, x_range, grid_points)
    X, Y = np.meshgrid(x, y)

    # Add each slab as a surface
    colors = px.colors.sequential.Viridis
    n_colors = len(colors)

    for i, z_s in enumerate(calc.slab_heights):
        Z = np.ones_like(X) * z_s

        # Calculate Gaussian density at this height
        density = np.exp(-0.5 * ((X/puff_params['sigma_y'])**2 +
                                 (Y/puff_params['sigma_y'])**2))

        color_idx = int(i * n_colors / len(calc.slab_heights))

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=density,
            colorscale='Blues',
            opacity=0.3,
            showscale=False,
            name=f'Slab {i+1} ({z_s:.0f}m)',
            hovertemplate='Slab %{text}<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Height: %{z:.1f}m<extra></extra>',
            text=[[f'{i+1}' for _ in range(grid_points)] for _ in range(grid_points)]
        ))

    # Add puff center marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[puff_params['z']],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='diamond'),
        text=['Puff Center'],
        textposition='top center',
        name='Puff Center',
        hovertemplate='Puff Center<br>Height: %{z:.0f}m<extra></extra>'
    ))

    # Add ground plane
    Z_ground = np.zeros_like(X)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_ground,
        colorscale='Greys',
        opacity=0.2,
        showscale=False,
        name='Ground',
        hoverinfo='skip'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>10-Slab Structure Visualization</b><br>' +
                 f'<sub>Plane Source Method with σ_y={puff_params["sigma_y"]:.0f}m, σ_z={puff_params["sigma_z"]:.0f}m</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Height (m)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        showlegend=True,
        height=800,
        width=1200
    )

    return fig

def main():
    """Generate interactive HTML visualizations."""

    print("="*60)
    print("Interactive Plane Source Cloudshine Visualization")
    print("="*60)

    # Test scenarios
    scenarios = [
        {
            'name': 'medium_puff',
            'x': 0, 'y': 0, 'z': 100,
            'sigma_y': 500, 'sigma_z': 200,
            'activity': 1e4,
            'h_mix': 1000.0
        },
        {
            'name': 'large_puff',
            'x': 0, 'y': 0, 'z': 150,
            'sigma_y': 800, 'sigma_z': 350,
            'activity': 5e4,
            'h_mix': 1500.0
        },
        {
            'name': 'boundary_case',
            'x': 0, 'y': 0, 'z': 80,
            'sigma_y': 450, 'sigma_z': 380,
            'activity': 2e3,
            'h_mix': 800.0
        }
    ]

    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario['name']}")
        print(f"  σ_y={scenario['sigma_y']}m, σ_z={scenario['sigma_z']}m")

        # Create main interactive visualization
        fig_main = create_interactive_plane_source(scenario)
        filename_main = f"cloudshine_plane_{scenario['name']}_interactive.html"
        fig_main.write_html(filename_main)
        print(f"  [OK] Saved main visualization: {filename_main}")

        # Create slab structure visualization
        fig_slab = create_slab_visualization(scenario)
        filename_slab = f"cloudshine_slabs_{scenario['name']}_interactive.html"
        fig_slab.write_html(filename_slab)
        print(f"  [OK] Saved slab visualization: {filename_slab}")

    # Create comparison visualization
    print("\nCreating comparison visualization...")
    fig_compare = create_comparison_visualization()
    fig_compare.write_html("cloudshine_method_comparison.html")
    print("  [OK] Saved comparison: cloudshine_method_comparison.html")

    print("\n" + "="*60)
    print("Interactive visualizations complete!")
    print("Open the HTML files in your browser to explore.")
    print("="*60)

def create_comparison_visualization():
    """Create interactive comparison between 720-point and plane source methods."""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('720-Point Method (Small Puff)', 'Plane Source Method (Large Puff)'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )

    # Small puff (720-point regime)
    grid_size = 40
    extent = 500
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Simplified 720-point calculation
    Z1 = np.zeros_like(X)
    puff1 = {'x': 0, 'y': 0, 'z': 50, 'sigma_y': 100, 'sigma_z': 50, 'activity': 1e3}

    for i in range(grid_size):
        for j in range(grid_size):
            r = np.sqrt(X[i,j]**2 + Y[i,j]**2 + puff1['z']**2)
            Z1[i,j] = 1e-6 * puff1['activity'] / (4*PI*r**2) * np.exp(-MU_AIR*r) * 3.6e6

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z1, colorscale='Viridis', showscale=False,
                  name='720-Point'),
        row=1, col=1
    )

    # Large puff (plane source regime)
    Z2 = np.zeros_like(X)
    puff2 = {'x': 0, 'y': 0, 'z': 100, 'sigma_y': 500, 'sigma_z': 200, 'activity': 1e4}
    calc2 = PlaneSourceCalculator(
        puff2['x'], puff2['y'], puff2['z'],
        puff2['sigma_y'], puff2['sigma_z'], puff2['activity']
    )

    for i in range(grid_size):
        for j in range(grid_size):
            Z2[i,j] = calc2.calculate_dose_rate(X[i,j], Y[i,j])

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z2, colorscale='Viridis',
                  colorbar=dict(title='Dose Rate<br>(mrem/hr)', x=1.02),
                  name='Plane Source'),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Cloudshine Method Comparison</b><br>' +
                 '<sub>720-Point (σ<400m) vs Plane Source (σ≥400m)</sub>',
            x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Dose (mrem/hr)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene2=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Dose (mrem/hr)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600, width=1400
    )

    return fig

if __name__ == "__main__":
    main()