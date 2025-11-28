"""
Cloudshine Visualization - Semi-Infinite Cloud Method
======================================================
Visualizes cloudshine dose using the semi-infinite cloud approximation
for large vertical dispersion (σ_z ≥ 400m) as implemented in GPUFF-RCAPv3.

Based on RASCAL 4 NUREG-1940 methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Constants from CUDA code
MU_AIR = 0.01    # Air attenuation coefficient [m^-1]
K_BUILD = 1.4    # Buildup correction factor
PI = np.pi

class SemiInfiniteCloudshine:
    """
    Calculate cloudshine dose using semi-infinite cloud approximation.
    Used when vertical dispersion is very large (σ_z ≥ 400m).
    """

    def __init__(self, puff_x, puff_y, puff_z, sigma_y, sigma_z, activity, h_mix=None):
        """
        Initialize semi-infinite cloud calculator.

        Args:
            puff_x, puff_y, puff_z: Puff center coordinates (m)
            sigma_y, sigma_z: Dispersion parameters (m)
            activity: Total activity (Ci)
            h_mix: Mixing layer height (m), if None uses 2*sigma_z
        """
        self.puff_x = puff_x
        self.puff_y = puff_y
        self.puff_z = puff_z
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.activity = activity

        # Mixing height - use larger of specified value or 2*sigma_z
        if h_mix is None:
            self.h_mix = max(1000.0, 2 * sigma_z)
        else:
            self.h_mix = max(h_mix, 2 * sigma_z)

        # Dose conversion factor for semi-infinite cloud
        self.dcf_sic = 1e-5  # [(rem/s)/(Ci/m^3)]

    def calculate_chi_over_q(self, x, y):
        """
        Calculate χ/Q (chi over Q) - atmospheric dispersion factor.
        This represents concentration per unit source strength.

        For semi-infinite cloud: χ/Q = exp(-0.5*(r/σ_y)²) / (2π σ_y² H)

        Returns:
            χ/Q in (s/m³)
        """
        dx = x - self.puff_x
        dy = y - self.puff_y
        r = np.sqrt(dx**2 + dy**2)

        # Lateral Gaussian dispersion
        lateral = np.exp(-0.5 * (r / self.sigma_y)**2) / (2.0 * PI * self.sigma_y**2)

        # Uniform vertical mixing assumption
        chi_over_q = lateral / self.h_mix

        return chi_over_q

    def calculate_dose_rate(self, x, y, z=1.0):
        """
        Calculate cloudshine dose rate at receptor position.

        For semi-infinite cloud: D = Q × (χ/Q) × DCF_sic

        Args:
            x, y, z: Receptor coordinates (m)

        Returns:
            Dose rate (mrem/hr)
        """
        chi_over_q = self.calculate_chi_over_q(x, y)

        # Dose rate = Activity × dispersion factor × DCF
        dose_rate = self.activity * chi_over_q * self.dcf_sic

        # Convert to mrem/hr
        return dose_rate * 3.6e6

    def get_cloud_profile(self, x, y):
        """
        Get vertical concentration profile at position (x,y).
        For semi-infinite cloud, concentration is uniform with height.
        """
        chi_over_q = self.calculate_chi_over_q(x, y)
        concentration = self.activity * chi_over_q  # Ci/m³

        # Uniform from ground to mixing height
        heights = np.linspace(0, self.h_mix, 100)
        concentrations = np.ones_like(heights) * concentration

        return heights, concentrations

def visualize_semi_infinite_3d(puff_params):
    """
    Create 3D visualization of semi-infinite cloud dose field.
    """

    print(f"Creating semi-infinite cloud visualization...")
    print(f"  σ_y={puff_params['sigma_y']}m, σ_z={puff_params['sigma_z']}m")
    print(f"  Mixing height={puff_params.get('h_mix', 2*puff_params['sigma_z'])}m")

    # Create calculator
    calc = SemiInfiniteCloudshine(
        puff_params['x'], puff_params['y'], puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity'], puff_params.get('h_mix')
    )

    # Create ground grid
    extent = max(5 * puff_params['sigma_y'], 3000)
    grid_size = 50
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Calculate dose field
    Z = np.zeros_like(X)
    for i in range(grid_size):
        if i % 10 == 0:
            print(f"  Progress: {i}/{grid_size}")
        for j in range(grid_size):
            Z[i, j] = calc.calculate_dose_rate(X[i, j], Y[i, j])

    # Create matplotlib visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9,
                           edgecolor='none', antialiased=True)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Dose Rate (mrem/hr)')
    ax1.set_title('Semi-Infinite Cloud Dose Surface')
    ax1.view_init(elev=25, azim=45)

    # Mark puff center
    ax1.scatter([puff_params['x']], [puff_params['y']], [Z.max()],
               color='red', s=100, marker='*', label='Puff Center')

    # Top-down contour view
    ax2 = fig.add_subplot(222)
    levels = 20
    contour = ax2.contourf(X, Y, Z, levels=levels, cmap='plasma')
    ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    ax2.plot(puff_params['x'], puff_params['y'], 'r*', markersize=15, label='Puff Center')

    # Add sigma boundaries
    theta = np.linspace(0, 2*np.pi, 100)
    for n_sigma, style in [(1, '--'), (2, ':'), (3, '-.')]:
        circle_x = puff_params['x'] + n_sigma * puff_params['sigma_y'] * np.cos(theta)
        circle_y = puff_params['y'] + n_sigma * puff_params['sigma_y'] * np.sin(theta)
        ax2.plot(circle_x, circle_y, 'r' + style, label=f'{n_sigma}σ_y')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (Semi-Infinite Cloud)')
    ax2.legend()
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2, label='Dose Rate (mrem/hr)')

    # Cross-section
    ax3 = fig.add_subplot(223)
    center_y = X.shape[0] // 2
    ax3.plot(x, Z[center_y, :], 'b-', linewidth=2)
    ax3.axvline(x=puff_params['x'], color='r', linestyle='--', label='Puff Center')
    ax3.fill_between(x, 0, Z[center_y, :], alpha=0.3)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Dose Rate (mrem/hr)')
    ax3.set_title('Cross-Section (Y=0)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Vertical profile illustration
    ax4 = fig.add_subplot(224)

    # Show uniform mixing assumption
    test_positions = [0, 500, 1000, 2000]
    colors = ['red', 'orange', 'green', 'blue']

    for pos, color in zip(test_positions, colors):
        heights, conc = calc.get_cloud_profile(pos, 0)
        # Normalize for visualization
        conc_normalized = conc / (calc.activity * 1e-6) if calc.activity > 0 else conc
        ax4.plot(conc_normalized, heights, color=color, linewidth=2,
                label=f'x={pos}m')

    ax4.axhline(y=calc.h_mix, color='black', linestyle='--', linewidth=2,
               label=f'Mixing Height ({calc.h_mix:.0f}m)')
    ax4.fill_betweenx([0, calc.h_mix], 0, 1, alpha=0.1, color='gray')

    ax4.set_xlabel('Relative Concentration')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Vertical Profile (Uniform Mixing)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, calc.h_mix * 1.2])

    plt.suptitle(f'Semi-Infinite Cloud Cloudshine\n' +
                f'σ_y={puff_params["sigma_y"]:.0f}m, σ_z={puff_params["sigma_z"]:.0f}m, ' +
                f'H_mix={calc.h_mix:.0f}m, Activity={puff_params["activity"]:.2e} Ci',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cloudshine_semi_infinite.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: cloudshine_semi_infinite.png")

    return fig

def create_interactive_semi_infinite(puff_params):
    """
    Create interactive HTML visualization for semi-infinite cloud.
    """

    print("Creating interactive semi-infinite cloud visualization...")

    # Create calculator
    calc = SemiInfiniteCloudshine(
        puff_params['x'], puff_params['y'], puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity'], puff_params.get('h_mix')
    )

    # Create grid
    extent = max(5 * puff_params['sigma_y'], 3000)
    grid_size = 60
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Calculate dose field
    print(f"Calculating dose field ({grid_size}x{grid_size})...")
    Z = np.zeros_like(X)
    for i in range(grid_size):
        if i % 20 == 0:
            print(f"  Progress: {i}/{grid_size}")
        for j in range(grid_size):
            Z[i, j] = calc.calculate_dose_rate(X[i, j], Y[i, j])

    # Create figure
    fig = go.Figure()

    # Add 3D surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Plasma',
        name='Dose Field',
        colorbar=dict(
            title=dict(text='Dose Rate<br>(mrem/hr)', side='right'),
            thickness=20,
            len=0.6
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
    ))

    # Add cloud volume representation (transparent box)
    # Create box edges to show mixing height
    box_size = 3 * puff_params['sigma_y']
    box_x = [-box_size, box_size, box_size, -box_size, -box_size,
             -box_size, box_size, box_size, -box_size, -box_size,
             None,
             -box_size, -box_size, None,
             box_size, box_size, None,
             box_size, box_size, None,
             -box_size, -box_size]

    box_y = [-box_size, -box_size, box_size, box_size, -box_size,
             -box_size, -box_size, box_size, box_size, -box_size,
             None,
             -box_size, box_size, None,
             -box_size, box_size, None,
             box_size, -box_size, None,
             box_size, -box_size]

    box_z = [0, 0, 0, 0, 0,
             calc.h_mix, calc.h_mix, calc.h_mix, calc.h_mix, calc.h_mix,
             None,
             0, 0, None,
             0, 0, None,
             0, 0, None,
             0, 0]

    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Mixing Layer',
        opacity=0.5,
        hoverinfo='skip'
    ))

    # Add vertical planes to show cloud extent
    plane_x = np.array([[-box_size, box_size], [-box_size, box_size]])
    plane_y = np.array([[0, 0], [0, 0]])
    plane_z = np.array([[0, 0], [calc.h_mix, calc.h_mix]])

    fig.add_trace(go.Surface(
        x=plane_x, y=plane_y, z=plane_z,
        colorscale=[[0, 'rgba(0, 255, 255, 0.1)'], [1, 'rgba(0, 255, 255, 0.1)']],
        showscale=False,
        name='Cloud Cross-Section',
        hoverinfo='skip'
    ))

    # Add puff center marker
    fig.add_trace(go.Scatter3d(
        x=[puff_params['x']],
        y=[puff_params['y']],
        z=[calc.h_mix / 2],  # Place at mid-height
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=['Cloud Center'],
        textposition='top center',
        name='Cloud Center',
        hovertemplate='Cloud Center<br>Mixing Height: %{z:.0f}m<extra></extra>'
    ))

    # Add sigma circles on ground
    theta = np.linspace(0, 2*np.pi, 100)
    for n_sigma, color in [(1, 'orange'), (2, 'yellow'), (3, 'cyan')]:
        circle_x = puff_params['x'] + n_sigma * puff_params['sigma_y'] * np.cos(theta)
        circle_y = puff_params['y'] + n_sigma * puff_params['sigma_y'] * np.sin(theta)
        circle_z = np.zeros_like(circle_x)

        fig.add_trace(go.Scatter3d(
            x=circle_x, y=circle_y, z=circle_z,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'{n_sigma}σ_y boundary',
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Semi-Infinite Cloud Cloudshine</b><br>' +
                 f'<sub>σ_y={puff_params["sigma_y"]:.0f}m, σ_z={puff_params["sigma_z"]:.0f}m, ' +
                 f'H_mix={calc.h_mix:.0f}m, Activity={puff_params["activity"]:.2e} Ci</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Dose Rate (mrem/hr)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3),
            zaxis=dict(range=[0, max(Z.max(), calc.h_mix)])
        ),
        showlegend=True,
        height=800,
        width=1200
    )

    return fig

def compare_three_methods():
    """
    Create comparison of all three methods: 720-point, plane source, and semi-infinite.
    """

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Small Puff (720-Point)',
                       'Medium Puff (Plane Source)',
                       'Large Puff (Semi-Infinite)'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )

    # Common grid
    grid_size = 30
    extent = 1500
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Method 1: Small puff (720-point approximation)
    Z1 = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            r = np.sqrt(X[i,j]**2 + Y[i,j]**2 + 50**2)
            Z1[i,j] = 1e-6 * 1e3 / (4*PI*r**2) * np.exp(-MU_AIR*r) * 3.6e6

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z1, colorscale='Viridis', showscale=False,
                  name='720-Point', hoverinfo='skip'),
        row=1, col=1
    )

    # Method 2: Plane source (10 slabs)
    Z2 = np.zeros_like(X)
    dcf_pn = 1e-5 / 241.2
    for i in range(grid_size):
        for j in range(grid_size):
            r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
            lateral = np.exp(-0.5 * (r/500)**2) / (2*PI*500**2)
            buildup = 10 * (1 + K_BUILD*MU_AIR*100) * np.exp(-MU_AIR*100)  # Simplified
            Z2[i,j] = 0.1 * dcf_pn * lateral * buildup * 1e4 * 3.6e6

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z2, colorscale='Viridis', showscale=False,
                  name='Plane Source', hoverinfo='skip'),
        row=1, col=2
    )

    # Method 3: Semi-infinite cloud
    Z3 = np.zeros_like(X)
    h_mix = 2000
    for i in range(grid_size):
        for j in range(grid_size):
            r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
            chi_over_q = np.exp(-0.5 * (r/1000)**2) / (2*PI*1000**2*h_mix)
            Z3[i,j] = 1e5 * chi_over_q * 1e-5 * 3.6e6

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z3, colorscale='Viridis',
                  colorbar=dict(title='Dose Rate<br>(mrem/hr)', x=1.02),
                  name='Semi-Infinite', hoverinfo='skip'),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Cloudshine Calculation Methods Comparison</b><br>' +
                 '<sub>Transition from 720-point → Plane Source → Semi-Infinite Cloud</sub>',
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
        scene3=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Dose (mrem/hr)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600, width=1600,
        showlegend=False
    )

    return fig

def main():
    """
    Main function to run semi-infinite cloud visualization.
    """

    print("="*60)
    print("Semi-Infinite Cloud Cloudshine Visualization")
    print("="*60)

    # Test scenarios for semi-infinite cloud conditions
    scenarios = [
        {
            'name': 'Large Vertical Dispersion',
            'x': 0, 'y': 0, 'z': 200,
            'sigma_y': 1000,
            'sigma_z': 500,  # > 400m threshold
            'activity': 1e5,
            'h_mix': 1500
        },
        {
            'name': 'Very Large Cloud',
            'x': 0, 'y': 0, 'z': 300,
            'sigma_y': 1500,
            'sigma_z': 800,
            'activity': 5e5,
            'h_mix': 2000
        },
        {
            'name': 'Extreme Dispersion',
            'x': 0, 'y': 0, 'z': 500,
            'sigma_y': 2000,
            'sigma_z': 1000,
            'activity': 1e6,
            'h_mix': None  # Will use 2*sigma_z
        }
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")

        # Create static visualization
        visualize_semi_infinite_3d(scenario)

        # Create interactive visualization
        fig_interactive = create_interactive_semi_infinite(scenario)
        filename = f"cloudshine_semi_infinite_{scenario['name'].lower().replace(' ', '_')}.html"
        fig_interactive.write_html(filename)
        print(f"Saved interactive visualization to: {filename}")

        # Calculate and print statistics
        calc = SemiInfiniteCloudshine(
            scenario['x'], scenario['y'], scenario['z'],
            scenario['sigma_y'], scenario['sigma_z'],
            scenario['activity'], scenario['h_mix']
        )

        print(f"\nParameters:")
        print(f"  σ_y: {scenario['sigma_y']}m")
        print(f"  σ_z: {scenario['sigma_z']}m")
        print(f"  Mixing height: {calc.h_mix:.0f}m")
        print(f"  Activity: {scenario['activity']:.2e} Ci")

        print(f"\nDose rates at specific distances:")
        distances = [100, 500, 1000, 2000, 3000, 5000]
        for d in distances:
            dose = calc.calculate_dose_rate(d, 0)
            print(f"  {d:4d}m: {dose:.3e} mrem/hr")

        # Print χ/Q values
        print(f"\nχ/Q values (atmospheric dispersion):")
        for d in [100, 500, 1000, 2000]:
            chi_q = calc.calculate_chi_over_q(d, 0)
            print(f"  {d:4d}m: {chi_q:.3e} s/m³")

    # Create method comparison
    print(f"\n{'='*60}")
    print("Creating three-method comparison...")
    fig_compare = compare_three_methods()
    fig_compare.write_html("cloudshine_three_methods_comparison.html")
    print("Saved comparison to: cloudshine_three_methods_comparison.html")

    print("\n" + "="*60)
    print("Semi-Infinite Cloud Visualization Complete!")
    print("="*60)

if __name__ == "__main__":
    main()