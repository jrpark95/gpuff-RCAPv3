"""
GPUFF-RCAPv3 Cloudshine 3D Visualization
========================================
Visualizes cloudshine dose values on the ground surface near a stationary unit puff
using the 720-division summation method from the CUDA implementation.

The visualization shows:
- 3D surface plot of dose rates on the ground
- Color-coded dose intensity
- Contour lines for dose levels
- Puff position and extent
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import erf
from dataclasses import dataclass
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Constants from CUDA code
MU_AIR = 0.01  # Air attenuation coefficient [m^-1] for ~0.7 MeV
K_BUILD = 1.4  # Buildup correction factor
PI = np.pi

@dataclass
class PuffParameters:
    """Parameters for a radioactive puff"""
    x: float  # X position [m]
    y: float  # Y position [m]
    z: float  # Z height [m]
    sigma_y: float  # Horizontal dispersion parameter [m]
    sigma_z: float  # Vertical dispersion parameter [m]
    activity: float  # Total activity [Ci]

@dataclass
class DoseParameters:
    """Dose calculation parameters"""
    dcf_point: float = 1e-6  # Point kernel dose conversion factor [(rem/s)/(Ci/m)]
    dcf_sic: float = 1e-5    # Semi-infinite cloud DCF [(rem/s)/(Ci/m^3)]

class CloudshineCalculator:
    """
    Calculates cloudshine dose using the 720-point summation method
    from the GPUFF-RCAPv3 CUDA implementation
    """

    def __init__(self, puff: PuffParameters, dose_params: DoseParameters):
        self.puff = puff
        self.dose_params = dose_params
        self.puff_720_points = self._generate_720_points()

    def _generate_720_points(self) -> np.ndarray:
        """
        Generate 720 representative points distributed within the puff volume.
        Uses a structured grid approach similar to the CUDA implementation.

        Returns:
            Array of shape (720, 3) with normalized (x, y, z) coordinates
        """
        # Create a 3D grid of points within the puff
        # Use 10x12x6 = 720 points distributed in a normalized volume
        nx, ny, nz = 10, 12, 6

        # Generate normalized coordinates in [-2, 2] range (2-sigma coverage)
        x_norm = np.linspace(-2, 2, nx)
        y_norm = np.linspace(-2, 2, ny)
        z_norm = np.linspace(-2, 2, nz)

        # Create meshgrid
        X, Y, Z = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')

        # Flatten and combine
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Apply Gaussian weighting to points based on distance from center
        weights = np.exp(-0.5 * np.sum(points**2, axis=1))

        # Normalize weights
        weights /= weights.sum()

        # Store weights as 4th column for later use
        points_weighted = np.column_stack([points, weights])

        return points_weighted

    def point_kernel_dose(self, receptor: np.ndarray, source: np.ndarray) -> float:
        """
        Calculate point kernel dose from a source point to a receptor.

        Args:
            receptor: Receptor position [x, y, z] in meters
            source: Source position [x, y, z] in meters

        Returns:
            Dose rate contribution [rem/s]
        """
        # Calculate distance
        dx = receptor[0] - source[0]
        dy = receptor[1] - source[1]
        dz = receptor[2] - source[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6  # Avoid division by zero

        # Point kernel with buildup and attenuation
        buildup = 1.0 + K_BUILD * MU_AIR * distance
        attenuation = np.exp(-MU_AIR * distance)

        # Geometric factor (1/4πr²)
        geometric_factor = 1.0 / (4.0 * PI * distance**2)

        # Total dose rate
        dose_rate = self.dose_params.dcf_point * geometric_factor * buildup * attenuation

        return dose_rate

    def calculate_cloudshine_720(self, receptor: np.ndarray) -> float:
        """
        Calculate cloudshine dose using 720-point summation method.

        Args:
            receptor: Receptor position [x, y, z] in meters

        Returns:
            Total cloudshine dose rate [rem/s]
        """
        total_dose = 0.0

        # Small puff mode (sigma < 400m)
        if self.puff.sigma_y < 400 and self.puff.sigma_z < 400:
            # Sum contributions from all 720 points
            for i in range(720):
                # Scale normalized coordinates to actual puff dimensions
                point_norm = self.puff_720_points[i, :3]
                weight = self.puff_720_points[i, 3]

                source = np.array([
                    self.puff.x + point_norm[0] * self.puff.sigma_y,
                    self.puff.y + point_norm[1] * self.puff.sigma_y,
                    self.puff.z + point_norm[2] * self.puff.sigma_z
                ])

                # Calculate dose from this point
                dose_contrib = self.point_kernel_dose(receptor, source)

                # Apply weight and activity
                total_dose += dose_contrib * weight * self.puff.activity

        # Plane source mode (400m <= sigma_y < 400m, sigma_z < 400m)
        elif self.puff.sigma_y >= 400 and self.puff.sigma_z < 400:
            total_dose = self._plane_source_dose(receptor)

        # Semi-infinite cloud mode (sigma_z >= 400m)
        else:
            total_dose = self._semi_infinite_dose(receptor)

        return total_dose

    def _plane_source_dose(self, receptor: np.ndarray) -> float:
        """Calculate dose using plane source approximation with 10 slabs."""
        # Ground projection distance
        dx = receptor[0] - self.puff.x
        dy = receptor[1] - self.puff.y
        r = np.sqrt(dx**2 + dy**2)

        # Lateral Gaussian factor
        lateral = np.exp(-0.5 * (r / self.puff.sigma_y)**2) / (2 * PI * self.puff.sigma_y**2)

        # 10 slab heights using 5 quantiles
        c = np.array([0.127, 0.385, 0.674, 1.037, 1.645])
        z_slabs = []
        for ci in c:
            z_slabs.append(self.puff.z + ci * self.puff.sigma_z)
            z_slabs.append(self.puff.z - ci * self.puff.sigma_z)
        z_slabs = np.array(z_slabs)

        # Ground reflection
        z_slabs = np.abs(z_slabs)

        # Buildup sum for all slabs
        sum_buildup = 0.0
        for z in z_slabs:
            sum_buildup += (1.0 + K_BUILD * MU_AIR * z) * np.exp(-MU_AIR * z)

        # DCF_pn approximation
        dcf_pn = self.dose_params.dcf_sic / 241.2

        # Total dose (1/10 activity per slab)
        dose = 0.1 * dcf_pn * lateral * sum_buildup * self.puff.activity

        return dose

    def _semi_infinite_dose(self, receptor: np.ndarray) -> float:
        """Calculate dose using semi-infinite cloud approximation."""
        # Ground projection distance
        dx = receptor[0] - self.puff.x
        dy = receptor[1] - self.puff.y
        r = np.sqrt(dx**2 + dy**2)

        # Mixing height (assume 1000m if not specified)
        h_mix = max(1000.0, 2 * self.puff.sigma_z)

        # Chi/Q for uniform mixing
        chi_over_q = np.exp(-0.5 * (r / self.puff.sigma_y)**2) / \
                     (2 * PI * self.puff.sigma_y**2 * h_mix)

        # Total dose
        dose = self.puff.activity * chi_over_q * self.dose_params.dcf_sic

        return dose

def create_ground_grid(puff: PuffParameters, grid_size: int = 100,
                       extent_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of ground-level receptor points.

    Args:
        puff: Puff parameters
        grid_size: Number of grid points in each dimension
        extent_factor: Grid extent as multiple of sigma_y

    Returns:
        X, Y meshgrid arrays
    """
    # Define grid extent
    extent = max(extent_factor * puff.sigma_y, 1000.0)  # At least 1km

    # Create grid
    x = np.linspace(puff.x - extent, puff.x + extent, grid_size)
    y = np.linspace(puff.y - extent, puff.y + extent, grid_size)

    X, Y = np.meshgrid(x, y)

    return X, Y

def calculate_dose_field(puff: PuffParameters, dose_params: DoseParameters,
                        X: np.ndarray, Y: np.ndarray, z_receptor: float = 1.0) -> np.ndarray:
    """
    Calculate cloudshine dose field on a grid.

    Args:
        puff: Puff parameters
        dose_params: Dose calculation parameters
        X, Y: Meshgrid arrays for receptor positions
        z_receptor: Receptor height above ground [m]

    Returns:
        Dose field array matching X, Y shape
    """
    calculator = CloudshineCalculator(puff, dose_params)

    # Initialize dose field
    dose_field = np.zeros_like(X)

    # Calculate dose at each grid point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            receptor = np.array([X[i, j], Y[i, j], z_receptor])
            dose_field[i, j] = calculator.calculate_cloudshine_720(receptor)

    # Convert to mrem/hr for better units
    dose_field *= 3.6e6  # (rem/s) to (mrem/hr)

    return dose_field

def visualize_cloudshine_3d(puff: PuffParameters, dose_params: DoseParameters):
    """
    Create interactive 3D visualization of cloudshine dose field.
    """
    # Create ground grid
    X, Y = create_ground_grid(puff, grid_size=50, extent_factor=5.0)

    # Calculate dose field
    print("Calculating cloudshine dose field using 720-point method...")
    Z = calculate_dose_field(puff, dose_params, X, Y)

    # Create interactive Plotly visualization
    fig = go.Figure()

    # Add 3D surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        name='Cloudshine Dose',
        colorbar=dict(
            title=dict(
                text='Dose Rate<br>(mrem/hr)',
                side='right'
            ),
            tickmode='linear',
            tick0=0,
            dtick=Z.max()/10
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    ))

    # Add puff position marker
    fig.add_trace(go.Scatter3d(
        x=[puff.x],
        y=[puff.y],
        z=[Z.max()],
        mode='markers+text',
        name='Puff Center',
        marker=dict(size=10, color='red'),
        text=['Puff Center'],
        textposition='top center'
    ))

    # Add puff extent indicators (sigma boundaries)
    theta = np.linspace(0, 2*np.pi, 100)

    # 1-sigma circle
    circle_1sigma_x = puff.x + puff.sigma_y * np.cos(theta)
    circle_1sigma_y = puff.y + puff.sigma_y * np.sin(theta)
    circle_1sigma_z = np.zeros_like(theta)

    fig.add_trace(go.Scatter3d(
        x=circle_1sigma_x,
        y=circle_1sigma_y,
        z=circle_1sigma_z,
        mode='lines',
        name='1σ boundary',
        line=dict(color='orange', width=3),
        showlegend=True
    ))

    # 2-sigma circle
    circle_2sigma_x = puff.x + 2*puff.sigma_y * np.cos(theta)
    circle_2sigma_y = puff.y + 2*puff.sigma_y * np.sin(theta)

    fig.add_trace(go.Scatter3d(
        x=circle_2sigma_x,
        y=circle_2sigma_y,
        z=circle_1sigma_z,
        mode='lines',
        name='2σ boundary',
        line=dict(color='yellow', width=2),
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Cloudshine Dose Field (720-Point Method)<br>' +
                 f'Puff: σ_y={puff.sigma_y:.1f}m, σ_z={puff.sigma_z:.1f}m, ' +
                 f'Height={puff.z:.1f}m, Activity={puff.activity:.2e} Ci',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Dose Rate (mrem/hr)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        width=1200,
        height=800,
        showlegend=True
    )

    # Save interactive HTML
    fig.write_html("cloudshine_3d_visualization.html")
    print("Saved interactive visualization to cloudshine_3d_visualization.html")

    # Also create matplotlib static visualization
    create_matplotlib_visualization(X, Y, Z, puff)

def create_matplotlib_visualization(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                   puff: PuffParameters):
    """Create static matplotlib visualization with multiple views."""

    fig = plt.figure(figsize=(16, 12))

    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Dose Rate (mrem/hr)')
    ax1.set_title('3D Cloudshine Dose Surface')
    ax1.view_init(elev=30, azim=45)

    # Add puff position
    ax1.scatter([puff.x], [puff.y], [Z.max()], color='red', s=100, marker='*')

    # Top-down view with contours
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax2.plot(puff.x, puff.y, 'r*', markersize=15, label='Puff Center')

    # Add sigma circles
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(puff.x + puff.sigma_y * np.cos(theta),
             puff.y + puff.sigma_y * np.sin(theta),
             'r--', label='1σ boundary')
    ax2.plot(puff.x + 2*puff.sigma_y * np.cos(theta),
             puff.y + 2*puff.sigma_y * np.sin(theta),
             'r:', label='2σ boundary')

    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top-Down View with Contours')
    ax2.legend()
    ax2.axis('equal')
    plt.colorbar(contour, ax=ax2, label='Dose Rate (mrem/hr)')

    # Cross-section through puff center (X direction)
    ax3 = fig.add_subplot(223)
    center_idx_y = X.shape[0] // 2
    ax3.plot(X[center_idx_y, :], Z[center_idx_y, :], 'b-', linewidth=2)
    ax3.axvline(x=puff.x, color='r', linestyle='--', label='Puff Center')
    ax3.fill_between(X[center_idx_y, :], 0, Z[center_idx_y, :], alpha=0.3)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Dose Rate (mrem/hr)')
    ax3.set_title('Cross-Section (X-axis through puff center)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Cross-section through puff center (Y direction)
    ax4 = fig.add_subplot(224)
    center_idx_x = Y.shape[1] // 2
    ax4.plot(Y[:, center_idx_x], Z[:, center_idx_x], 'g-', linewidth=2)
    ax4.axvline(x=puff.y, color='r', linestyle='--', label='Puff Center')
    ax4.fill_between(Y[:, center_idx_x], 0, Z[:, center_idx_x], alpha=0.3)
    ax4.set_xlabel('Y Position (m)')
    ax4.set_ylabel('Dose Rate (mrem/hr)')
    ax4.set_title('Cross-Section (Y-axis through puff center)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add main title
    fig.suptitle(f'Cloudshine Dose Field Analysis (720-Point Method)\n' +
                 f'Puff Parameters: σ_y={puff.sigma_y:.1f}m, σ_z={puff.sigma_z:.1f}m, ' +
                 f'Height={puff.z:.1f}m, Activity={puff.activity:.2e} Ci',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cloudshine_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved static analysis to cloudshine_analysis.png")
    plt.close()  # Close without showing to prevent hanging

def main():
    """Main function to run cloudshine visualization."""

    # Define test scenarios
    scenarios = [
        # Small puff (uses 720-point method)
        {
            'name': 'Small Puff',
            'puff': PuffParameters(
                x=0, y=0, z=50,      # 50m height
                sigma_y=100,         # 100m horizontal dispersion
                sigma_z=50,          # 50m vertical dispersion
                activity=1e3         # 1000 Ci
            )
        },
        # Medium puff (plane source approximation)
        {
            'name': 'Medium Puff',
            'puff': PuffParameters(
                x=0, y=0, z=100,     # 100m height
                sigma_y=500,         # 500m horizontal dispersion
                sigma_z=200,         # 200m vertical dispersion
                activity=1e4         # 10000 Ci
            )
        },
        # Large puff (semi-infinite cloud)
        {
            'name': 'Large Puff',
            'puff': PuffParameters(
                x=0, y=0, z=200,     # 200m height
                sigma_y=1000,        # 1000m horizontal dispersion
                sigma_z=500,         # 500m vertical dispersion
                activity=1e5         # 100000 Ci
            )
        }
    ]

    # Dose parameters (example values)
    dose_params = DoseParameters(
        dcf_point=1e-6,  # Point kernel DCF
        dcf_sic=1e-5     # Semi-infinite cloud DCF
    )

    # Visualize each scenario
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Visualizing: {scenario['name']}")
        print(f"{'='*60}")

        visualize_cloudshine_3d(scenario['puff'], dose_params)

        # Print some statistics
        X, Y = create_ground_grid(scenario['puff'], grid_size=50)
        Z = calculate_dose_field(scenario['puff'], dose_params, X, Y)

        print(f"\nDose Field Statistics:")
        print(f"  Max dose rate: {Z.max():.2e} mrem/hr")
        print(f"  Mean dose rate: {Z.mean():.2e} mrem/hr")
        print(f"  Min dose rate: {Z.min():.2e} mrem/hr")

        # Find dose at specific distances
        distances = [100, 500, 1000, 2000]  # meters
        print(f"\nDose rates at specific distances from puff center:")
        for d in distances:
            receptor = np.array([d, 0, 1.0])  # Along positive X-axis
            calculator = CloudshineCalculator(scenario['puff'], dose_params)
            dose = calculator.calculate_cloudshine_720(receptor) * 3.6e6  # Convert to mrem/hr
            print(f"  {d:4d}m: {dose:.2e} mrem/hr")

if __name__ == "__main__":
    main()