"""
Cloudshine Visualization - Plane Source Method (10 Slabs)
=========================================================
Visualizes cloudshine dose using the plane source approximation
with 10 vertical slabs as implemented in GPUFF-RCAPv3.

Based on RASCAL 4 NUREG-1940 methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Constants from CUDA code
MU_AIR = 0.01    # Air attenuation coefficient [m^-1] at ~0.7 MeV
K_BUILD = 1.4    # Buildup correction factor
PI = np.pi

class PlaneSourceCloudshine:
    """
    Calculate cloudshine dose using plane source method with 10 slabs.
    This matches the CUDA implementation in gpuff_kernels_cloudshine.cuh
    """

    def __init__(self, puff_x, puff_y, puff_z, sigma_y, sigma_z, activity, h_mix=1000.0):
        """
        Initialize plane source calculator.

        Args:
            puff_x, puff_y, puff_z: Puff center coordinates (m)
            sigma_y, sigma_z: Dispersion parameters (m)
            activity: Total activity (Ci)
            h_mix: Mixing layer height (m)
        """
        self.puff_x = puff_x
        self.puff_y = puff_y
        self.puff_z = puff_z  # Effective release height
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.activity = activity
        self.h_mix = h_mix

        # Dose conversion factors
        self.dcf_sic = 1e-5  # Semi-infinite cloud DCF [(rem/s)/(Ci/m^3)]
        self.dcf_pn = self.dcf_sic / 241.2  # Plane source DCF

        # Generate 10 slab heights
        self.slab_heights = self._generate_slab_heights()

    def _generate_slab_heights(self):
        """
        Generate 10 slab heights using 5 quantile coefficients.
        Matches the plane_slab_heights function in CUDA code.
        """
        # 5 quantile coefficients from CUDA code
        c = np.array([0.127, 0.385, 0.674, 1.037, 1.645])

        z_slabs = []
        for ci in c:
            # Add both + and - offsets for each coefficient
            z_slabs.append(self.puff_z + ci * self.sigma_z)
            z_slabs.append(self.puff_z - ci * self.sigma_z)

        z_slabs = np.array(z_slabs)

        # Ground reflection (mirror negative heights)
        z_slabs = np.abs(z_slabs)

        # Mixing layer reflection
        if self.h_mix > 0:
            z_slabs = np.where(z_slabs > self.h_mix,
                             2.0 * self.h_mix - z_slabs,
                             z_slabs)

        return z_slabs

    def calculate_dose_rate(self, receptor_x, receptor_y, receptor_z=1.0):
        """
        Calculate cloudshine dose rate at receptor position using plane source method.

        Args:
            receptor_x, receptor_y, receptor_z: Receptor coordinates (m)

        Returns:
            Dose rate (rem/s)
        """
        # Calculate horizontal distance from puff center
        dx = receptor_x - self.puff_x
        dy = receptor_y - self.puff_y
        r = np.sqrt(dx**2 + dy**2)

        # Lateral Gaussian dispersion factor
        lateral = np.exp(-0.5 * (r / self.sigma_y)**2) / (2.0 * PI * self.sigma_y**2)

        # Calculate buildup sum for all 10 slabs
        sum_buildup = 0.0
        slab_contributions = []

        for i, z_s in enumerate(self.slab_heights):
            # Vertical distance from receptor to slab
            dz = abs(z_s - receptor_z)

            # Buildup factor with air attenuation
            buildup_factor = (1.0 + K_BUILD * MU_AIR * dz) * np.exp(-MU_AIR * dz)
            sum_buildup += buildup_factor

            # Store individual slab contribution for visualization
            slab_contribution = 0.1 * self.dcf_pn * lateral * buildup_factor * self.activity
            slab_contributions.append({
                'height': z_s,
                'buildup': buildup_factor,
                'dose_rate': slab_contribution
            })

        # Total dose rate (each slab has 1/10 of total activity)
        dose_rate = 0.1 * self.dcf_pn * lateral * sum_buildup * self.activity

        return dose_rate, slab_contributions

    def visualize_slab_structure(self):
        """
        Visualize the 10-slab structure in 3D space.
        """
        fig = plt.figure(figsize=(14, 10))

        # 3D visualization of slabs
        ax1 = fig.add_subplot(121, projection='3d')

        # Create grid for each slab
        x_range = 3 * self.sigma_y
        grid_points = 30
        x = np.linspace(-x_range, x_range, grid_points)
        y = np.linspace(-x_range, x_range, grid_points)
        X, Y = np.meshgrid(x, y)

        # Plot each slab as a transparent surface
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))

        for i, z_s in enumerate(self.slab_heights):
            Z = np.ones_like(X) * z_s

            # Calculate activity density at this slab height
            density = np.exp(-0.5 * ((X/self.sigma_y)**2 + (Y/self.sigma_y)**2))

            ax1.plot_surface(X, Y, Z, alpha=0.3, color=colors[i],
                           label=f'Slab {i+1}: z={z_s:.1f}m')

        # Mark puff center
        ax1.scatter([0], [0], [self.puff_z], color='red', s=100,
                   marker='*', label='Puff Center')

        # Add ground plane
        ground = np.zeros_like(X)
        ax1.plot_surface(X, Y, ground, alpha=0.1, color='gray')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Height (m)')
        ax1.set_title('10-Slab Plane Source Structure')
        ax1.view_init(elev=20, azim=45)

        # Side view showing slab heights
        ax2 = fig.add_subplot(122)

        # Plot Gaussian profile
        z_profile = np.linspace(0, max(self.slab_heights) + 50, 200)
        gaussian = np.exp(-0.5 * ((z_profile - self.puff_z) / self.sigma_z)**2)
        ax2.plot(gaussian, z_profile, 'b-', linewidth=2, label='Gaussian Profile')

        # Mark slab positions
        for i, z_s in enumerate(self.slab_heights):
            ax2.axhline(y=z_s, color=colors[i], linestyle='--', alpha=0.7)
            ax2.text(1.05, z_s, f'Slab {i+1}', fontsize=8, va='center')

        ax2.axhline(y=self.puff_z, color='red', linestyle='-',
                   linewidth=2, label='Puff Center')
        ax2.fill_betweenx(z_profile, 0, gaussian, alpha=0.2)

        ax2.set_xlabel('Normalized Concentration')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Vertical Distribution and Slab Positions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Plane Source Method: 10-Slab Structure\n' +
                    f'σ_y={self.sigma_y:.0f}m, σ_z={self.sigma_z:.0f}m, ' +
                    f'Release Height={self.puff_z:.0f}m',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

def calculate_dose_field_plane_source(puff_params, grid_size=50, extent_factor=5.0):
    """
    Calculate cloudshine dose field using plane source method.
    """
    # Create calculator
    calc = PlaneSourceCloudshine(
        puff_params['x'], puff_params['y'], puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity'], puff_params.get('h_mix', 1000.0)
    )

    # Create ground grid
    extent = max(extent_factor * puff_params['sigma_y'], 1500.0)
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Calculate dose field
    Z = np.zeros_like(X)
    slab_data = []

    for i in range(grid_size):
        if i % 10 == 0:
            print(f"  Progress: {i}/{grid_size} rows completed")
        for j in range(grid_size):
            dose_rate, slabs = calc.calculate_dose_rate(X[i,j], Y[i,j], 1.0)
            Z[i,j] = dose_rate * 3.6e6  # Convert to mrem/hr

            # Store slab data for center point
            if i == grid_size//2 and j == grid_size//2:
                slab_data = slabs

    return X, Y, Z, calc, slab_data

def visualize_plane_source_3d(puff_params):
    """
    Create comprehensive visualization of plane source cloudshine.
    """
    print(f"\nCalculating plane source cloudshine field...")
    print(f"Puff parameters: σ_y={puff_params['sigma_y']}m, σ_z={puff_params['sigma_z']}m")

    # Calculate dose field
    X, Y, Z, calc, slab_data = calculate_dose_field_plane_source(puff_params)

    # Create main visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                           edgecolor='none', antialiased=True)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Dose Rate (mrem/hr)')
    ax1.set_title('3D Cloudshine Dose Surface (Plane Source)')
    ax1.view_init(elev=25, azim=45)

    # Mark puff center
    ax1.scatter([puff_params['x']], [puff_params['y']], [Z.max()],
               color='red', s=100, marker='*')

    # Top-down contour view
    ax2 = fig.add_subplot(222)
    levels = 20
    contour = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    # Mark puff and dispersion boundaries
    ax2.plot(puff_params['x'], puff_params['y'], 'r*', markersize=15, label='Puff Center')
    theta = np.linspace(0, 2*np.pi, 100)

    for n_sigma, style in [(1, '--'), (2, ':'), (3, '-.')]:
        circle_x = puff_params['x'] + n_sigma * puff_params['sigma_y'] * np.cos(theta)
        circle_y = puff_params['y'] + n_sigma * puff_params['sigma_y'] * np.sin(theta)
        ax2.plot(circle_x, circle_y, 'r' + style,
                label=f'{n_sigma}σ boundary', linewidth=2-n_sigma*0.3)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View with Contours')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2, label='Dose Rate (mrem/hr)')

    # Cross-section along X
    ax3 = fig.add_subplot(223)
    center_y = X.shape[0] // 2
    x_values = X[center_y, :]
    z_values = Z[center_y, :]

    ax3.plot(x_values, z_values, 'b-', linewidth=2, label='Dose Rate')
    ax3.axvline(x=puff_params['x'], color='r', linestyle='--', label='Puff Center')
    ax3.fill_between(x_values, 0, z_values, alpha=0.3, color='blue')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Dose Rate (mrem/hr)')
    ax3.set_title('Cross-Section (Y=0) - Plane Source')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Slab contribution analysis
    ax4 = fig.add_subplot(224)

    if slab_data:
        slab_heights = [s['height'] for s in slab_data]
        slab_doses = [s['dose_rate'] * 3.6e6 for s in slab_data]  # to mrem/hr
        slab_buildup = [s['buildup'] for s in slab_data]

        # Bar plot of slab contributions
        bars = ax4.bar(range(len(slab_heights)), slab_doses,
                      color=plt.cm.plasma(np.linspace(0.3, 0.9, len(slab_heights))))

        # Add slab height labels
        for i, (h, d) in enumerate(zip(slab_heights, slab_doses)):
            ax4.text(i, d, f'{h:.0f}m', ha='center', va='bottom', fontsize=8)

        ax4.set_xlabel('Slab Number')
        ax4.set_ylabel('Dose Contribution (mrem/hr)')
        ax4.set_title('Individual Slab Contributions at Center')
        ax4.grid(True, alpha=0.3, axis='y')

    # Main title
    fig.suptitle(f'Plane Source Cloudshine (10-Slab Method)\n' +
                f'σ_y={puff_params["sigma_y"]:.0f}m, σ_z={puff_params["sigma_z"]:.0f}m, ' +
                f'Height={puff_params["z"]:.0f}m, Activity={puff_params["activity"]:.2e} Ci',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cloudshine_plane_source_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: cloudshine_plane_source_visualization.png")

    return fig

def compare_methods(puff_params):
    """
    Compare plane source method with 720-point method.
    """
    fig = plt.figure(figsize=(14, 8))

    # Calculate dose along radial distance for both methods
    distances = np.logspace(1, 3.5, 100)  # 10m to ~3000m
    dose_plane = []

    calc_plane = PlaneSourceCloudshine(
        0, 0, puff_params['z'],
        puff_params['sigma_y'], puff_params['sigma_z'],
        puff_params['activity']
    )

    for d in distances:
        dose_rate, _ = calc_plane.calculate_dose_rate(d, 0, 1.0)
        dose_plane.append(dose_rate * 3.6e6)  # to mrem/hr

    # Plot comparison
    ax1 = fig.add_subplot(121)
    ax1.loglog(distances, dose_plane, 'b-', linewidth=2, label='Plane Source (10 slabs)')

    # Mark transition boundaries
    if puff_params['sigma_y'] >= 400:
        ax1.axvline(x=400, color='orange', linestyle='--', alpha=0.5, label='σ_y = 400m boundary')

    ax1.set_xlabel('Distance from Puff Center (m)')
    ax1.set_ylabel('Dose Rate (mrem/hr)')
    ax1.set_title('Dose vs Distance (Log-Log Scale)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    # Vertical profile at fixed horizontal distance
    ax2 = fig.add_subplot(122)

    test_distance = puff_params['sigma_y']  # At 1 sigma distance
    heights = np.linspace(0.1, 200, 100)
    dose_vertical = []

    for h in heights:
        dose_rate, _ = calc_plane.calculate_dose_rate(test_distance, 0, h)
        dose_vertical.append(dose_rate * 3.6e6)

    ax2.semilogy(heights, dose_vertical, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Standard receptor height')

    # Mark slab positions
    for z_s in calc_plane.slab_heights:
        ax2.axvline(x=z_s, color='gray', linestyle=':', alpha=0.3)

    ax2.set_xlabel('Receptor Height (m)')
    ax2.set_ylabel('Dose Rate (mrem/hr)')
    ax2.set_title(f'Vertical Profile at r={test_distance:.0f}m')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('Plane Source Method Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cloudshine_plane_source_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved analysis to: cloudshine_plane_source_analysis.png")

def main():
    """
    Main function to run plane source cloudshine visualization.
    """
    print("="*60)
    print("Cloudshine Visualization - Plane Source Method")
    print("="*60)

    # Define test scenarios for plane source conditions
    scenarios = [
        {
            'name': 'Medium Puff (Plane Source Regime)',
            'x': 0, 'y': 0, 'z': 100,      # 100m height
            'sigma_y': 500,                 # 500m > 400m threshold
            'sigma_z': 200,                 # 200m < 400m threshold
            'activity': 1e4,                # 10,000 Ci
            'h_mix': 1000.0                 # Mixing height
        },
        {
            'name': 'Large Horizontal Dispersion',
            'x': 0, 'y': 0, 'z': 150,
            'sigma_y': 800,                 # 800m >> 400m
            'sigma_z': 300,                 # 300m < 400m
            'activity': 5e4,                # 50,000 Ci
            'h_mix': 1500.0
        },
        {
            'name': 'Near Transition Boundary',
            'x': 0, 'y': 0, 'z': 80,
            'sigma_y': 450,                 # Just above 400m threshold
            'sigma_z': 380,                 # Just below 400m threshold
            'activity': 2e3,
            'h_mix': 800.0
        }
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")

        # Visualize 3D dose field
        visualize_plane_source_3d(scenario)

        # Visualize slab structure
        calc = PlaneSourceCloudshine(
            scenario['x'], scenario['y'], scenario['z'],
            scenario['sigma_y'], scenario['sigma_z'],
            scenario['activity'], scenario['h_mix']
        )
        slab_fig = calc.visualize_slab_structure()
        plt.savefig(f'cloudshine_slabs_{scenario["name"].replace(" ", "_")}.png',
                   dpi=150, bbox_inches='tight')
        plt.close(slab_fig)

        # Print statistics
        print(f"\nSlab Heights:")
        for i, z in enumerate(calc.slab_heights):
            print(f"  Slab {i+1:2d}: {z:6.1f} m")

        # Calculate dose at specific distances
        print(f"\nDose rates at specific distances:")
        test_distances = [100, 200, 500, 1000, 1500, 2000]
        for d in test_distances:
            dose_rate, _ = calc.calculate_dose_rate(d, 0, 1.0)
            dose_mrem = dose_rate * 3.6e6
            print(f"  {d:4d}m: {dose_mrem:.3e} mrem/hr")

    # Compare methods for the first scenario
    print(f"\n{'='*60}")
    print("Comparing dose profiles...")
    compare_methods(scenarios[0])

    print("\n" + "="*60)
    print("Plane Source Visualization Complete!")
    print("="*60)

if __name__ == "__main__":
    main()