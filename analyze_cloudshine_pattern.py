"""
Analyze non-monotonic pattern in cloudshine dose distribution
=============================================================
Investigates why the dose doesn't decrease monotonically near the puff center.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Constants
MU_AIR = 0.01  # Air attenuation coefficient [m^-1]
K_BUILD = 1.4  # Buildup correction factor
PI = np.pi

def analyze_dose_contributions(puff_z, sigma_y, sigma_z, receptor_x, activity):
    """
    Analyze dose contributions at different receptor positions.
    """
    # Fixed parameters
    puff_x, puff_y = 0, 0
    receptor_y, receptor_z = 0, 1.0
    dcf_point = 1e-6

    # Generate 720 points
    nx, ny, nz = 10, 12, 6
    x_norm = np.linspace(-2, 2, nx)
    y_norm = np.linspace(-2, 2, ny)
    z_norm = np.linspace(-2, 2, nz)

    # Analyze contributions by height layers
    dose_by_layer = {}
    total_dose = 0.0

    for iz, zi in enumerate(z_norm):
        layer_dose = 0.0
        for xi in x_norm:
            for yi in y_norm:
                # Source position
                source_x = puff_x + xi * sigma_y
                source_y = puff_y + yi * sigma_y
                source_z = puff_z + zi * sigma_z

                # Calculate distance components
                dx = receptor_x - source_x
                dy = receptor_y - source_y
                dz = receptor_z - source_z

                # 3D distance
                distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

                # Point kernel calculation
                buildup = 1.0 + K_BUILD * MU_AIR * distance
                attenuation = np.exp(-MU_AIR * distance)
                geometric = 1.0 / (4.0 * PI * distance**2)

                # Gaussian weight
                weight = np.exp(-0.5 * (xi**2 + yi**2 + zi**2))

                # Dose contribution
                dose_contrib = dcf_point * geometric * buildup * attenuation * weight
                layer_dose += dose_contrib
                total_dose += dose_contrib

        dose_by_layer[f'z_norm={zi:.1f}'] = layer_dose

    # Normalize
    total_dose *= activity / 720.0
    for key in dose_by_layer:
        dose_by_layer[key] *= activity / 720.0

    return total_dose * 3.6e6, dose_by_layer  # Convert to mrem/hr

def calculate_geometric_effects(puff_z, sigma_y, sigma_z):
    """
    Analyze geometric effects causing non-monotonic behavior.
    """
    receptor_positions = np.linspace(-200, 200, 401)
    doses = []
    vertical_angles = []
    horizontal_distances = []

    for rx in receptor_positions:
        # Calculate dose
        puff_x, puff_y = 0, 0
        receptor_y, receptor_z = 0, 1.0
        activity = 1e3
        dcf_point = 1e-6

        total_dose = 0.0

        # 720 points
        nx, ny, nz = 10, 12, 6
        x_norm = np.linspace(-2, 2, nx)
        y_norm = np.linspace(-2, 2, ny)
        z_norm = np.linspace(-2, 2, nz)

        for xi in x_norm:
            for yi in y_norm:
                for zi in z_norm:
                    source_x = puff_x + xi * sigma_y
                    source_y = puff_y + yi * sigma_y
                    source_z = puff_z + zi * sigma_z

                    dx = rx - source_x
                    dy = receptor_y - source_y
                    dz = receptor_z - source_z

                    distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

                    buildup = 1.0 + K_BUILD * MU_AIR * distance
                    attenuation = np.exp(-MU_AIR * distance)
                    geometric = 1.0 / (4.0 * PI * distance**2)

                    weight = np.exp(-0.5 * (xi**2 + yi**2 + zi**2))

                    dose_contrib = dcf_point * geometric * buildup * attenuation * weight
                    total_dose += dose_contrib

        total_dose *= activity / 720.0 * 3.6e6  # to mrem/hr
        doses.append(total_dose)

        # Calculate viewing angle from receptor to puff center
        horizontal_dist = abs(rx)
        vertical_angle = np.arctan2(puff_z - receptor_z, horizontal_dist) * 180 / PI
        vertical_angles.append(vertical_angle)
        horizontal_distances.append(horizontal_dist)

    return receptor_positions, doses, vertical_angles, horizontal_distances

def main():
    print("Analyzing non-monotonic cloudshine dose pattern")
    print("=" * 60)

    # Puff parameters
    puff_z = 50  # Height (m)
    sigma_y = 100  # Horizontal dispersion (m)
    sigma_z = 50  # Vertical dispersion (m)
    activity = 1e3  # Activity (Ci)

    # 1. Detailed analysis near the center
    print("\n1. Analyzing dose profile with high resolution...")
    positions, doses, angles, hdists = calculate_geometric_effects(puff_z, sigma_y, sigma_z)

    # Find local maximum/minimum
    doses_array = np.array(doses)
    center_idx = len(doses) // 2

    # Look for non-monotonic behavior near center
    search_range = 50  # indices around center
    local_doses = doses_array[center_idx-search_range:center_idx+search_range+1]
    local_positions = positions[center_idx-search_range:center_idx+search_range+1]

    # Find local extrema
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(local_doses)
    valleys, _ = find_peaks(-local_doses)

    print(f"\nPuff parameters:")
    print(f"  Height: {puff_z} m")
    print(f"  σ_y: {sigma_y} m, σ_z: {sigma_z} m")

    if len(peaks) > 0 or len(valleys) > 0:
        print(f"\nNon-monotonic behavior detected!")
        print(f"  Local peaks at positions: {local_positions[peaks] if len(peaks) > 0 else 'None'}")
        print(f"  Local valleys at positions: {local_positions[valleys] if len(valleys) > 0 else 'None'}")

    # 2. Analyze dose contributions by layer
    print("\n2. Analyzing dose contributions by vertical layers...")
    test_positions = [0, 10, 30, 50, 100]

    fig1 = plt.figure(figsize=(15, 10))

    # Plot 1: Dose profile
    ax1 = fig1.add_subplot(221)
    ax1.plot(positions, doses, 'b-', linewidth=2)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Puff center')

    # Highlight non-monotonic region
    zoom_range = 100
    zoom_idx_start = center_idx - zoom_range//2
    zoom_idx_end = center_idx + zoom_range//2
    ax1.fill_between(positions[zoom_idx_start:zoom_idx_end],
                     0, doses[zoom_idx_start:zoom_idx_end],
                     alpha=0.2, color='orange', label='Zoom region')

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Dose Rate (mrem/hr)')
    ax1.set_title('Full Dose Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed view
    ax2 = fig1.add_subplot(222)
    zoom_positions = positions[zoom_idx_start:zoom_idx_end]
    zoom_doses = doses[zoom_idx_start:zoom_idx_end]
    ax2.plot(zoom_positions, zoom_doses, 'b-', linewidth=2)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Puff center')

    # Mark peaks and valleys in zoom
    if len(peaks) > 0:
        ax2.plot(local_positions[peaks], local_doses[peaks], 'r^',
                markersize=10, label='Local peaks')
    if len(valleys) > 0:
        ax2.plot(local_positions[valleys], local_doses[valleys], 'gv',
                markersize=10, label='Local valleys')

    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Dose Rate (mrem/hr)')
    ax2.set_title('Zoomed View (±50m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Viewing angle analysis
    ax3 = fig1.add_subplot(223)
    ax3.plot(positions, angles, 'g-', linewidth=2)
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=45, color='k', linestyle=':', alpha=0.3, label='45° angle')
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Viewing Angle to Puff Center (degrees)')
    ax3.set_title('Geometric Viewing Angle')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Derivative analysis
    ax4 = fig1.add_subplot(224)
    # Calculate numerical derivative
    dx = positions[1] - positions[0]
    dose_derivative = np.gradient(doses, dx)
    ax4.plot(positions, dose_derivative, 'r-', linewidth=1.5)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('X Position (m)')
    ax4.set_ylabel('Dose Rate Gradient (mrem/hr/m)')
    ax4.set_title('First Derivative of Dose')
    ax4.grid(True, alpha=0.3)

    # Highlight where derivative changes sign unexpectedly
    zero_crossings = np.where(np.diff(np.sign(dose_derivative)))[0]
    for zc in zero_crossings:
        if abs(positions[zc]) < 100:  # Only near center
            ax4.axvline(x=positions[zc], color='orange', linestyle=':', alpha=0.5)

    plt.suptitle(f'Non-Monotonic Behavior Analysis\nPuff: z={puff_z}m, σ_y={sigma_y}m, σ_z={sigma_z}m',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cloudshine_pattern_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved analysis to: cloudshine_pattern_analysis.png")

    # 3. Theoretical explanation
    print("\n3. Theoretical Explanation of Non-Monotonic Behavior:")
    print("-" * 50)

    # Calculate critical distance where viewing angle is maximum
    critical_distance = puff_z  # For maximum solid angle
    print(f"\nCritical distance (z = horizontal distance): {critical_distance:.1f} m")

    # Dose at critical points
    critical_idx = np.argmin(np.abs(positions - critical_distance))
    dose_at_critical = doses[critical_idx]
    dose_at_center = doses[center_idx]

    print(f"Dose at center (x=0): {dose_at_center:.3e} mrem/hr")
    print(f"Dose at critical distance (x={critical_distance:.0f}m): {dose_at_critical:.3e} mrem/hr")

    if dose_at_critical > dose_at_center:
        print("\n⚠️ NON-MONOTONIC BEHAVIOR CONFIRMED!")
        print("   Dose increases away from center before decreasing.")
        print("\nPossible causes:")
        print("1. Geometric effect: Optimal viewing angle to puff volume")
        print("2. 720-point discretization: Sampling artifacts")
        print("3. Combined effect of:")
        print("   - Decreasing 1/r² geometric factor")
        print("   - Increasing visible puff volume")
        print("   - Vertical separation reducing at optimal angle")

    # 4. Create 3D visualization of point contributions
    fig2 = plt.figure(figsize=(14, 10))
    ax3d = fig2.add_subplot(111, projection='3d')

    # Plot 720 points colored by contribution
    nx, ny, nz = 10, 12, 6
    x_norm = np.linspace(-2, 2, nx)
    y_norm = np.linspace(-2, 2, ny)
    z_norm = np.linspace(-2, 2, nz)

    # Calculate contributions for a receptor at x=50m (near critical distance)
    receptor_x = 50
    receptor_y = 0
    receptor_z = 1.0

    contributions = []
    positions_3d = []

    for xi in x_norm:
        for yi in y_norm:
            for zi in z_norm:
                source_x = xi * sigma_y
                source_y = yi * sigma_y
                source_z = puff_z + zi * sigma_z

                dx = receptor_x - source_x
                dy = receptor_y - source_y
                dz = receptor_z - source_z

                distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

                buildup = 1.0 + K_BUILD * MU_AIR * distance
                attenuation = np.exp(-MU_AIR * distance)
                geometric = 1.0 / (4.0 * PI * distance**2)
                weight = np.exp(-0.5 * (xi**2 + yi**2 + zi**2))

                contribution = geometric * buildup * attenuation * weight
                contributions.append(contribution)
                positions_3d.append([source_x, source_y, source_z])

    contributions = np.array(contributions)
    positions_3d = np.array(positions_3d)

    # Normalize for visualization
    contributions_norm = contributions / contributions.max()

    # Create scatter plot with size proportional to contribution
    scatter = ax3d.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                          c=contributions_norm, s=contributions_norm*100,
                          cmap='hot', alpha=0.6)

    # Add receptor position
    ax3d.scatter([receptor_x], [receptor_y], [receptor_z],
                color='blue', s=200, marker='*', label='Receptor')

    # Add puff center
    ax3d.scatter([0], [0], [puff_z],
                color='green', s=200, marker='^', label='Puff Center')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'720-Point Contributions to Receptor at x={receptor_x}m\n(Point size ∝ dose contribution)')
    ax3d.legend()

    plt.colorbar(scatter, ax=ax3d, label='Normalized Contribution')

    plt.tight_layout()
    plt.savefig('cloudshine_3d_contributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved 3D contribution analysis to: cloudshine_3d_contributions.png")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    # Install scipy if needed
    try:
        from scipy.signal import find_peaks
    except ImportError:
        import subprocess
        print("Installing scipy...")
        subprocess.run(["pip", "install", "scipy"], check=True)
        from scipy.signal import find_peaks

    main()