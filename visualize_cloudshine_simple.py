"""
Simplified Cloudshine 3D Visualization
=======================================
A simplified version that generates just one visualization quickly.
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

def cloudshine_720_points(puff_x, puff_y, puff_z, sigma_y, sigma_z, receptor_x, receptor_y, receptor_z, activity):
    """
    Calculate cloudshine dose using 720-point summation.
    Simplified version for demonstration.
    """
    # Generate 720 points in a 10x12x6 grid
    nx, ny, nz = 10, 12, 6

    # Create normalized grid points
    x_norm = np.linspace(-2, 2, nx)
    y_norm = np.linspace(-2, 2, ny)
    z_norm = np.linspace(-2, 2, nz)

    total_dose = 0.0
    dcf_point = 1e-6  # Dose conversion factor

    # Sum contributions from all points
    for xi in x_norm:
        for yi in y_norm:
            for zi in z_norm:
                # Scale to actual puff dimensions
                source_x = puff_x + xi * sigma_y
                source_y = puff_y + yi * sigma_y
                source_z = puff_z + zi * sigma_z

                # Calculate distance
                dx = receptor_x - source_x
                dy = receptor_y - source_y
                dz = receptor_z - source_z
                distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

                # Point kernel calculation
                buildup = 1.0 + K_BUILD * MU_AIR * distance
                attenuation = np.exp(-MU_AIR * distance)
                geometric = 1.0 / (4.0 * PI * distance**2)

                # Gaussian weight
                weight = np.exp(-0.5 * (xi**2 + yi**2 + zi**2))

                # Add contribution
                dose_contrib = dcf_point * geometric * buildup * attenuation * weight
                total_dose += dose_contrib

    # Normalize by number of points and apply activity
    total_dose *= activity / 720.0

    # Convert to mrem/hr
    return total_dose * 3.6e6

def main():
    print("Cloudshine 3D Visualization - 720-Point Method")
    print("=" * 60)

    # Puff parameters (small puff case)
    puff_x, puff_y, puff_z = 0, 0, 50  # Position (m)
    sigma_y, sigma_z = 100, 50  # Dispersion parameters (m)
    activity = 1e3  # Activity (Ci)

    print(f"Puff parameters:")
    print(f"  Position: ({puff_x}, {puff_y}, {puff_z}) m")
    print(f"  Sigma: σ_y={sigma_y}m, σ_z={sigma_z}m")
    print(f"  Activity: {activity:.1e} Ci")

    # Create ground grid
    grid_size = 40  # Reduced for faster computation
    extent = 500  # Grid extent (m)

    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    print(f"\nCalculating dose field on {grid_size}x{grid_size} grid...")

    # Calculate dose field
    Z = np.zeros_like(X)
    receptor_z = 1.0  # 1m above ground

    for i in range(grid_size):
        if i % 10 == 0:
            print(f"  Progress: {i}/{grid_size} rows completed")
        for j in range(grid_size):
            Z[i, j] = cloudshine_720_points(
                puff_x, puff_y, puff_z, sigma_y, sigma_z,
                X[i, j], Y[i, j], receptor_z, activity
            )

    print("Calculation complete!")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Dose Rate (mrem/hr)')
    ax1.set_title('3D Cloudshine Dose Surface')
    ax1.view_init(elev=25, azim=45)

    # Add puff marker
    ax1.scatter([puff_x], [puff_y], [Z.max()], color='red', s=100, marker='*', label='Puff Center')

    # Top-down contour view
    ax2 = fig.add_subplot(222)
    levels = 20
    contour = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    # Mark puff center and sigma boundaries
    ax2.plot(puff_x, puff_y, 'r*', markersize=15, label='Puff Center')
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(puff_x + sigma_y * np.cos(theta),
             puff_y + sigma_y * np.sin(theta),
             'r--', label='1σ boundary', linewidth=2)
    ax2.plot(puff_x + 2*sigma_y * np.cos(theta),
             puff_y + 2*sigma_y * np.sin(theta),
             'r:', label='2σ boundary', linewidth=1.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View with Contours')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2, label='Dose Rate (mrem/hr)')

    # X-axis cross-section
    ax3 = fig.add_subplot(223)
    center_y = grid_size // 2
    ax3.plot(x, Z[center_y, :], 'b-', linewidth=2)
    ax3.axvline(x=puff_x, color='r', linestyle='--', label='Puff Center')
    ax3.fill_between(x, 0, Z[center_y, :], alpha=0.3, color='blue')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Dose Rate (mrem/hr)')
    ax3.set_title('Cross-Section (Y=0)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Y-axis cross-section
    ax4 = fig.add_subplot(224)
    center_x = grid_size // 2
    ax4.plot(y, Z[:, center_x], 'g-', linewidth=2)
    ax4.axvline(x=puff_y, color='r', linestyle='--', label='Puff Center')
    ax4.fill_between(y, 0, Z[:, center_x], alpha=0.3, color='green')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Dose Rate (mrem/hr)')
    ax4.set_title('Cross-Section (X=0)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Main title
    fig.suptitle('Cloudshine Dose Field - 720-Point Summation Method', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cloudshine_720point_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: cloudshine_720point_visualization.png")

    # Print statistics
    print(f"\nDose Field Statistics:")
    print(f"  Maximum dose: {Z.max():.2e} mrem/hr")
    print(f"  Mean dose: {Z.mean():.2e} mrem/hr")
    print(f"  Minimum dose: {Z.min():.2e} mrem/hr")

    # Calculate dose at specific distances
    distances = [50, 100, 200, 300, 400, 500]
    print(f"\nDose rates at specific distances:")
    for d in distances:
        dose = cloudshine_720_points(puff_x, puff_y, puff_z, sigma_y, sigma_z,
                                    d, 0, receptor_z, activity)
        print(f"  {d:3d}m: {dose:.2e} mrem/hr")

if __name__ == "__main__":
    main()