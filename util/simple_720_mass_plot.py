"""
Simple visualization showing all 720 sectors have equal mass.
One clear graph: x-axis = sector index (0-719), y-axis = probability mass
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def exact_cell_probabilities():
    """Calculate exact probability for each of 720 cells."""
    # Radial bounds from Rayleigh CDF quantiles
    rb = [0.0]
    for i in range(1, 6):
        rb.append(np.sqrt(-2.0 * np.log(1.0 - i/6.0)))
    rb.append(np.inf)

    # Vertical bounds from normal CDF quantiles
    zb = [-np.inf]
    for k in range(1, 10):
        zb.append(norm.ppf(k/10.0))
    zb.append(np.inf)

    dtheta = 2*np.pi/12.0

    probs = []
    for i in range(6):
        r1, r2 = rb[i], rb[i+1]
        FR1 = 0.0 if r1 == 0.0 else 1.0 - np.exp(-0.5*r1*r1)
        FR2 = 1.0 if np.isinf(r2) else 1.0 - np.exp(-0.5*r2*r2)
        p_r = FR2 - FR1

        for k in range(10):
            z1, z2 = zb[k], zb[k+1]
            p_z = norm.cdf(z2) - norm.cdf(z1)

            for j in range(12):
                p_theta = dtheta/(2*np.pi)
                probs.append(p_r * p_z * p_theta)

    return np.array(probs)

def plot_720_equal_mass():
    """Create a simple, clear plot of 720 equal-mass sectors."""

    probs = exact_cell_probabilities()

    # Create figure with specific size for clarity
    plt.figure(figsize=(14, 7))

    # Main plot
    ax = plt.subplot(111)

    # Plot all 720 sectors
    x = np.arange(720)
    bars = ax.bar(x, probs, width=1.0, edgecolor='none', linewidth=0)

    # Color by radial shell (6 colors, each for 120 consecutive sectors)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for i in range(720):
        shell_idx = i // 120
        bars[i].set_color(colors[shell_idx])
        bars[i].set_alpha(0.8)

    # Add horizontal line for expected value
    expected = 1/720
    ax.axhline(y=expected, color='red', linestyle='--', linewidth=2.5,
               label=f'Expected value: 1/720 = {expected:.6f}', zorder=5)

    # Labels and title
    ax.set_xlabel('Sector Index (0-719)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Mass', fontsize=14, fontweight='bold')
    ax.set_title('Equal Mass Distribution Across 720 Sectors\n' +
                 '6 radial shells × 10 vertical layers × 12 angular sectors',
                 fontsize=16, fontweight='bold', pad=20)

    # Set axis limits for clarity
    ax.set_xlim(-5, 724)
    ax.set_ylim(0, expected * 1.2)

    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add text annotations
    actual_mean = probs.mean()
    actual_std = probs.std()

    # Statistics box
    stats_text = f'Statistics:\nMean = {actual_mean:.8f}\nStd = {actual_std:.2e}\nSum = {probs.sum():.6f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Add shell labels
    for i in range(6):
        x_pos = i * 120 + 60
        y_pos = expected * 1.15
        ax.text(x_pos, y_pos, f'Shell {i+1}',
                ha='center', fontsize=10, color=colors[i], fontweight='bold')

    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    filename = '720_sectors_equal_mass_simple.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nFigure saved as: {filename}")

    return probs

if __name__ == "__main__":
    print("="*60)
    print("Creating Simple 720 Equal-Mass Sectors Visualization")
    print("="*60)

    probs = plot_720_equal_mass()

    # Print verification
    print(f"\nVerification Results:")
    print(f"  Expected value (1/720): {1/720:.10f}")
    print(f"  Actual mean:            {probs.mean():.10f}")
    print(f"  Difference:             {abs(probs.mean() - 1/720):.2e}")
    print(f"  All values equal?       {len(np.unique(probs)) == 1}")

    if len(np.unique(probs)) == 1:
        print("\n✓ SUCCESS: All 720 sectors have EXACTLY the same mass!")
    else:
        print(f"\n✗ WARNING: Found {len(np.unique(probs))} different values")

    print("="*60)