"""
Simple scatter plot showing all 720 sectors have equal mass.
Just dots, no colors.
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

def plot_720_dots():
    """Create a simple scatter plot of 720 equal-mass sectors."""

    probs = exact_cell_probabilities()

    # Create figure
    plt.figure(figsize=(14, 6))

    # Plot dots
    x = np.arange(720)
    plt.scatter(x, probs, s=5, c='black', alpha=0.6)

    # Add horizontal line for expected value
    expected = 1/720
    plt.axhline(y=expected, color='red', linestyle='--', linewidth=1.5,
                label=f'Expected: 1/720 = {expected:.6f}')

    # Labels and title
    plt.xlabel('Sector Index (0-719)', fontsize=12)
    plt.ylabel('Probability Mass', fontsize=12)
    plt.title('720 Equal-Mass Sectors', fontsize=14, fontweight='bold')

    # Set axis limits
    plt.xlim(-10, 730)
    plt.ylim(expected * 0.95, expected * 1.05)

    # Grid
    plt.grid(True, alpha=0.3)

    # Legend
    plt.legend(loc='upper right')

    # Add text showing all values are the same
    plt.text(360, expected * 1.03, f'All 720 points at y = {probs[0]:.8f}',
             ha='center', fontsize=10, color='blue')

    # Tight layout
    plt.tight_layout()

    # Save
    filename = '720_dots_equal_mass.png'
    plt.savefig(filename, dpi=150)
    plt.show()

    print(f"Figure saved as: {filename}")
    print(f"\nAll 720 sectors have mass = {probs[0]:.10f}")
    print(f"Expected mass (1/720)    = {1/720:.10f}")
    print(f"Difference               = {abs(probs[0] - 1/720):.2e}")

    return probs

if __name__ == "__main__":
    print("="*60)
    print("Plotting 720 Equal-Mass Sectors (dots only)")
    print("="*60)

    probs = plot_720_dots()

    print("="*60)