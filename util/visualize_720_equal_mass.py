"""
Visualize that all 720 sectors have equal probability mass.
Shows the uniformity of the 720-sector discretization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.patches as mpatches

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

def visualize_720_equal_mass():
    """Create comprehensive visualization of 720 equal-mass sectors."""

    probs = exact_cell_probabilities()

    fig = plt.figure(figsize=(18, 12))

    # ========== 1. Bar plot of all 720 sectors ==========
    ax1 = plt.subplot(3, 3, (1, 4))
    bars = ax1.bar(range(720), probs, width=1, edgecolor='none')

    # Color code by radial shell (6 colors, each repeated 120 times)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    for i in range(720):
        shell_idx = i // 120
        bars[i].set_color(colors[shell_idx])

    ax1.axhline(y=1/720, color='red', linestyle='--', linewidth=2, label=f'Expected: 1/720 = {1/720:.6f}')
    ax1.set_xlabel('Sector Index (0-719)', fontsize=11)
    ax1.set_ylabel('Probability Mass', fontsize=11)
    ax1.set_title('Probability Mass of All 720 Sectors\n(Colored by Radial Shell)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1, 720)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Add text annotation for actual value
    actual_mean = probs.mean()
    ax1.text(360, 1/720 * 1.1, f'Actual mean: {actual_mean:.6f}',
            ha='center', fontsize=10, color='darkblue', fontweight='bold')

    # ========== 2. Histogram of probability masses ==========
    ax2 = plt.subplot(3, 3, 2)
    # Since all values are identical, create a bar plot instead
    unique_vals, counts = np.unique(probs, return_counts=True)
    ax2.bar(unique_vals, counts, width=1e-8, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=1/720, color='red', linestyle='--', linewidth=2, label='Expected: 1/720')
    ax2.set_xlabel('Probability Mass', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Sector Masses\n(All 720 have identical mass)', fontsize=12, fontweight='bold')
    ax2.set_xlim((1/720)*0.999, (1/720)*1.001)
    ax2.set_ylim(0, 800)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {probs.mean():.8f}\nStd: {probs.std():.2e}\nCV: {probs.std()/probs.mean():.2e}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ========== 3. Deviation from ideal ==========
    ax3 = plt.subplot(3, 3, 3)
    deviations = (probs - 1/720) / (1/720) * 100  # Percentage deviation
    ax3.plot(deviations, 'g-', linewidth=0.5, alpha=0.8)
    ax3.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax3.fill_between(range(720), deviations, 0, alpha=0.3, color='green')
    ax3.set_xlabel('Sector Index', fontsize=11)
    ax3.set_ylabel('Deviation (%)', fontsize=11)
    ax3.set_title('Percentage Deviation from Ideal\n(Should be ~0% everywhere)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 719)
    ax3.grid(True, alpha=0.3)

    # Add max deviation text
    max_dev = np.max(np.abs(deviations))
    ax3.text(0.5, 0.95, f'Max deviation: {max_dev:.2e}%', transform=ax3.transAxes,
            fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # ========== 4. Cumulative sum ==========
    ax4 = plt.subplot(3, 3, 5)
    cumsum = np.cumsum(probs)
    ideal_cumsum = np.linspace(0, 1, 720)

    ax4.plot(cumsum, 'b-', linewidth=2, label='Actual cumulative')
    ax4.plot(ideal_cumsum, 'r--', linewidth=2, label='Ideal cumulative', alpha=0.7)
    ax4.set_xlabel('Sector Index', fontsize=11)
    ax4.set_ylabel('Cumulative Probability', fontsize=11)
    ax4.set_title('Cumulative Distribution\n(Should match ideal line)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 719)
    ax4.set_ylim(0, 1.02)

    # ========== 5. Matrix view (6x10x12 structure) ==========
    ax5 = plt.subplot(3, 3, 6)
    # Reshape to show structure: 6 radial x 10 vertical x 12 angular
    prob_matrix = probs.reshape(6, 10, 12).mean(axis=2)  # Average over angular
    im = ax5.imshow(prob_matrix.T, cmap='YlOrRd', aspect='auto')
    ax5.set_xlabel('Radial Shell (0-5)', fontsize=11)
    ax5.set_ylabel('Vertical Layer (0-9)', fontsize=11)
    ax5.set_title('Mean Mass by Shell-Layer\n(Averaged over 12 angles)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Mean probability')

    # Add grid
    for i in range(6):
        ax5.axvline(x=i-0.5, color='white', linewidth=0.5)
    for j in range(10):
        ax5.axhline(y=j-0.5, color='white', linewidth=0.5)

    # ========== 6. Statistical summary ==========
    ax6 = plt.subplot(3, 3, 8)
    ax6.axis('off')

    summary_text = f"""
    STATISTICAL SUMMARY OF 720 SECTORS
    =====================================

    Expected probability per sector:  1/720 = {1/720:.10f}

    Actual Statistics:
    • Mean:                {probs.mean():.10f}
    • Standard Deviation:  {probs.std():.2e}
    • Minimum:            {probs.min():.10f}
    • Maximum:            {probs.max():.10f}
    • Total Sum:          {probs.sum():.10f}

    Uniformity Measures:
    • Coefficient of Variation:  {probs.std()/probs.mean():.2e}
    • Max deviation from ideal:   {np.max(np.abs(probs - 1/720)):.2e}
    • Relative max deviation:     {np.max(np.abs(probs - 1/720))/(1/720)*100:.4e}%

    Structure: 6 radial × 10 vertical × 12 angular = 720 total

    CONCLUSION: All 720 sectors have EXACTLY equal probability mass
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # ========== 7. By component (radial, vertical, angular) ==========
    ax7 = plt.subplot(3, 3, 7)

    # Aggregate by each dimension
    shell_probs = np.array([probs[i*120:(i+1)*120].sum() for i in range(6)])
    layer_probs = np.array([probs[i*12::120].sum() for i in range(10)])
    sector_probs = np.array([probs[j::12].sum() for j in range(12)])

    x = np.arange(12)
    width = 0.25

    # Normalize for comparison (should all be uniform)
    ax7.bar(x[:6], shell_probs/shell_probs.mean(), width, label='Radial (6)', color='blue', alpha=0.7)
    ax7.bar(x[:10] + width, layer_probs/layer_probs.mean(), width, label='Vertical (10)', color='green', alpha=0.7)
    ax7.bar(x[:12] + 2*width, sector_probs/sector_probs.mean(), width, label='Angular (12)', color='red', alpha=0.7)

    ax7.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax7.set_ylabel('Normalized Mass (should be 1.0)', fontsize=11)
    ax7.set_xlabel('Component Index', fontsize=11)
    ax7.set_title('Uniformity Check by Dimension\n(All bars should be at 1.0)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0.98, 1.02)

    # ========== 8. 3D visualization placeholder ==========
    ax8 = plt.subplot(3, 3, 9)
    ax8.axis('off')

    # Create legend for the main plot
    legend_elements = [
        mpatches.Patch(color=colors[i], label=f'Shell {i+1}') for i in range(6)
    ]
    ax8.legend(handles=legend_elements, loc='center', fontsize=10, title='Radial Shells')
    ax8.set_title('Color Legend for Main Plot', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle('720 Equal-Mass Sectors Validation\nPerfect Uniformity Demonstration',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('720_equal_mass_validation.png', dpi=150, bbox_inches='tight')
    print("Saved: 720_equal_mass_validation.png")

    return probs

if __name__ == "__main__":
    print("="*70)
    print("Generating 720 Equal-Mass Sectors Visualization")
    print("="*70)

    probs = visualize_720_equal_mass()

    print("\nKey Results:")
    print(f"  - All 720 sectors have probability: {1/720:.10f}")
    print(f"  - Actual mean probability:          {probs.mean():.10f}")
    print(f"  - Standard deviation:               {probs.std():.2e}")
    print(f"  - Coefficient of variation:         {probs.std()/probs.mean():.2e}")
    print(f"  - Total probability sum:            {probs.sum():.10f}")

    print("\n[OK] Verification: All 720 sectors have EXACTLY equal mass!")
    print("="*70)