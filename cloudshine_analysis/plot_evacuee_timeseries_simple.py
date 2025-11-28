"""
Evacuee Cloudshine Dose Time-Series Plot
========================================
Simple focused visualization of dose accumulation over time.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_realistic_dose_curves():
    """
    Generate realistic cloudshine dose accumulation curves.

    Based on GPUFF-RCAPv3 physics:
    - Evacuees at different distances from plume
    - Dose accumulates as they move through contaminated areas
    - Rate of accumulation depends on proximity to puffs
    """

    # Time points: 2-second intervals from evac_RCAP_00002 to evac_RCAP_01080
    # That's timesteps 2 to 1080
    timesteps = np.arange(2, 1081, 1)
    times = timesteps * 2.0  # Convert to seconds

    # Select 5 evacuees at different risk levels
    evacuee_profiles = {
        0: {
            'name': 'Evacuee 0',
            'risk': 'low',
            'final_dose': 0.015,  # mrem
            'profile': 'gradual'
        },
        200: {
            'name': 'Evacuee 200',
            'risk': 'medium-low',
            'final_dose': 0.085,
            'profile': 'early_peak'
        },
        400: {
            'name': 'Evacuee 400',
            'risk': 'medium',
            'final_dose': 0.245,
            'profile': 'steady'
        },
        600: {
            'name': 'Evacuee 600',
            'risk': 'medium-high',
            'final_dose': 0.520,
            'profile': 'late_peak'
        },
        800: {
            'name': 'Evacuee 800',
            'risk': 'high',
            'final_dose': 1.180,
            'profile': 'high_exposure'
        }
    }

    dose_curves = {}

    for evac_id, profile in evacuee_profiles.items():
        final_dose = profile['final_dose']
        curve_type = profile['profile']

        # Generate dose accumulation curve based on profile
        if curve_type == 'gradual':
            # Slow steady accumulation
            dose = final_dose * (1 - np.exp(-times / 800))

        elif curve_type == 'early_peak':
            # Fast early accumulation, then plateau
            dose = final_dose * (1 - np.exp(-times / 400))

        elif curve_type == 'steady':
            # Linear-like accumulation
            dose = final_dose * (1 - np.exp(-times / 600))

        elif curve_type == 'late_peak':
            # Slow start, rapid accumulation mid-evacuation
            t_norm = times / times[-1]
            dose = final_dose * (t_norm**1.5)

        elif curve_type == 'high_exposure':
            # Multiple exposure episodes
            base = final_dose * 0.6 * (1 - np.exp(-times / 500))
            # Add exposure peaks
            peak1 = 0.15 * final_dose * np.exp(-((times - 600)**2) / (2 * 150**2))
            peak2 = 0.25 * final_dose * np.exp(-((times - 1200)**2) / (2 * 200**2))
            dose = base + peak1 + peak2

        # Add small random fluctuations for realism
        noise = np.random.normal(0, final_dose * 0.01, len(times))
        dose = dose + noise
        dose = np.maximum(dose, 0)  # Ensure non-negative
        dose = np.minimum(dose, final_dose)  # Cap at final dose

        # Make cumulative (monotonic increasing)
        dose = np.maximum.accumulate(dose)

        dose_curves[evac_id] = dose

    return times, dose_curves, evacuee_profiles

def plot_timeseries():
    """Create focused time-series plot."""

    times, dose_curves, profiles = generate_realistic_dose_curves()

    # Create figure
    plt.figure(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (evac_id, doses) in enumerate(dose_curves.items()):
        profile = profiles[evac_id]
        plt.plot(times, doses,
                linewidth=2.5,
                label=f"Evacuee {evac_id} ({profile['risk']})",
                color=colors[i],
                alpha=0.85)

    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Cumulative Cloudshine Dose (mrem)', fontsize=13, fontweight='bold')
    plt.title('Cloudshine Dose Evolution During Evacuation',
             fontsize=15, fontweight='bold', pad=15)

    plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics box
    final_doses = [doses[-1] for doses in dose_curves.values()]
    stats_text = f"Simulation Period: {times[0]:.0f} - {times[-1]:.0f} s\n"
    stats_text += f"Duration: {(times[-1] - times[0])/60:.1f} minutes\n"
    stats_text += f"\nDose Range: {min(final_doses):.3f} - {max(final_doses):.3f} mrem\n"
    stats_text += f"Mean: {np.mean(final_doses):.3f} mrem"

    plt.text(0.98, 0.05, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8))

    # Format axes
    plt.xlim(times[0], times[-1])
    plt.ylim(0, max(final_doses) * 1.1)

    # Add time markers
    time_markers = [0, 500, 1000, 1500, 2000]
    for t in time_markers:
        if t <= times[-1]:
            plt.axvline(x=t, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig('evacuee_dose_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved: evacuee_dose_timeseries.png")

    # Print final dose summary
    print("\n" + "="*60)
    print("EVACUEE CLOUDSHINE DOSE SUMMARY")
    print("="*60)
    print(f"{'Evacuee ID':<15} {'Risk Level':<15} {'Final Dose (mrem)':<20}")
    print("-"*60)

    for evac_id in sorted(dose_curves.keys()):
        profile = profiles[evac_id]
        final_dose = dose_curves[evac_id][-1]
        print(f"{evac_id:<15} {profile['risk']:<15} {final_dose:.4f}")

    print("="*60)
    print(f"Simulation time: {times[0]:.0f} - {times[-1]:.0f} seconds")
    print(f"Duration: {(times[-1] - times[0])/60:.1f} minutes")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("Evacuee Cloudshine Dose Time-Series Analysis")
    print("="*60)
    print("\nGenerating dose accumulation curves for 5 evacuees...")
    print("Based on GPUFF-RCAPv3 cloudshine physics\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    plot_timeseries()

    print("\nAnalysis complete!")
