"""
Evacuee Cloudshine Dose Time-Series Analysis
============================================
Tracks cumulative dose over time for selected evacuees.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def parse_vtk_dose(filename):
    """Parse dose_cloudshine_cumulative from VTK file."""

    with open(filename, 'rb') as f:
        content = f.read()

    # Find dose_cloudshine_cumulative section
    marker = b'dose_cloudshine_cumulative'
    idx = content.find(marker)

    if idx == -1:
        return None

    # Find LOOKUP_TABLE default
    lookup_idx = content.find(b'LOOKUP_TABLE default', idx)
    if lookup_idx == -1:
        return None

    # Start of data
    data_start = lookup_idx + len(b'LOOKUP_TABLE default') + 1

    # Find next section or EOF
    next_section = content.find(b'SCALARS', data_start)
    if next_section == -1:
        data_end = len(content)
    else:
        data_end = next_section

    # Extract binary data
    dose_data = content[data_start:data_end]

    try:
        # Try big-endian floats
        n_floats = min(828, len(dose_data) // 4)
        doses = np.frombuffer(dose_data[:n_floats*4], dtype='>f4')

        # Check reasonableness
        if doses.max() > 1000 or np.any(np.isnan(doses)) or np.any(np.isinf(doses)):
            # Try little-endian
            doses = np.frombuffer(dose_data[:n_floats*4], dtype='<f4')

        # Final sanity check - if still unreasonable, return None
        if doses.max() > 1e10 or np.any(np.isnan(doses)) or np.any(np.isinf(doses)):
            return None

        return doses
    except Exception as e:
        return None

def plot_evacuee_timeseries():
    """Plot cumulative dose over time for selected evacuees."""

    vtk_files = sorted(glob.glob('../evac/evac_RCAP_*.vtk'))
    print(f"Found {len(vtk_files)} VTK files")

    if len(vtk_files) == 0:
        print("No VTK files found!")
        return

    # Select 5 evacuee IDs distributed across the population
    evacuee_ids = [0, 200, 400, 600, 800]

    # Storage for time-series data
    times = []
    dose_history = {evac_id: [] for evac_id in evacuee_ids}

    print(f"\nTracking evacuees: {evacuee_ids}")
    print("Processing VTK files...")

    # Parse all files
    for i, filename in enumerate(vtk_files):
        # Extract timestamp
        match = re.search(r'evac_RCAP_(\d+)\.vtk', filename)
        timestep = int(match.group(1)) if match else 0
        time = timestep * 2.0  # 2 second intervals

        doses = parse_vtk_dose(filename)

        if doses is not None and len(doses) >= 828:
            times.append(time)

            # Extract doses for selected evacuees
            for evac_id in evacuee_ids:
                dose_history[evac_id].append(doses[evac_id])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(vtk_files)} files")

    print(f"\nSuccessfully processed {len(times)} timesteps")

    if len(times) == 0:
        print("No valid data found in VTK files!")
        return

    # Convert to arrays
    times = np.array(times)
    for evac_id in evacuee_ids:
        dose_history[evac_id] = np.array(dose_history[evac_id]) * 1000  # Convert to mrem

    # Create plot
    plt.figure(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, evac_id in enumerate(evacuee_ids):
        plt.plot(times, dose_history[evac_id],
                linewidth=2,
                label=f'Evacuee {evac_id}',
                color=colors[i],
                marker='o',
                markersize=3,
                markevery=max(1, len(times)//20))

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Cumulative Cloudshine Dose (mrem)', fontsize=12)
    plt.title('Evacuee Cloudshine Dose Evolution Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    final_doses = [dose_history[evac_id][-1] for evac_id in evacuee_ids]
    stats_text = f"Final time: {times[-1]:.0f}s\n"
    stats_text += f"Final dose range: {min(final_doses):.3f} - {max(final_doses):.3f} mrem"

    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('evacuee_dose_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: evacuee_dose_timeseries.png")

    # Print final statistics
    print("\nFinal Dose Summary:")
    print("=" * 50)
    for evac_id in evacuee_ids:
        final_dose = dose_history[evac_id][-1]
        print(f"  Evacuee {evac_id:3d}: {final_dose:.4f} mrem")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 60)
    print("Evacuee Cloudshine Dose Time-Series Analysis")
    print("=" * 60)

    plot_evacuee_timeseries()

    print("\nAnalysis Complete!")
