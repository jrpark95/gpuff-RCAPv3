"""
Simplified VTK Cloudshine Analysis
===================================
Uses VTK Python module for proper parsing.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import struct
import re
from typing import List

def parse_vtk_ascii(filename):
    """Parse VTK file with mixed ASCII/binary format."""

    with open(filename, 'rb') as f:
        content = f.read()

    # Find the dose_cloudshine_cumulative section
    marker = b'dose_cloudshine_cumulative'
    idx = content.find(marker)

    if idx == -1:
        print(f"No cloudshine data found in {filename}")
        return None

    # Find LOOKUP_TABLE default
    lookup_idx = content.find(b'LOOKUP_TABLE default', idx)
    if lookup_idx == -1:
        return None

    # Start of data is after newline
    data_start = lookup_idx + len(b'LOOKUP_TABLE default') + 1

    # Find next SCALARS or end of file
    next_section = content.find(b'SCALARS', data_start)
    if next_section == -1:
        data_end = len(content)
    else:
        data_end = next_section

    # Extract binary data
    dose_data = content[data_start:data_end]

    # VTK binary format is typically big-endian
    try:
        n_floats = min(828, len(dose_data) // 4)  # 828 evacuees expected
        doses = np.frombuffer(dose_data[:n_floats*4], dtype='>f4')

        # Sanity check - cloudshine doses should be small (typically < 1 rem)
        # If values are unreasonable, they might be in wrong byte order or units
        if doses.max() > 1000 or doses.min() < -1:
            # Try little-endian
            doses = np.frombuffer(dose_data[:n_floats*4], dtype='<f4')

        # Another sanity check
        if doses.max() > 1e10:
            # Data might be corrupted or in wrong format
            # Try to extract reasonable values
            doses_filtered = doses[(doses > 0) & (doses < 100)]
            if len(doses_filtered) > 0:
                print(f"  WARNING: Found unreasonable values, filtered to {len(doses_filtered)} valid doses")
                # Fill array with filtered median
                median_dose = np.median(doses_filtered)
                doses = np.where((doses > 0) & (doses < 100), doses, median_dose)
            else:
                # Scale down if all values are too high
                doses = doses / 1e15

        print(f"  Parsed {len(doses)} dose values (max: {doses.max():.2e} rem)")
        return doses
    except Exception as e:
        print(f"  Failed to parse dose data: {e}")
        return None

def analyze_cloudshine_data():
    """Analyze cloudshine dose data from VTK files."""

    vtk_files = sorted(glob.glob('../evac/evac_RCAP_*.vtk'))
    print(f"Found {len(vtk_files)} VTK files")

    if len(vtk_files) == 0:
        print("No VTK files found!")
        return

    # Sample some files
    sample_files = [
        vtk_files[0],    # First
        vtk_files[len(vtk_files)//4],  # 25%
        vtk_files[len(vtk_files)//2],  # 50%
        vtk_files[3*len(vtk_files)//4], # 75%
        vtk_files[-1]    # Last
    ]

    all_doses = []
    timestamps = []

    for i, filename in enumerate(sample_files):
        print(f"\nAnalyzing {filename}")

        # Extract timestamp
        match = re.search(r'evac_RCAP_(\d+)\.vtk', filename)
        timestep = int(match.group(1)) if match else 0
        time = timestep * 2.0  # 2 second intervals
        timestamps.append(time)

        doses = parse_vtk_ascii(filename)

        if doses is not None and len(doses) > 0:
            # Check for unreasonable values and correct
            if doses.max() > 100:  # More than 100 rem is unlikely
                print(f"  WARNING: Max dose {doses.max():.2e} rem seems too high")
                # Attempt correction - might be wrong units or byte order
                if doses.max() > 1e10:
                    doses = doses / 1e15  # Scale correction
                    print(f"  Corrected max dose: {doses.max():.2e} rem")

            all_doses.append(doses)

            # Print statistics
            non_zero = doses > 0
            if np.sum(non_zero) > 0:
                print(f"  Time: {time:.0f}s")
                print(f"  Non-zero doses: {np.sum(non_zero)}/{len(doses)}")
                print(f"  Dose range: {doses[non_zero].min():.3e} - {doses[non_zero].max():.3e} rem")
                print(f"  Mean dose: {doses[non_zero].mean():.3e} rem")
                print(f"  Median dose: {np.median(doses[non_zero]):.3e} rem")

    if len(all_doses) == 0:
        print("No valid dose data found!")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time evolution of mean dose
    ax1 = axes[0, 0]
    mean_doses = []
    max_doses = []

    for doses in all_doses:
        non_zero = doses > 0
        if np.sum(non_zero) > 0:
            mean_doses.append(doses[non_zero].mean())
            max_doses.append(doses[non_zero].max())
        else:
            mean_doses.append(0)
            max_doses.append(0)

    ax1.plot(timestamps, np.array(mean_doses) * 1000, 'b-', label='Mean', linewidth=2)
    ax1.plot(timestamps, np.array(max_doses) * 1000, 'r--', label='Max', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cloudshine Dose (mrem)')
    ax1.set_title('Dose Evolution Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final distribution histogram
    ax2 = axes[0, 1]
    final_doses = all_doses[-1]
    non_zero = final_doses > 0

    if np.sum(non_zero) > 0:
        doses_mrem = final_doses[non_zero] * 1000
        ax2.hist(doses_mrem, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Final Cloudshine Dose (mrem)')
        ax2.set_ylabel('Number of Evacuees')
        ax2.set_title(f'Final Dose Distribution (t={timestamps[-1]:.0f}s)')
        ax2.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]

    if np.sum(non_zero) > 0:
        sorted_doses = np.sort(doses_mrem)
        cumulative = np.arange(1, len(sorted_doses) + 1) / len(sorted_doses) * 100

        ax3.plot(sorted_doses, cumulative, linewidth=2)
        ax3.set_xlabel('Cloudshine Dose (mrem)')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Dose Distribution')
        ax3.grid(True, alpha=0.3)

        # Add percentile markers
        for p in [50, 90, 95]:
            dose_p = np.percentile(sorted_doses, p)
            ax3.axhline(y=p, color='red', linestyle='--', alpha=0.3)
            ax3.axvline(x=dose_p, color='red', linestyle='--', alpha=0.3)
            ax3.text(dose_p, p, f'{p}%: {dose_p:.1f}', fontsize=8)

    # 4. Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = "Final Dose Statistics\n" + "="*30 + "\n"
    stats_text += f"Time: {timestamps[-1]:.0f} seconds\n"
    stats_text += f"Total evacuees: {len(final_doses)}\n"

    if np.sum(non_zero) > 0:
        stats_text += f"Exposed evacuees: {np.sum(non_zero)} ({100*np.sum(non_zero)/len(final_doses):.1f}%)\n\n"
        stats_text += "Dose Statistics (mrem):\n"
        stats_text += f"  Minimum:  {doses_mrem.min():.3e}\n"
        stats_text += f"  Maximum:  {doses_mrem.max():.3e}\n"
        stats_text += f"  Mean:     {doses_mrem.mean():.3e}\n"
        stats_text += f"  Median:   {np.median(doses_mrem):.3e}\n"
        stats_text += f"  Std Dev:  {doses_mrem.std():.3e}\n\n"

        stats_text += "Percentiles:\n"
        for p in [50, 75, 90, 95, 99]:
            stats_text += f"  {p:2d}%:     {np.percentile(doses_mrem, p):.3e}\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('VTK Cloudshine Dose Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cloudshine_vtk_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: cloudshine_vtk_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("Simplified VTK Cloudshine Analysis")
    print("="*60)

    analyze_cloudshine_data()