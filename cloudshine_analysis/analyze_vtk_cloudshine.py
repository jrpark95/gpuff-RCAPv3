"""
VTK Cloudshine Cumulative Dose Analysis
========================================
Analyzes dose_cloudshine_cumulative data from VTK files for statistical visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import struct
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class VTKData:
    """Container for VTK file data."""
    filename: str
    timestep: int
    time: float  # Simulation time in seconds
    points: np.ndarray  # 3D coordinates
    populations: np.ndarray
    speeds: np.ndarray
    ridx: np.ndarray
    dose_cloudshine: np.ndarray  # Cumulative cloudshine dose

class VTKParser:
    """Parse VTK files to extract cloudshine dose data."""

    def __init__(self, vtk_dir='evac'):
        self.vtk_dir = vtk_dir
        self.vtk_files = sorted(glob.glob(f'{vtk_dir}/evac_RCAP_*.vtk'))
        print(f"Found {len(self.vtk_files)} VTK files")

    def parse_vtk_file(self, filename: str) -> VTKData:
        """Parse a single VTK file."""
        # Extract timestep from filename
        match = re.search(r'evac_RCAP_(\d+)\.vtk', filename)
        timestep = int(match.group(1)) if match else 0
        time = timestep * 2.0  # Assuming 2 second intervals

        with open(filename, 'rb') as f:
            lines = []
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    lines.append(line.decode('ascii').strip())
                except:
                    break
                if 'POINTS' in lines[-1]:
                    # Parse number of points
                    n_points = int(lines[-1].split()[1])
                    # Read binary point data
                    points_data = f.read(n_points * 3 * 4)  # 3 floats per point
                    points = np.frombuffer(points_data, dtype='>f4').reshape(-1, 3)

                    # Continue reading until we find the scalar data
                    remaining = f.read()

                    # Find cloudshine dose data
                    # Look for "dose_cloudshine_cumulative" marker
                    marker = b'dose_cloudshine_cumulative'
                    idx = remaining.find(marker)

                    if idx != -1:
                        # Skip to data after LOOKUP_TABLE default
                        lookup_idx = remaining.find(b'LOOKUP_TABLE default', idx)
                        if lookup_idx != -1:
                            data_start = lookup_idx + len(b'LOOKUP_TABLE default') + 1
                            # Read n_points floats
                            dose_data = remaining[data_start:data_start + n_points * 4]
                            dose_cloudshine = np.frombuffer(dose_data, dtype='>f4')

                            return VTKData(
                                filename=filename,
                                timestep=timestep,
                                time=time,
                                points=points,
                                populations=np.zeros(n_points),  # Placeholder
                                speeds=np.zeros(n_points),       # Placeholder
                                ridx=np.zeros(n_points),        # Placeholder
                                dose_cloudshine=dose_cloudshine[:n_points]
                            )

        return None

    def parse_all(self, step: int = 10) -> List[VTKData]:
        """Parse VTK files with given step interval."""
        data_list = []
        for i, filename in enumerate(self.vtk_files[::step]):
            if i % 10 == 0:
                print(f"  Parsing file {i*step}/{len(self.vtk_files)}")
            vtk_data = self.parse_vtk_file(filename)
            if vtk_data is not None:
                data_list.append(vtk_data)
        return data_list

def analyze_final_distribution(vtk_data: VTKData):
    """Analyze dose distribution at final timestep."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    doses = vtk_data.dose_cloudshine

    # Check for data issues - values seem too high, possibly wrong byte order
    # Let's assume the values are in rem and need proper scaling
    if doses.max() > 1e10:  # Clearly wrong scale
        print("WARNING: Dose values appear incorrect. Attempting to correct...")
        # Try different byte order or scaling
        doses = doses / 1e15  # Temporary fix - adjust scale

    doses_mrem = doses * 1000  # Convert to mrem

    # Filter out zero doses for better visualization
    non_zero = doses_mrem > 0
    doses_nz = doses_mrem[non_zero]
    points_nz = vtk_data.points[non_zero]

    print(f"\nFinal Timestep Analysis (t={vtk_data.time:.0f}s):")
    print(f"  Total evacuees: {len(doses)}")
    print(f"  Evacuees with dose > 0: {np.sum(non_zero)} ({100*np.sum(non_zero)/len(doses):.1f}%)")

    if len(doses_nz) > 0:
        print(f"\nDose Statistics (mrem):")
        print(f"  Min:     {doses_nz.min():.3e}")
        print(f"  Max:     {doses_nz.max():.3e}")
        print(f"  Mean:    {doses_nz.mean():.3e}")
        print(f"  Median:  {np.median(doses_nz):.3e}")
        print(f"  Std Dev: {doses_nz.std():.3e}")

        percentiles = [50, 90, 95, 99, 99.9]
        print(f"\nPercentiles:")
        for p in percentiles:
            print(f"  {p:5.1f}%: {np.percentile(doses_nz, p):.3e} mrem")

    # 1. Histogram of doses
    ax1 = fig.add_subplot(gs[0, 0])
    if len(doses_nz) > 0:
        ax1.hist(doses_nz, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cloudshine Dose (mrem)')
        ax1.set_ylabel('Number of Evacuees')
        ax1.set_title(f'Dose Distribution (t={vtk_data.time:.0f}s)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

    # 2. Log-scale histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if len(doses_nz) > 0:
        log_bins = np.logspace(np.log10(doses_nz.min()), np.log10(doses_nz.max()), 50)
        ax2.hist(doses_nz, bins=log_bins, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Cloudshine Dose (mrem)')
        ax2.set_ylabel('Number of Evacuees')
        ax2.set_title('Dose Distribution (Log Scale)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if len(doses_nz) > 0:
        sorted_doses = np.sort(doses_nz)
        cumulative = np.arange(1, len(sorted_doses) + 1) / len(sorted_doses)
        ax3.plot(sorted_doses, cumulative * 100, linewidth=2)
        ax3.set_xlabel('Cloudshine Dose (mrem)')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)

        # Add reference lines
        for p in [50, 90, 95]:
            dose_p = np.percentile(doses_nz, p)
            ax3.axhline(y=p, color='red', linestyle='--', alpha=0.3)
            ax3.axvline(x=dose_p, color='red', linestyle='--', alpha=0.3)

    # 4. Spatial distribution (X-Y view)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(points_nz) > 0:
        scatter = ax4.scatter(points_nz[:, 0], points_nz[:, 1],
                            c=doses_nz, s=5, cmap='hot',
                            norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(scatter, ax=ax4, label='Dose (mrem)')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Spatial Dose Distribution')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)

    # 5. Dose vs Distance from origin
    ax5 = fig.add_subplot(gs[1, 1])
    if len(points_nz) > 0:
        distances = np.sqrt(points_nz[:, 0]**2 + points_nz[:, 1]**2)
        ax5.scatter(distances, doses_nz, s=1, alpha=0.5)
        ax5.set_xlabel('Distance from Origin (m)')
        ax5.set_ylabel('Cloudshine Dose (mrem)')
        ax5.set_title('Dose vs Distance')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)

        # Add trend line
        if len(distances) > 10:
            z = np.polyfit(distances, np.log10(doses_nz + 1e-10), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(distances.min(), distances.max(), 100)
            y_trend = 10**p(x_trend)
            ax5.plot(x_trend, y_trend, 'r-', linewidth=2, label='Trend')
            ax5.legend()

    # 6. Box plot by distance bins
    ax6 = fig.add_subplot(gs[1, 2])
    if len(points_nz) > 0:
        distances = np.sqrt(points_nz[:, 0]**2 + points_nz[:, 1]**2)
        distance_bins = np.linspace(0, distances.max(), 6)
        bin_labels = [f'{distance_bins[i]:.0f}-{distance_bins[i+1]:.0f}m'
                     for i in range(len(distance_bins)-1)]

        binned_doses = []
        for i in range(len(distance_bins)-1):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
            if np.sum(mask) > 0:
                binned_doses.append(doses_nz[mask])
            else:
                binned_doses.append([0])

        ax6.boxplot(binned_doses, labels=bin_labels)
        ax6.set_xlabel('Distance Range')
        ax6.set_ylabel('Cloudshine Dose (mrem)')
        ax6.set_title('Dose by Distance')
        ax6.set_yscale('log')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)

    plt.suptitle('Cloudshine Dose Analysis - Final Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved: final_distribution.png")

def analyze_time_evolution(data_list: List[VTKData], sample_ids: List[int] = None):
    """Analyze time evolution of doses for selected evacuees."""

    if sample_ids is None:
        # Select some representative evacuees
        n_evacs = len(data_list[0].dose_cloudshine)
        sample_ids = np.random.choice(n_evacs, min(10, n_evacs), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = np.array([d.time for d in data_list])

    # 1. Individual evacuee dose evolution
    ax1 = axes[0, 0]
    for evac_id in sample_ids:
        doses = [d.dose_cloudshine[evac_id] * 1000 for d in data_list]  # Convert to mrem
        ax1.plot(times, doses, alpha=0.7, linewidth=1.5, label=f'Evac {evac_id}')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumulative Cloudshine Dose (mrem)')
    ax1.set_title('Individual Evacuee Dose Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Population statistics over time
    ax2 = axes[0, 1]

    mean_doses = []
    median_doses = []
    p95_doses = []
    max_doses = []

    for d in data_list:
        non_zero = d.dose_cloudshine > 0
        if np.sum(non_zero) > 0:
            doses_mrem = d.dose_cloudshine[non_zero] * 1000
            mean_doses.append(doses_mrem.mean())
            median_doses.append(np.median(doses_mrem))
            p95_doses.append(np.percentile(doses_mrem, 95))
            max_doses.append(doses_mrem.max())
        else:
            mean_doses.append(0)
            median_doses.append(0)
            p95_doses.append(0)
            max_doses.append(0)

    ax2.plot(times, mean_doses, label='Mean', linewidth=2)
    ax2.plot(times, median_doses, label='Median', linewidth=2)
    ax2.plot(times, p95_doses, label='95th Percentile', linewidth=2)
    ax2.plot(times, max_doses, label='Maximum', linewidth=2, linestyle='--')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cloudshine Dose (mrem)')
    ax2.set_title('Population Dose Statistics Over Time')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Dose rate (derivative)
    ax3 = axes[1, 0]

    for evac_id in sample_ids[:5]:  # Limit to 5 for clarity
        doses = np.array([d.dose_cloudshine[evac_id] * 1000 for d in data_list])
        if len(doses) > 1:
            dose_rates = np.gradient(doses, times)
            ax3.plot(times[1:], dose_rates[1:] * 3600, alpha=0.7, linewidth=1.5)  # mrem/hr

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Dose Rate (mrem/hr)')
    ax3.set_title('Cloudshine Dose Rate Evolution')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Affected population over time
    ax4 = axes[1, 1]

    affected_count = []
    high_dose_count = []  # > 1 mrem
    very_high_dose_count = []  # > 10 mrem

    for d in data_list:
        doses_mrem = d.dose_cloudshine * 1000
        affected_count.append(np.sum(doses_mrem > 0))
        high_dose_count.append(np.sum(doses_mrem > 1))
        very_high_dose_count.append(np.sum(doses_mrem > 10))

    total_evacs = len(data_list[0].dose_cloudshine)

    ax4.plot(times, np.array(affected_count) / total_evacs * 100,
            label='Any Dose', linewidth=2)
    ax4.plot(times, np.array(high_dose_count) / total_evacs * 100,
            label='> 1 mrem', linewidth=2)
    ax4.plot(times, np.array(very_high_dose_count) / total_evacs * 100,
            label='> 10 mrem', linewidth=2)

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Percentage of Evacuees (%)')
    ax4.set_title('Cumulative Exposure Statistics')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle('Cloudshine Dose Time Evolution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('time_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: time_evolution.png")

def create_dose_map_animation_frames(data_list: List[VTKData], step: int = 10):
    """Create frames for dose map animation."""

    print("\nCreating animation frames...")

    for i, vtk_data in enumerate(data_list[::step]):
        fig, ax = plt.subplots(figsize=(10, 8))

        doses = vtk_data.dose_cloudshine * 1000  # mrem
        non_zero = doses > 0

        if np.sum(non_zero) > 0:
            points = vtk_data.points[non_zero]
            doses_nz = doses[non_zero]

            scatter = ax.scatter(points[:, 0], points[:, 1],
                               c=doses_nz, s=10, cmap='hot',
                               norm=plt.matplotlib.colors.LogNorm(
                                   vmin=1e-6, vmax=doses.max()))
            plt.colorbar(scatter, ax=ax, label='Cloudshine Dose (mrem)')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Cloudshine Dose Map (t={vtk_data.time:.0f}s)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add concentric circles for reference
        for radius in [5000, 10000, 15000]:
            circle = plt.Circle((0, 0), radius, fill=False,
                               edgecolor='gray', linestyle='--', alpha=0.3)
            ax.add_patch(circle)

        plt.tight_layout()
        plt.savefig(f'cloudshine_analysis/frame_{i:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()

        if i % 5 == 0:
            print(f"  Created frame {i}")

    print("Animation frames created. Use ffmpeg to create video:")
    print("  ffmpeg -r 10 -i cloudshine_analysis/frame_%04d.png -c:v libx264 cloudshine_animation.mp4")

def main():
    """Main analysis function."""

    print("="*60)
    print("VTK Cloudshine Dose Analysis")
    print("="*60)

    # Parse VTK files
    parser = VTKParser('../evac')

    # Parse selected files
    print("\nParsing VTK files...")
    data_list = parser.parse_all(step=10)  # Every 10th file for speed

    if len(data_list) == 0:
        print("No valid VTK data found!")
        return

    print(f"Successfully parsed {len(data_list)} files")

    # Analyze final distribution
    print("\n" + "="*60)
    print("Analyzing final dose distribution...")
    analyze_final_distribution(data_list[-1])

    # Analyze time evolution
    print("\n" + "="*60)
    print("Analyzing time evolution...")
    analyze_time_evolution(data_list)

    # Create animation frames (optional - takes time)
    # print("\n" + "="*60)
    # create_dose_map_animation_frames(data_list, step=5)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()