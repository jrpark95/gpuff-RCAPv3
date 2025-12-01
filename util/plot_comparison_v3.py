import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 거리 데이터 (km) - 첫 번째 값(0.1km) 제외, 10km까지만 (14개)
distances = [0.22, 0.35, 0.51, 0.7, 0.95, 1.28, 1.72, 2.31, 3.03,
             3.92, 5, 6.36, 8.06, 10]

# RCAP 데이터 (Bq*s/m3) - 첫 번째 값 제외, 10km까지만 (B, C, D, E만)
rcap = {
    'B': [0.00108, 4.42E-04, 2.20E-04, 1.20E-04, 6.87E-05, 3.95E-05, 2.28E-05, 1.32E-05, 7.80E-06, 4.80E-06, 3.05E-06, 2.03E-06, 1.48E-06, 1.19E-06],
    'C': [0.0019, 8.54E-04, 4.52E-04, 2.58E-04, 1.53E-04, 9.17E-05, 5.52E-05, 3.34E-05, 2.09E-05, 1.35E-05, 9.08E-06, 6.25E-06, 4.37E-06, 3.15E-06],
    'D': [0.00312, 0.00161, 9.40E-04, 5.85E-04, 3.75E-04, 2.42E-04, 1.57E-04, 1.02E-04, 6.83E-05, 4.71E-05, 3.34E-05, 2.40E-05, 1.75E-05, 1.31E-05],
    'E': [0.00471, 0.0027, 0.00168, 0.00109, 7.15E-04, 4.71E-04, 3.12E-04, 2.07E-04, 1.42E-04, 1.01E-04, 7.40E-05, 5.57E-05, 4.28E-05, 3.39E-05]
}

# Gpuff 데이터 (Bq*s/m3) - Briggs-McElroy-Pooler 모델 적용
gpuff = {
    'B': [7.79E-04, 3.10E-04, 1.47E-04, 7.87E-05, 4.33E-05, 2.42E-05, 1.37E-05, 7.78E-06, 4.85E-06, 3.46E-06, 2.75E-06, 2.26E-06, 1.87E-06, 1.59E-06],
    'C': [1.74E-03, 6.99E-04, 3.37E-04, 1.83E-04, 1.03E-04, 5.91E-05, 3.45E-05, 2.05E-05, 1.28E-05, 8.35E-06, 5.66E-06, 3.98E-06, 2.96E-06, 2.38E-06],
    'D': [3.59E-03, 1.53E-03, 7.81E-04, 4.51E-04, 2.69E-04, 1.65E-04, 1.03E-04, 6.56E-05, 4.37E-05, 3.00E-05, 2.13E-05, 1.53E-05, 1.12E-05, 8.44E-06],
    'E': [8.86E-03, 3.65E-03, 1.81E-03, 1.02E-03, 5.93E-04, 3.57E-04, 2.21E-04, 1.40E-04, 9.44E-05, 6.65E-05, 4.87E-05, 3.66E-05, 2.81E-05, 2.25E-05]
}

# 안정도 등급별 색상
colors = {'B': '#377eb8', 'C': '#4daf4a', 'D': '#984ea3', 'E': '#ff7f00'}

stability_classes = ['B', 'C', 'D', 'E']

# ========== 그래프 1: 2x2 서브플롯 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, sc in enumerate(stability_classes):
    ax = axes[idx]

    # RCAP 데이터 (실선)
    ax.loglog(distances, rcap[sc], 'o-', color=colors[sc], label='RCAP', linewidth=2, markersize=5)

    # Gpuff 데이터 (점선)
    ax.loglog(distances, gpuff[sc], 's--', color=colors[sc], label='Gpuff (Briggs)', linewidth=2, markersize=5, alpha=0.7)

    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('TIC (Bq·s/m³)', fontsize=11)
    ax.set_title(f'Stability Class {sc}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0.15, 12])

plt.suptitle('RCAP vs Gpuff: Briggs-McElroy-Pooler Model', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_plot_v3.png', dpi=150, bbox_inches='tight')
print("Plot 1 saved to: X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_plot_v3.png")

# ========== 그래프 2: 전체 한 그래프 ==========
fig2, ax2 = plt.subplots(figsize=(12, 8))

for sc in stability_classes:
    # RCAP 데이터 (실선, 원 마커)
    ax2.loglog(distances, rcap[sc], 'o-', color=colors[sc], label=f'RCAP-{sc}', linewidth=2, markersize=5)

    # Gpuff 데이터 (점선, 사각 마커)
    ax2.loglog(distances, gpuff[sc], 's--', color=colors[sc], label=f'Gpuff-{sc}', linewidth=2, markersize=5, alpha=0.6)

ax2.set_xlabel('Distance (km)', fontsize=12)
ax2.set_ylabel('TIC (Bq·s/m³)', fontsize=12)
ax2.set_title('RCAP vs Gpuff: Briggs-McElroy-Pooler Model', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9, ncol=2)
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlim([0.15, 12])

plt.tight_layout()
plt.savefig('X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_plot_all_v3.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved to: X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_plot_all_v3.png")
