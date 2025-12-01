import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 거리 데이터 (km) - 첫 번째 값(0.1km) 제외, 10km까지만
distances = [0.22, 0.35, 0.51, 0.7, 0.95, 1.28, 1.72, 2.31, 3.03,
             3.92, 5, 6.36, 8.06, 10]

# RCAP 데이터 (Bq*s/m3) - 안정도 C
rcap_C = [1.73577e-03, 7.13596e-04, 3.72259e-04, 1.97252e-04, 1.12763e-04,
          7.36092e-05, 5.22127e-05, 3.92034e-05, 3.06783e-05, 2.47727e-05,
          2.05025e-05, 1.73078e-05, 1.48505e-05, 1.29165e-05]

# Gpuff 데이터 (Bq*s/m3) - 안정도 C
gpuff_C = [9.07445e-04, 4.27622e-04, 2.26490e-04, 1.31142e-04, 7.70155e-05,
           4.57741e-05, 2.74331e-05, 1.65820e-05, 1.05387e-05, 6.93206e-06,
           4.73657e-06, 3.34697e-06, 2.50085e-06, 2.01755e-06]

# 그래프 생성
fig, ax = plt.subplots(figsize=(10, 7))

# RCAP 데이터 (실선, 원 마커)
ax.loglog(distances, rcap_C, 'o-', color='#4daf4a', label='RCAP (Briggs)',
          linewidth=2, markersize=7)

# Gpuff 데이터 (점선, 사각 마커)
ax.loglog(distances, gpuff_C, 's--', color='#4daf4a', label='Gpuff (Briggs)',
          linewidth=2, markersize=7, alpha=0.7)

ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('TIC (Bq·s/m³)', fontsize=12)
ax.set_title('RCAP vs Gpuff: Stability Class C (Briggs-McElroy-Pooler)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim([0.15, 12])

# 비율 표시 (우측 y축)
ax2 = ax.twinx()
ratios = [r/g if g > 0 else 0 for r, g in zip(rcap_C, gpuff_C)]
ax2.plot(distances, ratios, 'x-', color='red', alpha=0.5, label='RCAP/Gpuff ratio')
ax2.set_ylabel('RCAP/Gpuff Ratio', fontsize=11, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
ax2.set_ylim([0, 10])

plt.tight_layout()
plt.savefig('X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_C_only.png', dpi=150, bbox_inches='tight')
print("Plot saved to: X:/code/gpuffv4/gpuff-RCAPv3/util/comparison_C_only.png")

# 수치 비교 출력
print("\n=== Stability Class C: RCAP vs Gpuff ===")
print(f"{'Distance (km)':<15} {'RCAP':<15} {'Gpuff':<15} {'RCAP/Gpuff':<12}")
print("-" * 57)
for d, r, g in zip(distances, rcap_C, gpuff_C):
    ratio = r/g if g > 0 else 0
    print(f"{d:<15.2f} {r:<15.5e} {g:<15.5e} {ratio:<12.2f}")
