"""
모든 대기안정도(A~F)를 한 그래프에 비교
- RCAP: 선 (기준)
- Gpuff: 점 (검증 대상)
- 같은 안정도는 같은 색
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = r"X:\code\gpuffv4\gpuff-RCAPv3\util"

# 색상 정의 (A~F) - 색맹 친화적 팔레트
COLORS = {
    'A': '#0072B2',
    'B': '#E69F00',
    'C': '#009E73',
    'D': '#D55E00',
    'E': '#CC79A7',
    'F': '#F0E442',
}


def parse_output_file(filepath):
    """A.txt~F.txt 파일 파싱"""
    data = {
        'distances': [],
        'rcap_conc': [], 'gpuff_conc': [],
        'rcap_sy': [], 'gpuff_sy': [],
        'rcap_sz': [], 'gpuff_sz': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_section = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-'):
            continue

        if 'RCAP Conc' in line and 'Gpuff Conc' in line:
            current_section = 'conc'
            continue
        elif 'RCAP' in line and 'Gpuff' in line and 'σy' in line:
            current_section = 'sy'
            continue
        elif 'RCAP' in line and 'Gpuff' in line and 'σz' in line:
            current_section = 'sz'
            continue
        elif 'Distance' in line:
            continue

        # 데이터 라인 파싱
        parts = line.split()
        if len(parts) >= 4 and current_section:
            try:
                dist = float(parts[0])
                val1 = float(parts[1])
                val2 = float(parts[2])

                if current_section == 'conc':
                    if dist not in data['distances']:
                        data['distances'].append(dist)
                    data['rcap_conc'].append(val1)
                    data['gpuff_conc'].append(val2)
                elif current_section == 'sy':
                    data['rcap_sy'].append(val1)
                    data['gpuff_sy'].append(val2)
                elif current_section == 'sz':
                    data['rcap_sz'].append(val1)
                    data['gpuff_sz'].append(val2)
            except ValueError:
                continue

    return data


def create_combined_plots():
    """모든 안정도를 한 그래프에 표시"""

    # 데이터 로드
    all_data = {}
    for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
        filepath = os.path.join(r"X:\code\gpuffv4\gpuff-RCAPv3\output", f"{stab}.txt")
        if os.path.exists(filepath):
            all_data[stab] = parse_output_file(filepath)
            print(f"Loaded: {stab}.txt")
        else:
            print(f"Not found: {filepath}")

    if not all_data:
        print("No data files found!")
        return

    # 그래프 1: Center Air Concentration
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for stab, data in all_data.items():
        if data['rcap_conc'] and data['gpuff_conc']:
            dist = data['distances'][:len(data['rcap_conc'])]
            ax1.loglog(dist, data['rcap_conc'], '--', color=COLORS[stab], linewidth=2.5, label=f'RCAP {stab}')
            ax1.loglog(dist, data['gpuff_conc'], 'o', color=COLORS[stab], markersize=7)

    ax1.set_xlabel('Distance (km)', fontsize=14)
    ax1.set_ylabel('Center Air Conc. (Bq·s/m³)', fontsize=14)
    ax1.set_title('Center Air Concentration: All Stability Classes', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, which='both', alpha=0.3)

    # Gpuff 점 범례 추가
    ax1.plot([], [], 'ko', markersize=7, label='Gpuff (points)')
    ax1.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'compare_all_conc.png'), dpi=150, bbox_inches='tight')
    print("Saved: compare_all_conc.png")
    plt.close(fig1)

    # 그래프 2: Sigma-y
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for stab, data in all_data.items():
        if data['rcap_sy'] and data['gpuff_sy']:
            dist = data['distances'][:len(data['rcap_sy'])]
            ax2.loglog(dist, data['rcap_sy'], '--', color=COLORS[stab], linewidth=2.5, label=f'RCAP {stab}')
            ax2.loglog(dist, data['gpuff_sy'], 'o', color=COLORS[stab], markersize=7)

    ax2.set_xlabel('Distance (km)', fontsize=14)
    ax2.set_ylabel('Sigma-y (m)', fontsize=14)
    ax2.set_title('Horizontal Dispersion (σy): All Stability Classes', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.plot([], [], 'ko', markersize=7, label='Gpuff (points)')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'compare_all_sigma_y.png'), dpi=150, bbox_inches='tight')
    print("Saved: compare_all_sigma_y.png")
    plt.close(fig2)

    # 그래프 3: Sigma-z
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for stab, data in all_data.items():
        if data['rcap_sz'] and data['gpuff_sz']:
            dist = data['distances'][:len(data['rcap_sz'])]
            ax3.loglog(dist, data['rcap_sz'], '--', color=COLORS[stab], linewidth=2.5, label=f'RCAP {stab}')
            ax3.loglog(dist, data['gpuff_sz'], 'o', color=COLORS[stab], markersize=7)

    ax3.set_xlabel('Distance (km)', fontsize=14)
    ax3.set_ylabel('Sigma-z (m)', fontsize=14)
    ax3.set_title('Vertical Dispersion (σz): All Stability Classes', fontsize=16, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=12)
    ax3.plot([], [], 'ko', markersize=7, label='Gpuff (points)')
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'compare_all_sigma_z.png'), dpi=150, bbox_inches='tight')
    print("Saved: compare_all_sigma_z.png")
    plt.close(fig3)

    # 종합 그래프 (3개 서브플롯)
    fig4, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Conc (범례는 여기에만)
    for stab, data in all_data.items():
        if data['rcap_conc'] and data['gpuff_conc']:
            dist = data['distances'][:len(data['rcap_conc'])]
            axes[0].loglog(dist, data['rcap_conc'], '--', color=COLORS[stab], linewidth=2, label=f'{stab}')
            axes[0].loglog(dist, data['gpuff_conc'], 'o', color=COLORS[stab], markersize=5)
    axes[0].plot([], [], 'ko', markersize=5, label='Gpuff')
    axes[0].plot([], [], 'k--', linewidth=2, label='RCAP')
    axes[0].set_xlabel('Distance (km)', fontsize=13)
    axes[0].set_ylabel('Bq·s/m³', fontsize=13)
    axes[0].set_title('Center Air Conc.', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=11)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, which='both', alpha=0.3)

    # Sigma-y
    for stab, data in all_data.items():
        if data['rcap_sy'] and data['gpuff_sy']:
            dist = data['distances'][:len(data['rcap_sy'])]
            axes[1].loglog(dist, data['rcap_sy'], '--', color=COLORS[stab], linewidth=2)
            axes[1].loglog(dist, data['gpuff_sy'], 'o', color=COLORS[stab], markersize=5)
    axes[1].set_xlabel('Distance (km)', fontsize=13)
    axes[1].set_ylabel('m', fontsize=13)
    axes[1].set_title('Sigma-y', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=11)
    axes[1].grid(True, which='both', alpha=0.3)

    # Sigma-z
    for stab, data in all_data.items():
        if data['rcap_sz'] and data['gpuff_sz']:
            dist = data['distances'][:len(data['rcap_sz'])]
            axes[2].loglog(dist, data['rcap_sz'], '--', color=COLORS[stab], linewidth=2)
            axes[2].loglog(dist, data['gpuff_sz'], 'o', color=COLORS[stab], markersize=5)
    axes[2].set_xlabel('Distance (km)', fontsize=13)
    axes[2].set_ylabel('m', fontsize=13)
    axes[2].set_title('Sigma-z', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='both', labelsize=11)
    axes[2].grid(True, which='both', alpha=0.3)

    plt.suptitle('RCAP (dashed) vs Gpuff (points): All Stability Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, 'compare_all_combined.png'), dpi=150, bbox_inches='tight')
    print("Saved: compare_all_combined.png")
    plt.close(fig4)


if __name__ == "__main__":
    print("=" * 50)
    print("  All Stability Classes Comparison")
    print("=" * 50)
    create_combined_plots()
    print(f"\nOutput saved to: {OUTPUT_DIR}")
