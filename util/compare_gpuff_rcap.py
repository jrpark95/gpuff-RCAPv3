"""
Gpuff vs RCAP 비교 도구
- dispersion_results.txt (Gpuff)와 test.out (RCAP) 파일을 파싱
- Center Air Conc., Sigma-y, Sigma-z를 비교하는 그래프 생성
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 파일 경로 설정
GPUFF_FILE = r"X:\code\gpuffv4\gpuff-RCAPv3\output\dispersion_results.txt"
RCAP_FILE = r"C:\Users\user\Desktop\gpuff\RCAP교육\RCAP교육\RCAP_DoseProejction\test.out"
OUTPUT_DIR = r"X:\code\gpuffv4\gpuff-RCAPv3\util"


# 고정 거리 값 (km) - 20개 구간 (RCAP test.out 출력 기준)
# 주의: 이 거리는 RCAP test.out과 Gpuff dispersion_results.txt 모두에 해당하는 outer boundary 값
FIXED_DISTANCES = [0.10, 0.22, 0.35, 0.51, 0.70, 0.95, 1.28, 1.72, 2.31, 3.03, 
                   3.92, 5.00, 6.36, 8.06, 10.00, 12.81, 16.09, 19.98, 24.57, 30.00]


def parse_data_line(line):
    """데이터 라인에서 수치 값들을 파싱"""
    values = []
    # | 이후의 숫자들 추출
    parts = line.split('|')
    if len(parts) < 2:
        return values

    data_part = parts[1]
    # 과학적 표기법 숫자 추출
    matches = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+', data_part)
    for m in matches:
        values.append(float(m))

    return values


def parse_gpuff_file(filepath):
    """Gpuff dispersion_results.txt 파싱"""
    data = {
        'distances': FIXED_DISTANCES.copy(),
        'center_air_conc': [],
        'sigma_y': [],
        'sigma_z': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Center Air Conc. 파싱
        if 'Center Air Conc.' in line:
            values = parse_data_line(line)
            if values:
                data['center_air_conc'] = values

        # Plume Sigma-y 파싱
        if 'Plume Sigma-y' in line or 'Sigma-y' in line:
            values = parse_data_line(line)
            if values:
                data['sigma_y'] = values

        # Plume Sigma-z 파싱
        if 'Plume Sigma-z' in line or 'Sigma-z' in line:
            values = parse_data_line(line)
            if values:
                data['sigma_z'] = values

    return data


def parse_rcap_file(filepath):
    """RCAP test.out 파싱 - Mean Values for Each Plume Dispersion 섹션"""
    data = {
        'distances': FIXED_DISTANCES.copy(),
        'center_air_conc': [],
        'sigma_y': [],
        'sigma_z': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_mean_section = False

    for i, line in enumerate(lines):
        # Mean Values 섹션 찾기
        if 'Mean Values for Each Plume Dispersion' in line:
            in_mean_section = True
            continue

        if not in_mean_section:
            continue

        # 섹션 종료 감지
        if '======' in line or ('---' in line and 'Plume ID' not in lines[i-1] if i > 0 else False):
            if data['center_air_conc']:  # 이미 데이터를 파싱했으면 종료
                break

        # Center Air Conc. 파싱
        if 'Center Air Conc.' in line:
            values = parse_data_line(line)
            if values:
                data['center_air_conc'] = values

        # Plume Sigma-y 파싱
        if 'Plume Sigma-y' in line:
            values = parse_data_line(line)
            if values:
                data['sigma_y'] = values

        # Plume Sigma-z 파싱
        if 'Plume Sigma-z' in line:
            values = parse_data_line(line)
            if values:
                data['sigma_z'] = values

    return data


def create_comparison_plots(gpuff_data, rcap_data, output_dir):
    """세 가지 비교 그래프 생성"""

    # 거리 데이터 확인
    gpuff_dist = gpuff_data['distances']
    rcap_dist = rcap_data['distances']

    print(f"Gpuff distances: {len(gpuff_dist)} points")
    print(f"RCAP distances: {len(rcap_dist)} points")

    # 데이터 길이 맞추기 (최소 길이 사용)
    min_len = min(len(gpuff_dist), len(rcap_dist))
    if min_len == 0:
        print("Error: No distance data found!")
        return

    # 양끝값 제외 (첫번째와 마지막 제외)
    start_idx = 1
    end_idx = min_len - 1

    # 그래프 1: Center Air Concentration
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    gpuff_conc = gpuff_data['center_air_conc'][start_idx:end_idx]
    rcap_conc = rcap_data['center_air_conc'][start_idx:end_idx]
    dist = gpuff_dist[start_idx:end_idx]

    if gpuff_conc and rcap_conc:
        ax1.loglog(dist, rcap_conc, '-', color='#e41a1c', label='RCAP', linewidth=2)
        ax1.loglog(dist, gpuff_conc, 'o', color='#377eb8', label='Gpuff', markersize=6)
        ax1.set_xlabel('Distance (km)', fontsize=12)
        ax1.set_ylabel('Center Air Conc. (Bq·s/m³)', fontsize=12)
        ax1.set_title('Center Air Concentration: RCAP vs Gpuff', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'compare_center_air_conc.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: compare_center_air_conc.png")
    else:
        print("Warning: Center Air Conc data not found")
    plt.close(fig1)

    # 그래프 2: Sigma-y
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    gpuff_sy = gpuff_data['sigma_y'][start_idx:end_idx]
    rcap_sy = rcap_data['sigma_y'][start_idx:end_idx]

    if gpuff_sy and rcap_sy:
        ax2.loglog(dist, rcap_sy, '-', color='#e41a1c', label='RCAP', linewidth=2)
        ax2.loglog(dist, gpuff_sy, 'o', color='#377eb8', label='Gpuff', markersize=6)
        ax2.set_xlabel('Distance (km)', fontsize=12)
        ax2.set_ylabel('Sigma-y (m)', fontsize=12)
        ax2.set_title('Horizontal Dispersion (σy): RCAP vs Gpuff', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'compare_sigma_y.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: compare_sigma_y.png")
    else:
        print("Warning: Sigma-y data not found")
    plt.close(fig2)

    # 그래프 3: Sigma-z
    fig3, ax3 = plt.subplots(figsize=(10, 7))

    gpuff_sz = gpuff_data['sigma_z'][start_idx:end_idx]
    rcap_sz = rcap_data['sigma_z'][start_idx:end_idx]

    if gpuff_sz and rcap_sz:
        ax3.loglog(dist, rcap_sz, '-', color='#e41a1c', label='RCAP', linewidth=2)
        ax3.loglog(dist, gpuff_sz, 'o', color='#377eb8', label='Gpuff', markersize=6)
        ax3.set_xlabel('Distance (km)', fontsize=12)
        ax3.set_ylabel('Sigma-z (m)', fontsize=12)
        ax3.set_title('Vertical Dispersion (σz): RCAP vs Gpuff', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=11)
        ax3.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'compare_sigma_z.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: compare_sigma_z.png")
    else:
        print("Warning: Sigma-z data not found")
    plt.close(fig3)

    # 종합 그래프 (3개 서브플롯)
    fig4, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Center Air Conc
    if gpuff_conc and rcap_conc:
        axes[0].loglog(dist, rcap_conc, '-', color='#e41a1c', label='RCAP', linewidth=2)
        axes[0].loglog(dist, gpuff_conc, 'o', color='#377eb8', label='Gpuff', markersize=5)
        axes[0].set_xlabel('Distance (km)')
        axes[0].set_ylabel('Bq·s/m³')
        axes[0].set_title('Center Air Conc.')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, which='both', alpha=0.3)

    # Sigma-y
    if gpuff_sy and rcap_sy:
        axes[1].loglog(dist, rcap_sy, '-', color='#e41a1c', label='RCAP', linewidth=2)
        axes[1].loglog(dist, gpuff_sy, 'o', color='#377eb8', label='Gpuff', markersize=5)
        axes[1].set_xlabel('Distance (km)')
        axes[1].set_ylabel('m')
        axes[1].set_title('Sigma-y')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, which='both', alpha=0.3)

    # Sigma-z
    if gpuff_sz and rcap_sz:
        axes[2].loglog(dist, rcap_sz, '-', color='#e41a1c', label='RCAP', linewidth=2)
        axes[2].loglog(dist, gpuff_sz, 'o', color='#377eb8', label='Gpuff', markersize=5)
        axes[2].set_xlabel('Distance (km)')
        axes[2].set_ylabel('m')
        axes[2].set_title('Sigma-z')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, which='both', alpha=0.3)

    plt.suptitle('RCAP vs Gpuff Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'compare_all.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: compare_all.png")
    plt.close(fig4)

    # 수치 비교 출력
    print("\n" + "="*70)
    print("                    RCAP vs Gpuff Comparison")
    print("="*70)

    if gpuff_conc and rcap_conc:
        print(f"\n{'Distance (km)':<12} {'RCAP Conc':<14} {'Gpuff Conc':<14} {'Ratio':<10}")
        print("-"*50)
        for i in range(min(len(dist), len(gpuff_conc), len(rcap_conc))):
            ratio = rcap_conc[i] / gpuff_conc[i] if gpuff_conc[i] > 0 else 0
            print(f"{dist[i]:<12.2f} {rcap_conc[i]:<14.5e} {gpuff_conc[i]:<14.5e} {ratio:<10.2f}")

    if gpuff_sy and rcap_sy:
        print(f"\n{'Distance (km)':<12} {'RCAP σy':<14} {'Gpuff σy':<14} {'Ratio':<10}")
        print("-"*50)
        for i in range(min(len(dist), len(gpuff_sy), len(rcap_sy))):
            ratio = rcap_sy[i] / gpuff_sy[i] if gpuff_sy[i] > 0 else 0
            print(f"{dist[i]:<12.2f} {rcap_sy[i]:<14.2f} {gpuff_sy[i]:<14.2f} {ratio:<10.2f}")

    if gpuff_sz and rcap_sz:
        print(f"\n{'Distance (km)':<12} {'RCAP σz':<14} {'Gpuff σz':<14} {'Ratio':<10}")
        print("-"*50)
        for i in range(min(len(dist), len(gpuff_sz), len(rcap_sz))):
            ratio = rcap_sz[i] / gpuff_sz[i] if gpuff_sz[i] > 0 else 0
            print(f"{dist[i]:<12.2f} {rcap_sz[i]:<14.2f} {gpuff_sz[i]:<14.2f} {ratio:<10.2f}")


def main():
    print("="*50)
    print("  Gpuff vs RCAP Comparison Tool")
    print("="*50)

    # 파일 존재 확인
    if not os.path.exists(GPUFF_FILE):
        print(f"Error: Gpuff file not found: {GPUFF_FILE}")
        return
    if not os.path.exists(RCAP_FILE):
        print(f"Error: RCAP file not found: {RCAP_FILE}")
        return

    print(f"\nGpuff file: {GPUFF_FILE}")
    print(f"RCAP file: {RCAP_FILE}")

    # 파일 파싱
    print("\nParsing Gpuff file...")
    gpuff_data = parse_gpuff_file(GPUFF_FILE)

    print("Parsing RCAP file...")
    rcap_data = parse_rcap_file(RCAP_FILE)

    # 디버그 출력
    print(f"\nGpuff data found:")
    print(f"  - Distances: {len(gpuff_data['distances'])} points")
    print(f"  - Center Air Conc: {len(gpuff_data['center_air_conc'])} values")
    print(f"  - Sigma-y: {len(gpuff_data['sigma_y'])} values")
    print(f"  - Sigma-z: {len(gpuff_data['sigma_z'])} values")

    print(f"\nRCAP data found:")
    print(f"  - Distances: {len(rcap_data['distances'])} points")
    print(f"  - Center Air Conc: {len(rcap_data['center_air_conc'])} values")
    print(f"  - Sigma-y: {len(rcap_data['sigma_y'])} values")
    print(f"  - Sigma-z: {len(rcap_data['sigma_z'])} values")

    # 비교 그래프 생성
    print("\nCreating comparison plots...")
    create_comparison_plots(gpuff_data, rcap_data, OUTPUT_DIR)

    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
