"""
Gpuff vs RCAP 비교 도구
- dispersion_results.txt (Gpuff)와 test.out (RCAP) 파일을 파싱
- Ground Air Conc. 비교하는 그래프 생성
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
FIXED_DISTANCES = [0.10, 0.22, 0.35, 0.51, 0.70, 0.95, 1.28, 1.72, 2.31, 3.03,
                   3.92, 5.00, 6.36, 8.06, 10.00, 12.81, 16.09, 19.98, 24.57, 30.00]


def parse_data_line(line):
    """데이터 라인에서 수치 값들을 파싱"""
    values = []
    parts = line.split('|')
    if len(parts) < 2:
        return values
    data_part = parts[1]
    matches = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+', data_part)
    for m in matches:
        values.append(float(m))
    return values


def parse_gpuff_file(filepath):
    """Gpuff dispersion_results.txt 파싱"""
    data = {'distances': FIXED_DISTANCES.copy(), 'ground_air_conc': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if 'Ground Air Conc.' in line:
            values = parse_data_line(line)
            if values:
                data['ground_air_conc'] = values
    return data


def parse_rcap_file(filepath):
    """RCAP test.out 파싱"""
    data = {'distances': FIXED_DISTANCES.copy(), 'ground_air_conc': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    in_mean_section = False
    for i, line in enumerate(lines):
        if 'Mean Values for Each Plume Dispersion' in line:
            in_mean_section = True
            continue
        if not in_mean_section:
            continue
        if '======' in line or ('---' in line and 'Plume ID' not in lines[i-1] if i > 0 else False):
            if data['ground_air_conc']:
                break
        if 'Ground Air Conc.' in line:
            values = parse_data_line(line)
            if values:
                data['ground_air_conc'] = values
    return data


def create_comparison_plots(gpuff_data, rcap_data, output_dir):
    """Ground Air Conc 비교 그래프 생성"""
    gpuff_dist = gpuff_data['distances']
    rcap_dist = rcap_data['distances']
    min_len = min(len(gpuff_dist), len(rcap_dist))
    if min_len == 0:
        print("Error: No distance data found!")
        return
    start_idx, end_idx = 1, min_len - 1
    
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    gpuff_conc = gpuff_data['ground_air_conc'][start_idx:end_idx]
    rcap_conc = rcap_data['ground_air_conc'][start_idx:end_idx]
    dist = gpuff_dist[start_idx:end_idx]
    
    valid_indices = [i for i in range(len(gpuff_conc)) if gpuff_conc[i] > 0 and rcap_conc[i] > 0]
    if valid_indices:
        valid_dist = [dist[i] for i in valid_indices]
        valid_gpuff = [gpuff_conc[i] for i in valid_indices]
        valid_rcap = [rcap_conc[i] for i in valid_indices]
        ax1.loglog(valid_dist, valid_rcap, '-', color='#e41a1c', label='RCAP', linewidth=2)
        ax1.loglog(valid_dist, valid_gpuff, 'o', color='#377eb8', label='Gpuff', markersize=6)
        ax1.set_xlabel('Distance (km)', fontsize=12)
        ax1.set_ylabel('Ground Air Conc. (Bq-s/m³)', fontsize=12)
        ax1.set_title('Ground Air Concentration: RCAP vs Gpuff', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'compare_ground_air_conc.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: compare_ground_air_conc.png")
    plt.close(fig1)
    
    print("\n" + "="*70)
    print("                    RCAP vs Gpuff Comparison (Ground Air Conc)")
    print("="*70)
    if gpuff_conc and rcap_conc:
        print(f"\n{'Distance (km)':<12} {'RCAP Conc':<14} {'Gpuff Conc':<14} {'Ratio':<10}")
        print("-"*50)
        for i in range(min(len(dist), len(gpuff_conc), len(rcap_conc))):
            ratio = rcap_conc[i] / gpuff_conc[i] if gpuff_conc[i] > 0 else 0
            print(f"{dist[i]:<12.2f} {rcap_conc[i]:<14.5e} {gpuff_conc[i]:<14.5e} {ratio:<10.2f}")


def main():
    print("="*50)
    print("  Gpuff vs RCAP Comparison Tool (Ground Air Conc)")
    print("="*50)
    if not os.path.exists(GPUFF_FILE):
        print(f"Error: Gpuff file not found: {GPUFF_FILE}")
        return
    if not os.path.exists(RCAP_FILE):
        print(f"Error: RCAP file not found: {RCAP_FILE}")
        return
    print(f"\nGpuff file: {GPUFF_FILE}")
    print(f"RCAP file: {RCAP_FILE}")
    gpuff_data = parse_gpuff_file(GPUFF_FILE)
    rcap_data = parse_rcap_file(RCAP_FILE)
    print(f"\nGpuff Ground Air Conc: {len(gpuff_data['ground_air_conc'])} values")
    print(f"RCAP Ground Air Conc: {len(rcap_data['ground_air_conc'])} values")
    create_comparison_plots(gpuff_data, rcap_data, OUTPUT_DIR)
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
