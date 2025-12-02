"""
Gpuff vs RCAP 비교 도구
- dispersion_results.txt (Gpuff)와 test.out (RCAP) 파일을 파싱
- 6가지 메트릭 비교: Center Air Conc, Ground Air Conc, Center Ground Conc, X/Q, Sigma-y, Sigma-z
- Test.inp에서 stability 값을 읽어 해당 이름으로 결과 파일 저장 (예: D.txt)
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
TEST_INP_FILE = r"X:\code\gpuffv4\gpuff-RCAPv3\input\RCAPdata\Test.inp"
OUTPUT_DIR = r"X:\code\gpuffv4\gpuff-RCAPv3\output"


# 고정 거리 값 (km) - 20개 구간 (RCAP test.out 출력 기준)
FIXED_DISTANCES = [0.10, 0.22, 0.35, 0.51, 0.70, 0.95, 1.28, 1.72, 2.31, 3.03,
                   3.92, 5.00, 6.36, 8.06, 10.00, 12.81, 16.09, 19.98, 24.57, 30.00]

# 메트릭 정의 (키, 파싱 패턴, 표시 이름, Y축 레이블)
METRICS = [
    ('center_air_conc', 'Center Air Conc.', 'Center Air Conc.', 'Bq-s/m³'),
    ('ground_air_conc', 'Ground Air Conc.', 'Ground Air Conc.', 'Bq-s/m³'),
    ('ground_conc', 'Center Ground Conc.', 'Center Ground Conc.', 'Bq/m²'),
    ('xq', 'Ground Dilution', 'X/Q', 's/m³'),
    ('sigma_y', 'Plume Sigma-y', 'Sigma-y', 'm'),
    ('sigma_z', 'Plume Sigma-z', 'Sigma-z', 'm'),
]


def parse_stability_from_test_inp(filepath):
    """Test.inp에서 RT350 라인의 stability 값 읽기 (두 번째 값)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == 'RT350':
                # RT350 뒤의 두 번째 값 (인덱스 2)
                if len(parts) >= 3:
                    stability = parts[2].upper()
                    if stability in ['A', 'B', 'C', 'D', 'E', 'F']:
                        return stability
                    else:
                        raise ValueError(f"Invalid stability value '{stability}' in RT350. Expected A-F.")
                else:
                    raise ValueError(f"RT350 line has insufficient values: {line.strip()}")
    raise ValueError(f"RT350 line not found in {filepath}")


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
    data = {'distances': FIXED_DISTANCES.copy()}
    for key, _, _, _ in METRICS:
        data[key] = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        for key, pattern, _, _ in METRICS:
            if pattern in line:
                values = parse_data_line(line)
                if values:
                    data[key] = values
                break

    return data


def parse_rcap_file(filepath):
    """RCAP test.out 파싱 - Mean Values for Each Plume Dispersion 섹션"""
    data = {'distances': FIXED_DISTANCES.copy()}
    for key, _, _, _ in METRICS:
        data[key] = []

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
            if data['center_air_conc']:
                break

        for key, pattern, _, _ in METRICS:
            if pattern in line:
                values = parse_data_line(line)
                if values:
                    data[key] = values
                break

    return data


def create_comparison_plots(gpuff_data, rcap_data, output_dir, stability):
    """6가지 메트릭 비교 그래프 생성 및 결과 파일 저장"""

    gpuff_dist = gpuff_data['distances']
    rcap_dist = rcap_data['distances']

    min_len = min(len(gpuff_dist), len(rcap_dist))
    if min_len == 0:
        print("Error: No distance data found!")
        return

    start_idx, end_idx = 1, min_len - 1
    dist = gpuff_dist[start_idx:end_idx]

    # 결과를 저장할 문자열 리스트
    output_lines = []

    # 개별 그래프 생성
    for key, pattern, title, ylabel in METRICS:
        gpuff_vals = gpuff_data[key][start_idx:end_idx] if len(gpuff_data[key]) > end_idx else gpuff_data[key]
        rcap_vals = rcap_data[key][start_idx:end_idx] if len(rcap_data[key]) > end_idx else rcap_data[key]

        if not gpuff_vals or not rcap_vals:
            print(f"Warning: {title} data not found")
            continue

        # 0이 아닌 값만 필터링 (log scale 때문)
        valid_indices = [i for i in range(min(len(gpuff_vals), len(rcap_vals), len(dist)))
                        if gpuff_vals[i] > 0 and rcap_vals[i] > 0]

        if not valid_indices:
            print(f"Warning: No valid (non-zero) data for {title}")
            continue

        valid_dist = [dist[i] for i in valid_indices]
        valid_gpuff = [gpuff_vals[i] for i in valid_indices]
        valid_rcap = [rcap_vals[i] for i in valid_indices]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.loglog(valid_dist, valid_rcap, '-', color='#e41a1c', label='RCAP', linewidth=2)
        ax.loglog(valid_dist, valid_gpuff, 'o', color='#377eb8', label='Gpuff', markersize=6)
        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel(f'{title} ({ylabel})', fontsize=12)
        ax.set_title(f'{title}: RCAP vs Gpuff', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'compare_{key}.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: compare_{key}.png")
        plt.close(fig)

    # 종합 그래프 (2x3 서브플롯)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (key, pattern, title, ylabel) in enumerate(METRICS):
        gpuff_vals = gpuff_data[key][start_idx:end_idx] if len(gpuff_data[key]) > end_idx else gpuff_data[key]
        rcap_vals = rcap_data[key][start_idx:end_idx] if len(rcap_data[key]) > end_idx else rcap_data[key]

        if not gpuff_vals or not rcap_vals:
            axes[idx].text(0.5, 0.5, f'No data for {title}', ha='center', va='center')
            axes[idx].set_title(title)
            continue

        valid_indices = [i for i in range(min(len(gpuff_vals), len(rcap_vals), len(dist)))
                        if gpuff_vals[i] > 0 and rcap_vals[i] > 0]

        if not valid_indices:
            axes[idx].text(0.5, 0.5, f'No valid data for {title}', ha='center', va='center')
            axes[idx].set_title(title)
            continue

        valid_dist = [dist[i] for i in valid_indices]
        valid_gpuff = [gpuff_vals[i] for i in valid_indices]
        valid_rcap = [rcap_vals[i] for i in valid_indices]

        axes[idx].loglog(valid_dist, valid_rcap, '-', color='#e41a1c', label='RCAP', linewidth=2)
        axes[idx].loglog(valid_dist, valid_gpuff, 'o', color='#377eb8', label='Gpuff', markersize=5)
        axes[idx].set_xlabel('Distance (km)')
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(title)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, which='both', alpha=0.3)

    plt.suptitle('RCAP vs Gpuff Comparison (All Metrics)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'compare_all.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: compare_all.png")
    plt.close(fig)

    # 수치 비교 출력 (터미널 + 파일)
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("                    RCAP vs Gpuff Comparison")
    output_lines.append("=" * 70)

    for key, pattern, title, ylabel in METRICS:
        gpuff_vals = gpuff_data[key][start_idx:end_idx] if len(gpuff_data[key]) > end_idx else gpuff_data[key]
        rcap_vals = rcap_data[key][start_idx:end_idx] if len(rcap_data[key]) > end_idx else rcap_data[key]

        if not gpuff_vals or not rcap_vals:
            continue

        output_lines.append(f"\n{title} ({ylabel})")
        output_lines.append(f"{'Distance (km)':<12} {'RCAP':<14} {'Gpuff':<14} {'Ratio':<10}")
        output_lines.append("-" * 50)
        for i in range(min(len(dist), len(gpuff_vals), len(rcap_vals))):
            if gpuff_vals[i] > 0:
                ratio = rcap_vals[i] / gpuff_vals[i]
                output_lines.append(f"{dist[i]:<12.2f} {rcap_vals[i]:<14.5e} {gpuff_vals[i]:<14.5e} {ratio:<10.2f}")
            else:
                output_lines.append(f"{dist[i]:<12.2f} {rcap_vals[i]:<14.5e} {gpuff_vals[i]:<14.5e} {'N/A':<10}")

    # 터미널 출력
    for line in output_lines:
        print(line)

    # 파일 저장 (stability 이름으로)
    output_file = os.path.join(output_dir, f"{stability}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved: {output_file}")


def main():
    print("="*50)
    print("  Gpuff vs RCAP Comparison Tool (6 Metrics)")
    print("="*50)

    # 파일 존재 확인
    if not os.path.exists(GPUFF_FILE):
        print(f"Error: Gpuff file not found: {GPUFF_FILE}")
        return
    if not os.path.exists(RCAP_FILE):
        print(f"Error: RCAP file not found: {RCAP_FILE}")
        return
    if not os.path.exists(TEST_INP_FILE):
        print(f"Error: Test.inp file not found: {TEST_INP_FILE}")
        return

    # Stability 값 읽기
    try:
        stability = parse_stability_from_test_inp(TEST_INP_FILE)
        print(f"\nStability class: {stability}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nGpuff file: {GPUFF_FILE}")
    print(f"RCAP file: {RCAP_FILE}")

    print("\nParsing Gpuff file...")
    gpuff_data = parse_gpuff_file(GPUFF_FILE)

    print("Parsing RCAP file...")
    rcap_data = parse_rcap_file(RCAP_FILE)

    print(f"\nGpuff data found:")
    for key, _, title, _ in METRICS:
        print(f"  - {title}: {len(gpuff_data[key])} values")

    print(f"\nRCAP data found:")
    for key, _, title, _ in METRICS:
        print(f"  - {title}: {len(rcap_data[key])} values")

    print("\nCreating comparison plots...")
    create_comparison_plots(gpuff_data, rcap_data, OUTPUT_DIR, stability)

    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
