"""
모든 대기안정도(A~F)를 비교 - 6가지 메트릭 전부
다양한 시각화 옵션 제공:
  Option 1: 기존 방식 (거리 vs 값, 점선=RCAP, 점=Gpuff)
  Option 2: 1:1 Scatter (X=RCAP, Y=Gpuff, y=x선 기준)
  Option 3: Ratio 플롯 (거리 vs Gpuff/RCAP, y=1 기준)
  Option 4: 안정도별 개별 서브플롯 (6개 패널)
  Option 5: 모든 옵션 출력
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

OUTPUT_DIR = r"X:\code\gpuffv4\gpuff-RCAPv3\output"
PNG_OUTPUT_DIR = r"X:\code\gpuffv4\gpuff-RCAPv3\output\png"

# 색상 정의 (A~F)
COLORS = {
    'A': '#0072B2',
    'B': '#E69F00',
    'C': '#009E73',
    'D': '#D55E00',
    'E': '#CC79A7',
    'F': '#56B4E9',
}

# 마커 정의 (A~F)
MARKERS = {
    'A': 'o',
    'B': 's',
    'C': '^',
    'D': 'D',
    'E': 'v',
    'F': 'p',
}

# 메트릭 정의
METRICS = [
    ('center_air_conc', 'Center Air Conc.', 'Bq-s/m³', 'Center Air Conc.'),
    ('ground_air_conc', 'Ground Air Conc.', 'Bq-s/m³', 'Ground Air Conc.'),
    ('ground_conc', 'Center Ground Conc.', 'Bq/m²', 'Center Ground Conc.'),
    ('xq', 'X/Q', 's/m³', 'X/Q'),
    ('sigma_y', 'Sigma-y', 'm', 'Sigma-y'),
    ('sigma_z', 'Sigma-z', 'm', 'Sigma-z'),
]


def parse_output_file(filepath):
    """A.txt~F.txt 파일 파싱 - 6가지 메트릭 전부"""
    data = {}
    for key, _, _, _ in METRICS:
        data[key] = {'distances': [], 'rcap': [], 'gpuff': []}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_metric = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('='):
            continue

        for key, _, _, header in METRICS:
            if header in line_stripped and 'Distance' not in line_stripped and 'RCAP' not in line_stripped:
                current_metric = key
                break

        if line_stripped.startswith('-'):
            continue
        if 'Distance' in line_stripped and 'RCAP' in line_stripped:
            continue

        if current_metric:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    dist = float(parts[0])
                    rcap_val = float(parts[1])
                    gpuff_val = float(parts[2])
                    data[current_metric]['distances'].append(dist)
                    data[current_metric]['rcap'].append(rcap_val)
                    data[current_metric]['gpuff'].append(gpuff_val)
                except ValueError:
                    continue

    return data


def load_all_data():
    """모든 안정도 데이터 로드"""
    all_data = {}
    for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
        filepath = os.path.join(OUTPUT_DIR, f"{stab}.txt")
        if os.path.exists(filepath):
            all_data[stab] = parse_output_file(filepath)
            print(f"Loaded: {stab}.txt")
        else:
            print(f"Not found: {filepath}")
    return all_data


def option1_original(all_data, output_subdir):
    """Option 1: 기존 방식 (거리 vs 값)"""
    print("\n[Option 1] Original style (Distance vs Value)")
    os.makedirs(output_subdir, exist_ok=True)

    for key, title, ylabel, _ in METRICS:
        fig, ax = plt.subplots(figsize=(14, 10))

        # 데이터 준비 (유효 데이터 필터링)
        plot_data = {}
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            if not metric_data['rcap'] or not metric_data['gpuff']:
                continue

            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue

            plot_data[stab] = {
                'dist': [metric_data['distances'][i] for i in valid_idx],
                'rcap': [metric_data['rcap'][i] for i in valid_idx],
                'gpuff': [metric_data['gpuff'][i] for i in valid_idx]
            }

        # RCAP 먼저 모두 플롯 (범례 왼쪽 열)
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in plot_data:
                continue
            ax.loglog(plot_data[stab]['dist'], plot_data[stab]['rcap'], '--',
                     color=COLORS[stab], linewidth=2.5, label=f'RCAP {stab}')

        # Gpuff 다음에 모두 플롯 (범례 오른쪽 열)
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in plot_data:
                continue
            ax.loglog(plot_data[stab]['dist'], plot_data[stab]['gpuff'], MARKERS[stab],
                     color=COLORS[stab], markersize=8, markerfacecolor='white',
                     markeredgewidth=2, label=f'Gpuff {stab}')

        ax.set_xlabel('Distance (km)', fontsize=22)
        ax.set_ylabel(f'{title} ({ylabel})', fontsize=22)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=14, ncol=2, loc='best')

        plt.tight_layout()
        fig.savefig(os.path.join(output_subdir, f'opt1_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved to: {output_subdir}")


def option2_scatter(all_data, output_subdir):
    """Option 2: 1:1 Scatter Plot (RCAP vs Gpuff)"""
    print("\n[Option 2] 1:1 Scatter (RCAP vs Gpuff)")
    os.makedirs(output_subdir, exist_ok=True)

    for key, title, ylabel, _ in METRICS:
        fig, ax = plt.subplots(figsize=(12, 12))

        all_rcap = []
        all_gpuff = []

        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            if not metric_data['rcap'] or not metric_data['gpuff']:
                continue

            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue

            rcap = [metric_data['rcap'][i] for i in valid_idx]
            gpuff = [metric_data['gpuff'][i] for i in valid_idx]
            all_rcap.extend(rcap)
            all_gpuff.extend(gpuff)

            ax.scatter(rcap, gpuff, c=COLORS[stab], marker=MARKERS[stab], s=100,
                      label=stab, alpha=0.8, edgecolors='black', linewidth=0.5)

        # y=x 기준선
        if all_rcap and all_gpuff:
            min_val = min(min(all_rcap), min(all_gpuff)) * 0.5
            max_val = max(max(all_rcap), max(all_gpuff)) * 2
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y = x')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'RCAP {title} ({ylabel})', fontsize=22)
        ax.set_ylabel(f'Gpuff {title} ({ylabel})', fontsize=22)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=16, loc='upper left')
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        fig.savefig(os.path.join(output_subdir, f'opt2_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved to: {output_subdir}")


def option3_ratio(all_data, output_subdir):
    """Option 3: Ratio Plot (Distance vs Gpuff/RCAP)"""
    print("\n[Option 3] Ratio Plot (Gpuff/RCAP)")
    os.makedirs(output_subdir, exist_ok=True)

    for key, title, ylabel, _ in METRICS:
        fig, ax = plt.subplots(figsize=(14, 10))

        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            if not metric_data['rcap'] or not metric_data['gpuff']:
                continue

            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue

            dist = [metric_data['distances'][i] for i in valid_idx]
            ratio = [metric_data['gpuff'][i] / metric_data['rcap'][i] for i in valid_idx]

            ax.semilogx(dist, ratio, '-', color=COLORS[stab], linewidth=2.5,
                       marker=MARKERS[stab], markersize=10, markerfacecolor='white',
                       markeredgewidth=2, label=stab)

        # y=1 기준선 (완벽 일치)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Perfect Match')

        ax.set_xlabel('Distance (km)', fontsize=22)
        ax.set_ylabel(f'Gpuff / RCAP Ratio', fontsize=22)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=16, loc='best')

        # Y축 범위 설정 (0.5 ~ 2.0 또는 데이터에 맞게)
        ax.set_ylim(0.4, 1.6)

        plt.tight_layout()
        fig.savefig(os.path.join(output_subdir, f'opt3_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved to: {output_subdir}")


def option4_subplots(all_data, output_subdir):
    """Option 4: 안정도별 개별 서브플롯 (2x3 패널)"""
    print("\n[Option 4] Stability-wise Subplots (2x3)")
    os.makedirs(output_subdir, exist_ok=True)

    for key, title, ylabel, _ in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()

        for idx, stab in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
            ax = axes[idx]

            if stab not in all_data:
                ax.text(0.5, 0.5, f'No data for {stab}', ha='center', va='center', fontsize=18)
                ax.set_title(f'Stability {stab}', fontsize=20, fontweight='bold')
                continue

            metric_data = all_data[stab][key]
            if not metric_data['rcap'] or not metric_data['gpuff']:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', fontsize=18)
                ax.set_title(f'Stability {stab}', fontsize=20, fontweight='bold')
                continue

            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]

            if not valid_idx:
                ax.text(0.5, 0.5, f'No valid data', ha='center', va='center', fontsize=18)
                ax.set_title(f'Stability {stab}', fontsize=20, fontweight='bold')
                continue

            dist = [metric_data['distances'][i] for i in valid_idx]
            rcap = [metric_data['rcap'][i] for i in valid_idx]
            gpuff = [metric_data['gpuff'][i] for i in valid_idx]

            ax.loglog(dist, rcap, '-', color='#e41a1c', linewidth=3, label='RCAP')
            ax.loglog(dist, gpuff, 'o', color='#377eb8', markersize=10,
                     markerfacecolor='white', markeredgewidth=2.5, label='Gpuff')

            ax.set_xlabel('Distance (km)', fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(f'Stability {stab}', fontsize=20, fontweight='bold', color=COLORS[stab])
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=14, loc='best')

        plt.suptitle(f'{title}', fontsize=24, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(output_subdir, f'opt4_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  Saved to: {output_subdir}")


def create_combined_summary(all_data, output_subdir):
    """각 옵션별 6메트릭 종합 그래프"""
    print("\n[Summary] Combined 6-metric figures for each option")
    os.makedirs(output_subdir, exist_ok=True)

    # Option 1 종합
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for idx, (key, title, ylabel, _) in enumerate(METRICS):
        ax = axes[idx]
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue
            dist = [metric_data['distances'][i] for i in valid_idx]
            rcap = [metric_data['rcap'][i] for i in valid_idx]
            gpuff = [metric_data['gpuff'][i] for i in valid_idx]
            ax.loglog(dist, rcap, '--', color=COLORS[stab], linewidth=2)
            ax.loglog(dist, gpuff, MARKERS[stab], color=COLORS[stab], markersize=6,
                     markerfacecolor='white', markeredgewidth=1.5)
        ax.set_xlabel('Distance (km)', fontsize=16)
        ax.set_ylabel(f'{title} ({ylabel})', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_subdir, 'summary_opt1.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Option 2 종합 (Scatter)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for idx, (key, title, ylabel, _) in enumerate(METRICS):
        ax = axes[idx]
        all_vals = []
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue
            rcap = [metric_data['rcap'][i] for i in valid_idx]
            gpuff = [metric_data['gpuff'][i] for i in valid_idx]
            all_vals.extend(rcap + gpuff)
            ax.scatter(rcap, gpuff, c=COLORS[stab], marker=MARKERS[stab], s=60, alpha=0.8)
        if all_vals:
            min_v, max_v = min(all_vals)*0.5, max(all_vals)*2
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5)
            ax.set_xlim(min_v, max_v)
            ax.set_ylim(min_v, max_v)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'RCAP', fontsize=14)
        ax.set_ylabel(f'Gpuff', fontsize=14)
        ax.set_title(f'{title}', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(os.path.join(output_subdir, 'summary_opt2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Option 3 종합 (Ratio)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    for idx, (key, title, ylabel, _) in enumerate(METRICS):
        ax = axes[idx]
        for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
            if stab not in all_data:
                continue
            metric_data = all_data[stab][key]
            valid_idx = [i for i in range(len(metric_data['rcap']))
                        if metric_data['rcap'][i] > 0 and metric_data['gpuff'][i] > 0]
            if not valid_idx:
                continue
            dist = [metric_data['distances'][i] for i in valid_idx]
            ratio = [metric_data['gpuff'][i] / metric_data['rcap'][i] for i in valid_idx]
            ax.semilogx(dist, ratio, '-', color=COLORS[stab], linewidth=2,
                       marker=MARKERS[stab], markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Distance (km)', fontsize=14)
        ax.set_ylabel('Gpuff/RCAP', fontsize=14)
        ax.set_title(f'{title}', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(0.4, 1.6)
    plt.tight_layout()
    fig.savefig(os.path.join(output_subdir, 'summary_opt3.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved summaries to: {output_subdir}")


def main():
    print("=" * 70)
    print("  RCAP vs Gpuff: 6 Metrics × All Stability Classes (A-F)")
    print("  Visualization Options: 1=Original, 2=Scatter, 3=Ratio, 4=Subplots, 5=All")
    print("=" * 70)

    # 옵션 선택 (커맨드라인 또는 기본값)
    if len(sys.argv) > 1:
        option = int(sys.argv[1])
    else:
        option = 5  # 기본값: 모든 옵션 출력

    all_data = load_all_data()
    if not all_data:
        print("No data files found!")
        return

    os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)

    if option == 1 or option == 5:
        option1_original(all_data, os.path.join(PNG_OUTPUT_DIR, 'opt1_original'))

    if option == 2 or option == 5:
        option2_scatter(all_data, os.path.join(PNG_OUTPUT_DIR, 'opt2_scatter'))

    if option == 3 or option == 5:
        option3_ratio(all_data, os.path.join(PNG_OUTPUT_DIR, 'opt3_ratio'))

    if option == 4 or option == 5:
        option4_subplots(all_data, os.path.join(PNG_OUTPUT_DIR, 'opt4_subplots'))

    if option == 5:
        create_combined_summary(all_data, os.path.join(PNG_OUTPUT_DIR, 'summary'))

    print(f"\n{'='*70}")
    print(f"All outputs saved to: {PNG_OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
