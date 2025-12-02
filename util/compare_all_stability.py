"""
모든 대기안정도(A~F)를 한 그래프에 비교 (Ground Air Conc)
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
    """A.txt~F.txt 파일 파싱 (Ground Air Conc 전용)"""
    data = {
        'distances': [],
        'rcap_conc': [], 'gpuff_conc': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_data_section = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith('='):
            continue

        # 헤더 라인 감지
        if 'RCAP Conc' in line and 'Gpuff Conc' in line:
            in_data_section = True
            continue
        if line.startswith('-'):
            continue
        if 'Distance' in line:
            continue

        # 데이터 라인 파싱
        if in_data_section:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    dist = float(parts[0])
                    val1 = float(parts[1])
                    val2 = float(parts[2])

                    data['distances'].append(dist)
                    data['rcap_conc'].append(val1)
                    data['gpuff_conc'].append(val2)
                except ValueError:
                    continue

    return data


def create_combined_plots():
    """모든 안정도를 한 그래프에 표시 (Ground Air Conc)"""

    # 데이터 로드
    all_data = {}
    for stab in ['A', 'B', 'C', 'D', 'E', 'F']:
        filepath = os.path.join(r"X:\code\gpuffv4\gpuff-RCAPv3\output", f"{stab}.txt")
        if os.path.exists(filepath):
            all_data[stab] = parse_output_file(filepath)
            print(f"Loaded: {stab}.txt - {len(all_data[stab]['distances'])} points")
        else:
            print(f"Not found: {filepath}")

    if not all_data:
        print("No data files found!")
        return

    # Ground Air Concentration 그래프
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    rcap_handles = []
    gpuff_handles = []

    for stab, data in all_data.items():
        if data['rcap_conc'] and data['gpuff_conc']:
            # 0보다 큰 값만 필터링 (로그 스케일용)
            valid_idx = []
            for i in range(len(data['rcap_conc'])):
                if data['rcap_conc'][i] > 0 and data['gpuff_conc'][i] > 0:
                    valid_idx.append(i)

            if valid_idx:
                dist = [data['distances'][i] for i in valid_idx]
                rcap = [data['rcap_conc'][i] for i in valid_idx]
                gpuff = [data['gpuff_conc'][i] for i in valid_idx]

                h1, = ax1.loglog(dist, rcap, '--', color=COLORS[stab], linewidth=2.5, label=f'RCAP {stab}')
                h2, = ax1.loglog(dist, gpuff, 'o', color=COLORS[stab], markersize=7, label=f'Gpuff {stab}')
                rcap_handles.append(h1)
                gpuff_handles.append(h2)

    ax1.set_xlabel('Distance (km)', fontsize=14)
    ax1.set_ylabel('Ground Air Conc. (Bq-s/m³)', fontsize=14)
    ax1.set_title('Ground Air Concentration: All Stability Classes', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, which='both', alpha=0.3)

    # 범례: RCAP 왼쪽열, Gpuff 오른쪽열
    all_handles = rcap_handles + gpuff_handles
    all_labels = [h.get_label() for h in all_handles]
    ax1.legend(all_handles, all_labels, loc='lower left', fontsize=12, ncol=2)

    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'compare_all_ground_air_conc.png'), dpi=150, bbox_inches='tight')
    print("Saved: compare_all_ground_air_conc.png")
    plt.close(fig1)


if __name__ == "__main__":
    print("=" * 50)
    print("  Ground Air Conc: All Stability Classes")
    print("=" * 50)
    create_combined_plots()
    print(f"\nOutput saved to: {OUTPUT_DIR}")
