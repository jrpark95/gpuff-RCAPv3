# GPUFF-RCAPv3

GPU 기반 원자력 사고 시나리오 대기확산 시뮬레이션 프로그램

## 개요

GPUFF-RCAPv3는 원자력 발전소 사고 시 방사성 물질의 대기 중 확산을 시뮬레이션하고, 주민 대피 및 피폭선량을 평가하는 CUDA 기반 고성능 계산 프로그램입니다.

### 주요 특징

- **GPU 병렬 연산**: CUDA를 활용한 고속 계산
- **가우시안 Puff 모델**: 대기확산 물리 모델 구현
- **다중 핵종 추적**: 최대 80개 방사성 핵종 동시 계산
- **극좌표 격자 시스템**: 발전소 중심 반경-방위각 격자
- **대피 시뮬레이션**: 주민 이동 경로 및 피폭선량 계산
- **건강영향 평가**: 급성 및 만성 건강영향 모델링

## 시스템 요구사항

- **GPU**: NVIDIA CUDA 지원 GPU
- **CUDA Toolkit**: 12.2 이상
- **컴파일러**: Visual Studio 2022 (v143)
- **OS**: Windows 10/11

## 빌드 방법

### 1. 간단한 방법 (명령줄)

```bash
# 빌드
./build.bat

# 실행
./gpuff.exe
```

### 2. Visual Studio 사용

1. `gpuff-RCAPv2.vcxproj` 파일을 Visual Studio에서 열기
2. `Ctrl+Shift+B`로 빌드
3. `F5` 또는 `Ctrl+F5`로 실행

## 프로젝트 구조

```
gpuff-RCAPv3/
├── main.cu                  # 메인 프로그램
├── gpuff.cuh               # 메인 클래스 정의
├── gpuff_struct.cuh        # 데이터 구조 정의
├── gpuff_kernels.cuh       # GPU 커널 함수
├── gpuff_init.cuh          # 초기화 함수
├── gpuff_func.cuh          # 주요 기능 함수
├── gpuff_mdata.cuh         # 기상 데이터 처리
├── gpuff_plot.cuh          # 출력 함수
├── input/RCAPdata/         # 입력 파일
│   ├── Test.inp           # 메인 설정 파일
│   ├── Test1.inp ~ Test5.inp  # 추가 시나리오
│   ├── MACCS60.NDL        # 핵종 라이브러리
│   ├── MACCS_DCF_New2.LIB # 선량환산계수
│   └── METEO.inp          # 기상 데이터
├── output/                 # 출력 파일
├── evac/                   # 대피자 데이터
├── plants/                 # 발전소 데이터
└── receptors/              # 수용점 데이터
```

## 주요 데이터 구조

### SimulationControl
시뮬레이션 전반적인 설정
- 발전소 정보 (이름, 출력, 타입)
- 격자 설정 (반경 개수, 방위각 개수)
- 파일 경로 (기상, 핵종 라이브러리, 선량계수)

### RadioNuclideTransport
방사성 물질 방출 정보
- Puff 개수 및 방출 시간
- 방출 높이 및 열 방출량
- 80개 핵종별 농도
- 입자 크기 분포

### NuclideData
핵종 물리적 특성
- 반감기, 원자량
- 화학족 (Xe, I, Cs, Te, Sr 등)
- 침적 속도 (건식/습식)
- 선량환산계수 (20개 장기)
- 붕괴 연쇄 정보

### Puffcenter_RCAP
개별 Puff 정보
- 위치 (x, y, z)
- 80개 핵종별 농도
- 확산 계수 (σ_h, σ_z)
- 기상 조건 (풍속, 풍향, 안정도, 강우)

### Evacuee
대피자 정보
- 인구수 및 위치 (극좌표/직교좌표)
- 이동 속도
- 누적 피폭선량 (흡입, 외부피폭)
- 장기별 선량 (20개 장기)

## 시뮬레이션 흐름

### 1. 초기화 단계 (main.cu:33-145)

```
입력 파일 읽기
    ↓
핵종 라이브러리 로드 (MACCS60.NDL, MACCS_DCF_New2.LIB)
    ↓
시뮬레이션 설정 읽기 (Test.inp 등)
    ↓
기상 데이터 읽기 (METEO.inp)
    ↓
Puff 초기화 (방출 위치, 시간, 농도)
    ↓
대피자 초기화 (인구 분포, 초기 위치)
    ↓
GPU 메모리 할당 및 데이터 전송
```

### 2. 시간 진행 루프 (main.cu:163)

각 시간 단계마다 다음을 수행:

```
Puff 플래그 업데이트 (방출 시간 확인)
    ↓
Puff 이동 (풍향, 풍속에 따라)
    ↓
Puff 확산 (대기 안정도에 따라 σ 증가)
    ↓
방사성 붕괴 계산 (반감기에 따라)
    ↓
침적 계산 (건식/습식)
    ↓
농도 계산 (가우시안 플룸 모델)
    ↓
대피자 이동 및 피폭선량 계산
    ↓
결과 출력 (지정된 시간 간격마다)
```

### 3. 결과 출력

- Puff 위치 및 농도 (바이너리 파일)
- 대피자 피폭선량 (장기별)
- 발전소 특정 지점 농도
- 지표 침적량

## 주요 물리 모델

### 가우시안 Puff 모델

Puff 중심 (x_p, y_p, z_p)에서 수용점 (x, y, z)의 농도:

```
C(x,y,z) = Q / ((2π)^1.5 × σ_x × σ_y × σ_z)
         × exp(-((x-x_p)²/(2σ_x²) + (y-y_p)²/(2σ_y²) + (z-z_p)²/(2σ_z²)))
```

여기서:
- Q: Puff 내 방사성 물질량
- σ_x, σ_y, σ_z: 확산 계수 (안정도 함수)

### 대기 안정도 분류

Pasquill-Gifford 안정도 클래스:
- A: 매우 불안정
- B: 불안정
- C: 약간 불안정
- D: 중립
- E: 약간 안정
- F: 안정

### 침적 모델

**건식 침적**:
```
dQ/dt = -v_d × C × A
```
- v_d: 침적 속도 (핵종 및 입자 크기 의존)
- C: 지표 농도
- A: 면적

**습식 침적**:
```
Λ = a × R^b
```
- Λ: 세정 계수
- R: 강우강도 (mm/h)
- a, b: 경험 상수

### 피폭선량 계산

**흡입 피폭**:
```
D_inh = ∫ C(t) × BR × DCF_inh dt
```
- BR: 호흡률 (m³/s)
- DCF_inh: 흡입 선량환산계수 (Sv/Bq)

**외부 피폭 (cloudshine)**:
```
D_ext = ∫ C(t) × DCF_ext dt
```
- DCF_ext: 외부피폭 선량환산계수 (Sv·m³/Bq·s)

## 주요 입력 파일

### Test.inp 구조

```
SC10: 시뮬레이션 제목
SC20: 발전소 정보 (이름, 출력, 타입)
SC30: 극좌표 격자 설정 (반경 개수, 방위각 개수)
SC31: 반경 거리 (km)
SC40: 파일 경로 (기상, 핵종, 선량계수)

RT100: 핵종 방출 정보 (입력 파일 개수)
RT200: Puff 개수
RT210: Puff 세부 정보 (방출시간, 높이, 열량)
RT220: 입자 크기 분포

RT310: 기상 샘플링 방법
RT350: 기상 조건 (풍속, 안정도, 강우)

EP200: 경보 시간
EP210: 대피 범위
EP220: 대피소 대기 시간
EP230: 대피소 체류 시간
EP240: 대피 속도 프로파일

SD50: 지표 거칠기
SD150: 인구 분포

PF100: 차폐계수
```

### MACCS60.NDL 파일

80개 방사성 핵종 정보:
- 핵종 이름, ID
- 반감기 (초)
- 원자량 (g/mol)
- 화학족
- 건식/습식 침적 여부
- 노심 재고량 (Ci/MWth)
- 붕괴 연쇄 (딸핵종, 분기비)

### MACCS_DCF_New2.LIB 파일

선량환산계수:
- 80개 핵종 × 20개 장기
- 흡입 선량계수 (Sv/Bq)
- 외부피폭 선량계수 (Sv·m³/Bq·s)
- 지표피폭 선량계수

### METEO.inp 파일

시간별 기상 데이터:
- 일자, 시간
- 풍향 (도)
- 풍속 (m/s)
- 안정도 클래스 (1-6)
- 강우강도 (mm/h)

## GPU 커널 함수

### 주요 커널 (gpuff_kernels.cuh)

1. **update_puff_flags_RCAP2**: Puff 방출 시간 확인
2. **move_puffs_by_wind_RCAP2**: Puff 이동 및 확산
3. **ComputeExposureHmix**: 대피자 피폭선량 계산
4. **reduce_organDose**: 장기별 선량 집계

### 병렬화 전략

- Puff 단위 병렬화: 각 Puff를 독립적인 스레드가 처리
- 대피자 단위 병렬화: 각 대피자를 독립적인 스레드가 처리
- 격자 단위 병렬화: 농도 계산 시 격자점별 병렬 처리

## 출력 파일

### Puff 출력 (puffs/puff_*.bin)

바이너리 형식:
```
[시간단계]
[Puff 개수]
[Puff 1 데이터] x, y, z, 80개 핵종 농도
[Puff 2 데이터] ...
```

### 대피자 출력 (evac/evac_*.bin)

바이너리 형식:
```
[시간단계]
[대피자 개수]
[대피자 1 데이터] x, y, 인구, 누적선량, 20개 장기별 선량
[대피자 2 데이터] ...
```

### 발전소 출력 (plants/plant_*.bin)

각 입력 시나리오별 특정 지점 농도 시계열

## 성능

### 테스트 환경
- GPU: NVIDIA RTX 시리즈
- Puff 수: ~수천 개
- 대피자 수: ~수만 명
- 핵종 수: 80개

### 실행 시간
- 시뮬레이션 시간: 7일 (604,800초)
- 시간 간격: 1초
- 전체 실행 시간: 수십 분 ~ 수 시간 (GPU 성능 의존)

## 주요 파라미터

### 시뮬레이션 설정 (gpuff.cuh)

```cpp
float time_end;        // 종료 시간 (초)
float dt;              // 시간 간격 (초)
int freq_output;       // 출력 주기
int nop;              // Puff 총 개수
```

### 물리 상수

```cpp
EARTH_RADIUS = 6,371,000 m
MAX_NUCLIDES = 80         // 최대 핵종 수
MAX_ORGANS = 20           // 최대 장기 수
```

## 검증 및 테스트

프로그램 검증을 위한 디버그 플래그 (gpuff.cuh:40-49):

```cpp
#define CHECK_SC 0     // 시뮬레이션 설정 출력
#define CHECK_RT 0     // 방출 정보 출력
#define CHECK_WD 0     // 기상 데이터 출력
#define CHECK_EP 0     // 대피 데이터 출력
#define CHECK_NDL 0    // 핵종 라이브러리 출력
```

각 플래그를 1로 설정하면 해당 데이터가 콘솔에 출력됩니다.

## 코드 구조

### 클래스 다이어그램

```
Gpuff 클래스
├── 멤버 변수
│   ├── puffs_RCAP: std::vector<Puffcenter_RCAP>
│   ├── d_puffs_RCAP: GPU 메모리 포인터
│   ├── evacuees: std::vector<Evacuee>
│   ├── d_evacuees: GPU 메모리 포인터
│   └── RCAP_metdata: 기상 데이터
│
└── 멤버 함수
    ├── 초기화
    │   ├── initializePuffs()
    │   ├── initializeEvacuees()
    │   └── read_meteorological_data_RCAP2()
    │
    ├── GPU 메모리 관리
    │   ├── allocate_and_copy_puffs_RCAP_to_device()
    │   ├── allocate_and_copy_evacuees_to_device()
    │   └── free_puffs_RCAP_device_memory()
    │
    ├── 시뮬레이션
    │   └── time_update_RCAP2()
    │
    └── 출력
        ├── puff_output_binary_RCAP()
        ├── evac_output_binary_RCAP()
        └── plant_output_binary_RCAP()
```

## 참고문헌

- NUREG/CR-6613: "Code Manual for MACCS2"
- Pasquill-Gifford Atmospheric Dispersion Model
- ICRP Publication 119: "Compendium of Dose Coefficients"

## 라이선스

프로젝트 라이선스 정보는 별도 문서 참조

## 문의

프로젝트 관련 문의는 개발팀에 연락

---

**마지막 업데이트**: 2025년 1월
