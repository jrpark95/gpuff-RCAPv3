# RCAP Center Air Conc. 매칭 분석 보고서

## 목표
GPUFF의 Center Air Conc. 출력값을 RCAP 레퍼런스와 일치시키기

## 파일 경로

### GPUFF (수정 대상)
- **프로젝트 경로**: `C:\code\gpuffv4\gpuffv4\gpuff-RCAPv3`
- **입력 파일**: `.\input\RCAPdata\Test.inp`
- **출력 파일**: `.\output\dispersion_results.txt`
- **핵심 소스 파일**:
  - `gpuff_plot.cuh` - `update_max_values()` 함수 (농도 계산)
  - `gpuff_kernels_dispersion.cuh` - sigma 계산 함수

### RCAP (레퍼런스)
- **프로젝트 경로**: `C:\code\RCAP_DoseProejction (3)\RCAP_DoseProejction`
- **입력 파일**: `Test.inp`
- **출력 파일**: `test.out`

## 빌드 및 실행 방법

### GPUFF 빌드
```bash
cd C:\code\gpuffv4\gpuffv4\gpuff-RCAPv3
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" main.cu -o gpuff.exe -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
```

### GPUFF 실행
```bash
cd C:\code\gpuffv4\gpuffv4\gpuff-RCAPv3
./gpuff.exe
```

결과는 `.\output\dispersion_results.txt`에 저장됨

---

## 분석 과정

### 1단계: 초기 비교 (Building Wake 있음)

RCAP 입력에 building height=50m, width=50m 설정 시:

| 거리 | RCAP | GPUFF | 비율 |
|------|------|-------|------|
| 0~100m | 1.596e-03 | 3.471e-01 | ~217배 |

**원인**: Building wake effect로 RCAP sigma가 크게 증가

### 2단계: Building Wake 제거

RCAP `Test.inp`의 RT215 라인 수정:
```
# 변경 전
RT215  1          50.00            50.0

# 변경 후
RT215  1          0.0              0.0
```

### 3단계: Building Wake 제거 후 비교

| 거리 | RCAP (building=0) | GPUFF | 비율 |
|------|-------------------|-------|------|
| 0~100m | 1.954e-02 | 3.499e-02 | 1.79x |
| 100~220m | 2.399e-03 | 8.460e-03 | 3.53x |
| 220~350m | 9.595e-04 | 3.772e-03 | 3.93x |
| 350~510m | 5.027e-04 | 1.989e-03 | 3.96x |
| 0.9~1.3km | 1.119e-04 | 4.393e-04 | 3.93x |

**발견**: GPUFF가 일관되게 3.5~4배 높음 → sigma 계산 차이

### 4단계: Sigma 비교 분석

RCAP 출력의 sigma 값 vs Pasquill-Gifford 공식 계산값:

| 거리 | RCAP σ_y | PG σ_y | 비율 | RCAP σ_z | PG σ_z | 비율 |
|------|----------|--------|------|----------|--------|------|
| 100m | 6.75 | 7.85 | 0.86 | 5.42 | 4.71 | 1.15 |
| 220m | 20.51 | 16.67 | 1.23 | 14.50 | 9.26 | 1.57 |
| 350m | 34.68 | 25.85 | 1.34 | 21.37 | 13.53 | 1.58 |
| 510m | 50.32 | 36.78 | 1.37 | 28.01 | 18.23 | 1.54 |
| 1280m | 119.04 | 86.22 | 1.38 | 52.27 | 36.37 | 1.44 |
| 3030m | 261.98 | 188.92 | 1.39 | 92.51 | 66.16 | 1.40 |

**발견**: RCAP sigma가 PG보다 σ_y는 ~1.37배, σ_z는 ~1.4~1.5배 큼

---

## 확산 모델 분석

### 사용 가능한 모델 (MACCS 매뉴얼 기준)

#### Power Law 공식
```
σ_y = a_y × (x/x0)^b_y
σ_z = a_z × (x/x0)^b_z
```

#### Table 2-4: Tadmor-Gur (기존 Pasquill-Gifford)
D class:
- a_y = 0.1474, b_y = 0.9031
- a_z = 0.3000, b_z = 0.6532

#### Table 2-5: NUREG/CR-7161 (Expert Elicitation)
D class:
- a_y = 0.2779, b_y = 0.881
- a_z = 0.2636, b_z = 0.751

### RCAP 결과 역산 (추정값)

RCAP 출력 sigma 값들을 fitting한 결과:
```
D class (추정):
- a_y = 0.0715, b_y = 1.0358
- a_z = 0.1873, b_z = 0.7865
```

**주의**: 이는 역산값이며, RCAP이 실제로 사용하는 공식과 다를 수 있음

### 모델 비교 (D class, x=1000m 기준)

| 모델 | σ_y (m) | σ_z (m) |
|------|---------|---------|
| Tadmor-Gur | 77.3 | 40.8 |
| NUREG/CR-7161 | 145.1 | 59.7 |
| RCAP (역산) | 93.6 | 49.3 |
| RCAP (실제 출력) | ~95 | ~48 |

---

## 현재 GPUFF 구현 상태

### Center Air Conc. 계산 공식
```cpp
// gpuff_plot.cuh - update_max_values() 함수
float center_air = Q / (2.0f * PI * sigma_y * sigma_z * ws);
```

### Sigma 계산
- 현재: Pasquill-Gifford 공식 사용 (`Sigma_h_Pasquill_Gifford_cpu()`)
- Ring outer boundary 거리에서 계산

### 수정 이력
1. 공식을 `Q/(π×σ×σ×u)` → `Q/(2π×σ×σ×u)`로 변경
2. Ring 중심 거리 → Ring 외경(outer boundary) 거리로 변경

---

## 남은 차이의 원인 (추정)

1. **Plume Rise**: RCAP에서 plume height가 50m → 136m로 상승
2. **Virtual Source 보정**: Stability class 변경 시 연속성 유지를 위한 보정
3. **ZSCALE/YSCALE**: Surface roughness 보정 계수 적용 가능성
4. **다른 확산 계수**: NUREG/CR-7161이나 다른 lookup table 사용

---

## 진행 상황 (2024-11-29)

### 완료된 작업

#### 1. 확산 계수 함수 추가 (gpuff_kernels_dispersion.cuh)
- `Sigma_y_TadmorGur_cpu()`, `Sigma_z_TadmorGur_cpu()` - MACCS Table 2-4
- `Sigma_y_NUREG7161_cpu()`, `Sigma_z_NUREG7161_cpu()` - MACCS Table 2-5

#### 2. Plume Rise 구현 (Briggs Model)
- `calculate_plume_rise()` 함수 추가
- 입력: rel_heat (W), windspeed (m/s), stability class
- 불안정/중립 조건 (A-D): Δh = 21.4 × F^0.75 / u (F < 55일 때)
- 안정 조건 (E, F): Δh = 2.4 × (F / (u × S))^(1/3)
- puff 초기화 시 `he` (effective height) 자동 설정

#### 3. 입력 파일 동기화
- GPUFF Test.inp의 RT215 수정: `50.0, 50.0` → `0.0, 0.0` (Building Wake 제거)

### 현재 비교 결과 (NUREG/CR-7161 + Plume Rise)

| 거리 | RCAP | GPUFF | 비율 | 비고 |
|------|------|-------|------|------|
| 0~100m | 1.954e-02 | 1.443e-02 | 0.74x | 근접! |
| 100~220m | 2.399e-03 | 4.464e-03 | 1.86x | |
| 0.9~1.3km | 1.119e-04 | 3.192e-04 | 2.85x | |
| 5~6.4km | 9.175e-06 | 2.648e-05 | 2.89x | |

### RCAP test.out 분석 결과

#### Plume Height 변화 (거리별)
```
0~100m:   61.38m
100~220m: 80.64m
0.7~0.9km: 134.89m
0.9km~:   136.46m (최종 도달)
```
- GPUFF 현재: 97.85m (고정, Briggs 공식 기준)
- RCAP은 거리에 따라 plume height가 점진적으로 상승

#### Sigma 값 비교 (100m 기준, D class)
| 항목 | RCAP | GPUFF (NUREG) | 비고 |
|------|------|---------------|------|
| σ_y | 6.75m | ~16.0m | GPUFF가 2.4배 큼 |
| σ_z | 5.42m | ~8.3m | GPUFF가 1.5배 큼 |

→ RCAP의 T-G_Modi는 NUREG/CR-7161과 다른 별도 계수 사용 추정

#### **중요 발견: Deposition에 의한 Q 감소**

RCAP의 `Local Source Inv., Q (Bq)` 값이 거리에 따라 감소:

| 거리 | Q (Bq) | 잔여율 |
|------|--------|--------|
| 0~100m | 9.888 | 100% |
| 0.9~1.3km | 9.625 | 97.3% |
| 5~6.4km | 8.314 | **84.1%** |
| 25~30km | 4.281 | **43.3%** |

**현재 GPUFF는 Q를 고정값으로 사용 → Deposition 미반영!**

### 다음 단계 (TODO)

1. [x] Plume Rise 구현
2. [x] NUREG/CR-7161 계수 테스트
3. [ ] **Deposition에 의한 Q 감소 반영** ← 다음 우선순위
   - 원거리 개선 효과: 최대 ~55% (25~30km 기준)
4. [ ] RCAP T-G_Modi 계수 역산 또는 확인
5. [ ] Plume Height 거리별 증가 로직 검토
6. [ ] Virtual source 구현

### 예상 개선 효과 (Deposition 반영 시)

| 거리 | 현재 비율 | Q 잔여율 | 예상 비율 |
|------|-----------|----------|-----------|
| 0~100m | 0.74x | 100% | 0.74x |
| 5~6.4km | 2.89x | 84.1% | **2.43x** |
| 25~30km | ~2.9x | 43.3% | **~1.25x** |

→ **원거리에서 상당한 개선 기대!**

---

## MACCS 대기확산 모델 상세 분석 (material2.txt 기반)

### 1. Wake Effects (건물 후류 효과)
MACCS는 building wake를 직접 모델링하지 않음. 대신 사용자가 초기 플룸 폭(SIGYINIT)과 높이(SIGZINIT)를 지정:

```
σ_y,init = W_b / 4.3
σ_z,init = H_b / 2.15
```
- W_b: 건물 폭(m)
- H_b: 건물 높이(m)

초기 플룸 차원은 **virtual source point**를 통해 가우시안 플룸 방정식에 통합됨.

### 2. Plume Rise (플룸 상승)

#### 2.1 Liftoff Criterion
플룸 상승은 풍속이 critical wind speed 미만일 때만 발생:
```
u_c = (9.09 × F / H_b)^(1/3)
```
- F: buoyancy flux (m⁴/s³)

#### 2.2 Plume Rise Equations
**Improved Briggs Model (권장)**:

불안정/중립 조건 (A-D class):
```
Δh_f = 38.7 × F^0.6 / u     (if F ≥ 55 m⁴/s³)
Δh_f = 21.4 × F^0.75 / u    (if F < 55 m⁴/s³)
```

안정 조건 (E, F class):
```
Δh_f = 2.4 × (F / (u × S))^(1/3)
```
- S: stability parameter (E: 5.04×10⁻⁴ s⁻², F: 1.27×10⁻³ s⁻²)

#### 2.3 Buoyancy Flux 계산 (Power Model)
```
F = 8.79 × 10⁻⁶ × Q̇
```
- Q̇: sensible heat release rate (watts)

#### 2.4 Average Windspeed
고도에 따른 풍속 변화:
```
u = u₀ × (h / h')^p
```
| Stability | A | B | C | D | E | F |
|-----------|-----|-----|-----|------|------|------|
| Rural p | 0.07| 0.07| 0.10| 0.15 | 0.35 | 0.55 |

### 3. Gaussian Plume Equation

기본 가우시안 플룸 방정식:
```
χ(x,y,z) = Q/u × f_G(y) × ψ(z)
```

여기서:
```
f_G(y) = 1/(√(2π)σ_y) × exp(-y²/(2σ_y²))
```

수직 분포 ψ(z)는 지면과 역전층에서 반사:
```
ψ(z) = 1/(√(2π)σ_z) × Σ[exp(-((z-h+2nH)/σ_z)²/2) + exp(-((z+h+2nH)/σ_z)²/2)]
```
- n = -100 to +100 (무한 반사 근사)
- H: mixing height

Well-mixed 조건 (H/σ_z < 0.03):
```
ψ(z) = 1/H
```

### 4. Dispersion Rate Models (확산 계수 모델)

#### 4.1 Power Law Option
```
σ_yi(x) = a_yi × (x/x₀)^b_yi
σ_zi(x) = a_zi × (x/x₀)^b_zi
```
- x₀ = 1m (단위 거리)

**Table 2-4: Tadmor-Gur (기존 P-G)**
| Class | a_y | b_y | a_z | b_z |
|-------|--------|--------|----------|--------|
| A | 0.3658 | 0.9031 | 0.00025 | 2.125 |
| B | 0.2751 | 0.9031 | 0.0019 | 1.6021 |
| C | 0.2089 | 0.9031 | 0.2 | 0.8543 |
| **D** | **0.1474** | **0.9031** | **0.3** | **0.6532** |
| E | 0.1046 | 0.9031 | 0.4 | 0.6021 |
| F | 0.0722 | 0.9031 | 0.2 | 0.6020 |

**Table 2-5: NUREG/CR-7161 (Expert Elicitation)**
| Class | a_y | b_y | a_z | b_z |
|-------|--------|-------|--------|-------|
| A/B | 0.7507 | 0.866 | 0.0361 | 1.277 |
| C | 0.4063 | 0.865 | 0.2036 | 0.859 |
| **D** | **0.2779** | **0.881** | **0.2636** | **0.751** |
| E/F | 0.2158 | 0.866 | 0.2463 | 0.619 |

#### 4.2 Virtual Source Calculation
Stability class 변경 시 연속성 유지:
```
x_yj = (σ_yi / a_j)^(1/b_j)
x_zj = (σ_zi / c_j)^(1/d_j)
```

초기 플룸 차원(SIGYINIT, SIGZINIT)도 virtual source로 처리됨.

#### 4.3 Dispersion Scaling Factors
**YSCALE, ZSCALE**: σ_y, σ_z에 대한 선형 스케일링 계수

Surface roughness 보정:
```
ZSCALE = (z₀ / z₀,ref)^q = (z₀ / 3cm)^0.2
```
- z₀: 실제 지표 거칠기 (cm)
- z₀,ref: 기준 거칠기 (3cm, Prairie Grass 실험)
- q: 지수 (보통 0.2)

대표적인 Surface Roughness 값:
- Natural Snow: 0.1 cm
- Plowed Field: 2 cm
- Grassland: 3-8 cm
- Suburban: 20-80 cm
- Urban Area: 100-400 cm
- Woodland Forest: 100+ cm

### 5. Plume Meander

#### 5.1 Original Model
```
f_m = (Δt_release / Δt₀)^F1    if Δt₀ < Δt_release ≤ Δt₁
f_m = (Δt_release / Δt₀)^F2    if Δt₁ < Δt_release ≤ 10hr
```
- Δt₀ = 600s (Prairie Grass 기준)
- F1 = 0.2, F2 = 0.25 (권장값)

#### 5.2 Regulatory Guide 1.145 Model
```
f_m = m_i × f(u)    if x ≤ d
f_m = 1             if x > d
```
- m_i = [1, 1, 1, 2, 3, 4] for stability class A-F
- u₁ = 2 m/s, u₂ = 6 m/s
- d = 800 m

### 6. Centerline Air Concentration 계산

**Centerline Air Conc (z = h)**:
```
χ(z=h̄)_j = Q̄_j / (√(2π) × σ̄_y,j × ū_j) × ψ_j(h̄)
```

**Ground-level Air Conc (z = 0)**:
```
χ(z=0)_j = Q̄_j / (√(2π) × σ̄_y,j × ū_j) × ψ_j(0)
```

**Ground Concentration**:
```
GC(y=0)_j = ΔQ_j / (√(2π) × σ_y × L_j)
```

### 7. Dry Deposition
```
f_d = exp(-v_d × ψ₀ × Δt_ref)
```
- v_d: dry deposition velocity (m/s)
- ψ₀: 지표면에서의 수직 분포 값

### 8. Wet Deposition
```
f_w = exp(-C₁ × (I/I₀)^C₂ × Δt_w)
```
- C₁: linear coefficient (CWASH1)
- C₂: exponential coefficient (CWASH2)
- I: 강수 강도 (mm/hr)

---

## GPUFF vs RCAP 매칭을 위한 핵심 포인트

### 1. Sigma 계산 방식
| 항목 | GPUFF (현재) | RCAP (추정) | 매칭 필요사항 |
|------|-------------|------------|--------------|
| 확산계수 | Tadmor-Gur | NUREG/CR-7161 또는 커스텀 | 계수 변경 검토 |
| Virtual Source | 미구현 | 구현됨 | 구현 필요 |
| ZSCALE | 1.0 | surface roughness 보정 | 보정 적용 검토 |

### 2. Center Air Conc. 공식 비교
```
GPUFF:  χ = Q / (2π × σ_y × σ_z × u)
MACCS:  χ = Q / (√(2π) × σ_y × u) × ψ(z)
```
**차이점**: GPUFF는 단순화된 공식 사용, MACCS는 수직 분포 함수 ψ(z) 사용

### 3. 다음 검토 사항
1. [ ] RCAP이 NUREG/CR-7161 계수를 사용하는지 확인
2. [ ] Virtual source 구현 여부 확인
3. [ ] ZSCALE (surface roughness) 적용 여부 확인
4. [ ] Plume rise 효과가 sigma에 미치는 영향 분석
5. [ ] 수직 분포 함수 ψ(z) 구현 검토

---

## 참고 자료

- `docs/material.txt` - MACCS 매뉴얼 발췌 (확산 모델 설명, Tadmor-Gur 계수)
- `docs/material2.txt` - MACCS 매뉴얼 상세 (Plume Rise, Virtual Source, Scaling Factors 등)
- RCAP test.out - sigma_y, sigma_z, plume height 등 상세 출력 포함

---

## 검증용 Python 코드

### Sigma 계산 비교
```python
import math

# Power Law 계수 (D class)
models = {
    'Tadmor-Gur': (0.1474, 0.9031, 0.3000, 0.6532),
    'NUREG/CR-7161': (0.2779, 0.881, 0.2636, 0.751),
    'RCAP_estimated': (0.0715, 1.0358, 0.1873, 0.7865),
}

def calc_sigma(model, distance):
    ay, by, az, bz = models[model]
    sigma_y = ay * (distance ** by)
    sigma_z = az * (distance ** bz)
    return sigma_y, sigma_z

# 테스트
for d in [100, 500, 1000, 3000]:
    print(f"\nDistance = {d}m:")
    for name in models:
        sy, sz = calc_sigma(name, d)
        print(f"  {name}: σ_y={sy:.2f}, σ_z={sz:.2f}")
```

### Center Air Conc. 계산
```python
import math

Q = 9.9  # Bq (Cs-137, 10Bq × 0.99 release fraction)
u = 2.2  # m/s (wind speed)

def center_air_conc(Q, sigma_y, sigma_z, u):
    return Q / (2 * math.pi * sigma_y * sigma_z * u)

# 예시: 100m 거리
sigma_y, sigma_z = 6.75, 5.42  # RCAP 값
chi = center_air_conc(Q, sigma_y, sigma_z, u)
print(f"Center Air Conc @ 100m: {chi:.4e} Bq-s/m³")
```
