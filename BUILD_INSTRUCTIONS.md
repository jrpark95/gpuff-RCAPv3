# GPUFF-RCAPv3 빌드 방법

## 🚀 초간단 빌드 (모든 환경에서 작동)

**어떤 명령 프롬프트에서든 실행 가능:**
- ✅ 일반 cmd.exe
- ✅ PowerShell
- ✅ Git Bash
- ✅ Windows Terminal
- ✅ VS Developer Command Prompt
- ✅ 기타 모든 터미널

### 빌드 명령 (단 하나!)

```cmd
build.bat
```

**끝입니다.** 스크립트가 자동으로:
1. Visual Studio 설치 위치 자동 탐지 (vswhere.exe 사용)
2. 환경 변수 자동 설정
3. CUDA 컴파일러 확인
4. 컴파일 실행
5. 결과 확인 및 출력

---

## ⚙️ 필수 요구사항

빌드 전에 설치만 되어 있으면 됩니다:

### 1. Visual Studio 2019 또는 2022
- **Edition:** Community / Professional / Enterprise (아무거나)
- **필수 워크로드:** "C++를 사용한 데스크톱 개발"
- **다운로드:** https://visualstudio.microsoft.com/

### 2. NVIDIA CUDA Toolkit
- **버전:** 12.2 이상
- **다운로드:** https://developer.nvidia.com/cuda-downloads

### 3. NVIDIA GPU 드라이버
- **최소:** Compute Capability 5.0 이상
- **권장:** 최신 버전

---

## 📁 프로젝트 파일 확인

빌드하기 전에 다음 파일들이 있는지 확인:

```
gpuff-RCAPv3/
├── main.cu                  ← 메인 소스 파일
├── gpuff.cuh                ← 헤더 파일들
├── gpuff_struct.cuh
├── gpuff_kernels.cuh
├── gpuff_init.cuh
├── gpuff_func.cuh
├── gpuff_mdata.cuh
├── gpuff_plot.cuh
└── build.bat                ← 자동 빌드 스크립트
```

---

## ✅ 빌드 확인

빌드가 성공하면:

```
============================================================================
BUILD SUCCESSFUL
============================================================================

Output files:
  - gpuff.exe  (1,234,567 bytes)
  - gpuff.lib  (1,234 bytes)
  - gpuff.exp  (567 bytes)
```

---

## 🏃 실행 방법

```cmd
gpuff.exe
```

**필수 입력 파일** (`input/RCAPdata/` 폴더에 있어야 함):
- `Test1.inp` - 시뮬레이션 시나리오
- `MACCS60.NDL` - 핵종 데이터
- `MACCS_DCF_New2.LIB` - 선량 변환 계수
- `METEO.inp` - 기상 데이터

**출력 위치:**
- `output/` - Puff 시각화 (VTK 파일)
- `evac/` - 대피자 추적 (VTK 파일)

---

## 🔧 자동 감지 기능

`build.bat`는 다음을 **자동으로 처리**합니다:

### ✨ Visual Studio 자동 탐지
1. **vswhere.exe** 사용 (공식 VS 위치 탐색 도구)
2. VS 2022 (Community/Professional/Enterprise) 순서로 검색
3. VS 2019 폴백 지원

### ✨ 환경 자동 설정
- 이미 VS 환경이 초기화된 경우 스킵
- `cl.exe`가 PATH에 없으면 자동 초기화
- 중복 초기화 방지

### ✨ CUDA 자동 확인
- `nvcc.exe` PATH 확인
- CUDA 버전 출력

---

## ❌ 문제 해결

### 문제 1: "Visual Studio not found"

**해결:**
```
Visual Studio 2019 또는 2022 설치 필요
설치 시 "Desktop development with C++" 워크로드 선택
```

### 문제 2: "CUDA Toolkit not found"

**해결:**
```
CUDA Toolkit 12.2+ 설치
설치 후 시스템 재시작 (PATH 적용)
```

### 문제 3: 빌드 실패 (컴파일 오류)

**해결:**
1. 모든 `.cuh` 파일 존재 확인
2. Git으로 파일 무결성 확인:
   ```bash
   git status
   git diff
   ```
3. `docs/TROUBLESHOOTING.md` 참조

---

## 🎯 빌드 플래그 설명

현재 사용 중인 nvcc 플래그:

| 플래그 | 설명 |
|--------|------|
| `main.cu` | 메인 소스 파일 |
| `-o gpuff.exe` | 출력 파일명 |
| `-allow-unsupported-compiler` | 최신 VS 버전 허용 |
| `-rdc=true` | Device 코드 분리 컴파일 활성화 |

**수동 빌드 (고급 사용자):**

Debug 빌드:
```cmd
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true -G -lineinfo
```

Release 빌드 (최대 성능):
```cmd
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true -O3 -use_fast_math
```

특정 GPU 타겟팅 (RTX 3090):
```cmd
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true -arch=sm_86
```

---

## 💡 왜 이렇게 만들었나?

### 기존 문제:
- ❌ x64 Native Tools Command Prompt 필수
- ❌ 수동 환경 설정 필요
- ❌ 초보자 진입 장벽

### 현재 해결:
- ✅ **어떤 터미널에서든 작동**
- ✅ Visual Studio 자동 탐지
- ✅ 환경 자동 초기화
- ✅ 명확한 오류 메시지
- ✅ 단 하나의 명령: `build.bat`

---

## 📚 추가 문서

- **빠른 시작:** `docs/QUICK_START.md`
- **프로젝트 구조:** `docs/PROJECT_STRUCTURE.md`
- **코딩 표준:** `docs/CODING_STANDARDS.md`
- **문제 해결:** `docs/TROUBLESHOOTING.md`

---

## 🎓 빌드 스크립트 작동 원리

`build.bat`는 다음 단계로 작동합니다:

1. **[1/4] Visual Studio 자동 탐지**
   - vswhere.exe로 최신 VS 찾기
   - 없으면 기본 경로 순회
   - VS 2022 → VS 2019 순

2. **[2/4] 컴파일러 환경 확인**
   - `cl.exe`가 PATH에 있는지 확인
   - 없으면 `vcvarsall.bat x64` 실행
   - 있으면 스킵 (이미 초기화됨)

3. **[3/4] CUDA 툴킷 확인**
   - `nvcc.exe` 존재 확인
   - CUDA 버전 출력

4. **[4/4] 컴파일 실행**
   - `nvcc main.cu -o gpuff.exe ...`
   - 성공/실패 메시지 출력
   - 생성된 파일 크기 표시

---

**마지막 업데이트:** 2025-10-28
**버전:** Universal Build System (환경 독립적)
