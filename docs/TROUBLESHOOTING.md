# GPUFF-RCAPv3 빌드 문제 해결 가이드

## 현재 상황

MD 파일들이 `docs/` 폴더로 이동되었습니다:
- ✅ CODING_STANDARDS.md
- ✅ PROJECT_STRUCTURE.md
- ✅ MODERNIZATION_SUMMARY.md
- ✅ QUICK_START.md
- ✅ README.md

## 빌드 실행 방법

### 방법 1: 제공된 빌드 스크립트 사용

```bash
# Windows 명령 프롬프트에서:
cd X:\code\gpuffv4\gpuff-RCAPv3
build.bat
```

또는 테스트 컴파일 스크립트:

```bash
test_compile.bat
```

### 방법 2: 수동 빌드 (Visual Studio x64 Native Tools Command Prompt)

1. **시작 메뉴에서 "x64 Native Tools Command Prompt for VS 2022" 실행**

2. **프로젝트 디렉토리로 이동:**
   ```cmd
   cd X:\code\gpuffv4\gpuff-RCAPv3
   ```

3. **컴파일 명령 실행:**
   ```cmd
   nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true
   ```

### 방법 3: Visual Studio IDE 사용

1. `gpuff-RCAPv2.vcxproj` 파일을 Visual Studio 2022에서 열기
2. 빌드 → 솔루션 빌드 (Ctrl+Shift+B)

## 일반적인 빌드 오류 및 해결 방법

### 오류 1: "Cannot find compiler 'cl.exe' in PATH"

**원인:** Visual Studio 환경이 초기화되지 않음

**해결방법:**
- "x64 Native Tools Command Prompt for VS 2022"를 사용하거나
- `build.bat` 스크립트 사용 (자동으로 환경 설정)

### 오류 2: "nvcc: command not found"

**원인:** CUDA Toolkit이 설치되지 않았거나 PATH에 없음

**해결방법:**
1. CUDA Toolkit 12.2 이상 설치 확인
2. 환경 변수 확인:
   ```cmd
   echo %CUDA_PATH%
   ```
   (예상 출력: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)

3. PATH에 CUDA bin 디렉토리 추가:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   ```

### 오류 3: Visual Studio 2022가 설치되지 않음

**해결방법:**
1. Visual Studio 2022 다운로드 (Community Edition 무료)
2. 설치 시 "C++를 사용한 데스크톱 개발" 워크로드 선택
3. CUDA 개발을 위한 추가 구성요소 포함

### 오류 4: 구문 오류 (Syntax Errors)

현대화 과정에서 소스 코드는 변경되지 않았으므로, 구문 오류가 발생하면:

1. **파일 무결성 확인:**
   - 모든 `.cuh` 파일이 존재하는지 확인
   - 파일이 손상되지 않았는지 확인

2. **Git으로 복원:**
   ```bash
   git status
   git diff
   # 필요시 특정 파일 복원:
   git checkout -- <파일명>
   ```

### 오류 5: 링크 오류 (Linking Errors)

**원인:** `-rdc=true` 플래그 누락 또는 device 코드 분리 컴파일 문제

**해결방법:**
- 컴파일 명령에 `-rdc=true` 플래그 포함 확인
- 현재 build.bat은 이미 포함하고 있음

## 빌드 검증 체크리스트

실행하기 전에 다음을 확인하세요:

- [ ] Visual Studio 2022 설치됨 (C++ 워크로드 포함)
- [ ] CUDA Toolkit 12.2+ 설치됨
- [ ] NVIDIA GPU 드라이버 최신 버전
- [ ] 모든 소스 파일 존재 (main.cu, *.cuh)
- [ ] 입력 파일 존재 (`input/RCAPdata/` 디렉토리)

## 컴파일 성공 확인

빌드가 성공하면 다음 파일들이 생성됩니다:

```
X:\code\gpuffv4\gpuff-RCAPv3\
├── gpuff.exe       (실행 파일, ~1.4MB)
├── gpuff.lib       (라이브러리)
└── gpuff.exp       (익스포트 파일)
```

## 실행 방법

빌드 후 실행:

```cmd
gpuff.exe
```

**필수 입력 파일:**
- `input/RCAPdata/Test1.inp` (또는 Test.inp)
- `input/RCAPdata/MACCS60.NDL`
- `input/RCAPdata/MACCS_DCF_New2.LIB`
- `input/RCAPdata/METEO.inp`

## 추가 도움말

더 자세한 정보는 다음 문서를 참조하세요:

- **빠른 시작:** `docs/QUICK_START.md`
- **프로젝트 구조:** `docs/PROJECT_STRUCTURE.md`
- **코딩 표준:** `docs/CODING_STANDARDS.md`

## 문의

빌드 문제가 계속되면:

1. 컴파일 오류 전체 메시지 복사
2. 시스템 환경 확인:
   - Windows 버전
   - Visual Studio 버전
   - CUDA Toolkit 버전
   - GPU 모델
3. 상세 정보와 함께 문의

---

**마지막 업데이트:** 2025-10-28
