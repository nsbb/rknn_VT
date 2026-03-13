# 환경 설정 및 재현 방법

## 하드웨어 / 소프트웨어 환경

| 항목 | 값 |
|------|-----|
| 보드 | RK3588 (ARM Cortex-A76×4 + A55×4, NPU 6TOPS) |
| OS | Linux 5.10 (Rockchip BSP) |
| Python | 3.8 (conda env: RKNN-Toolkit2) |
| RKNN-Toolkit2 | v2.3.2 (PC 변환용) |
| rknn-toolkit-lite2 | v2.3.2 cp38 aarch64 (보드 추론용) |
| onnxruntime | 설치 필요 (검증용) |

---

## 설치

### rknn-toolkit-lite2 (RK3588 보드에서)

```bash
pip install rknn_toolkit_lite2-2.3.2-cp38-cp38-linux_aarch64.whl
```

### conda 환경 (RKNN-Toolkit2)

```bash
conda create -n RKNN-Toolkit2 python=3.8
conda activate RKNN-Toolkit2
pip install rknn-toolkit2==2.3.2
pip install onnxruntime numpy librosa soundfile
```

---

## 주요 파일 구조

```
rknn-wakeword/
├── BCResNet-t2-Focal-ep110.onnx          # 원본 모델
├── BCResNet-t2-npu-fixed.onnx            # 수정된 ONNX (NPU 호환)
├── BCResNet-t2-npu-fixed.rknn            # 최종 NPU 모델
├── fix_rknn_graph.py                     # ONNX 그래프 수정 (핵심)
├── convert_fixed_only.py                 # ONNX → RKNN 변환
├── test_npu_fixed.py                     # 단일 샘플 동작 확인
├── inference_rknn.py                     # 전체 평가 (test.csv)
├── threshold_sweep.py                    # threshold 최적화
├── measure_far_npu.py                    # FAR 측정 (배경음)
├── bench_npu.py                          # NPU 레이턴시 벤치마크
├── bench_e2e.py                          # E2E 레이턴시 벤치마크
├── npu_probs_cache.npz                   # 추론 결과 캐시
├── measure_FA/                           # FAR 측정용 배경음 파일
├── docs/                                 # 이 문서들
└── benchmark_results.md                  # 전체 성능 요약
```

---

## 처음부터 재현하는 순서

### Step 1. ONNX 그래프 수정

```bash
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py
# 출력:
#   ReduceMean(axis=2) → Conv depthwise: ...
#   Pad/Slice/Expand: ...
#   Original ONNX probs: [0.078 0.922]
#   Fixed  ONNX probs:   [0.078 0.922]
#   Max diff: 0.000000
```

### Step 2. RKNN 변환

```bash
conda run -n RKNN-Toolkit2 python convert_fixed_only.py
# → BCResNet-t2-npu-fixed.rknn 생성
```

### Step 3. NPU 동작 확인

```bash
conda run -n RKNN-Toolkit2 python test_npu_fixed.py
# 출력:
#   NPU probs: [0.079, 0.921]  Match: True
```

### Step 4. 전체 정확도 평가

```bash
conda run -n RKNN-Toolkit2 python inference_rknn.py
# → test.csv 1897샘플 평가, 정확도/FAR 출력
```

### Step 5. Threshold 최적화

```bash
conda run -n RKNN-Toolkit2 python threshold_sweep.py
# → npu_probs_cache.npz 생성 + threshold 스윕 결과
```

### Step 6. 장시간 FAR 측정

```bash
conda run -n RKNN-Toolkit2 python measure_far_npu.py > far_result.txt 2>&1 &
# measure_FA/ 디렉토리의 WAV 파일 처리 (~118분 소요 시간: ~15분)
```

---

## 알려진 제한사항

### rknn.api / rknnlite.api 충돌

같은 Python 프로세스에서 두 API를 동시에 import하면 충돌한다.

```python
# 불가 — 같은 스크립트에서
from rknn.api import RKNN          # 변환용
from rknnlite.api import RKNNLite  # 추론용
```

→ 변환(`convert_fixed_only.py`)과 추론(`inference_rknn.py` 등) 스크립트를 분리했다.

### RKNN-Toolkit2 v2.3.2 미지원 연산자

| 연산자 | 상태 |
|--------|------|
| ReduceMean | NPU 미지원 (CPU fallback → 버그) |
| H=1 중간 Conv | 런타임 오류 |
| AddReLU broadcast | 퓨전 버그 |

모두 `fix_rknn_graph.py`에서 우회 처리됨.

---

## 추천 추론 파라미터

```python
THRESHOLD = 0.55       # 검출 임계값 (F1 최적)
EMA_ALPHA = 0.3        # 지수이동평균
REFRAC_SEC = 2.0       # 재검출 쿨다운 (초)
N_N, N_M = 3, 5        # N-of-M (5프레임 중 3개)
HOP_MS = 160           # 슬라이딩 윈도우 hop
```
