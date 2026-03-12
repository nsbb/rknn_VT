# 프로젝트 인수인계 문서 — BCResNet-t2 RKNN NPU 포팅

> **작성일**: 2026-03-12
> **목적**: 새 Claude 세션이 아무 사전 지식 없이 이 작업을 이어받을 수 있도록 모든 컨텍스트를 정리

---

## 1. 프로젝트 목표

BCResNet-t2 웨이크워드 모델을 **RK3588 NPU에서 정확하게 동작**시키는 것.
ONNX 기준 92.3% 정확도를 NPU에서도 재현해야 함.

---

## 2. 환경 정보

| 항목 | 값 |
|------|-----|
| 하드웨어 | RK3588 (ARM64) |
| OS | Linux 5.10 (rockchip) |
| Python | 3.8 (conda 환경 `RKNN-Toolkit2`) |
| RKNN-Toolkit2 | v2.3.2 (변환/빌드용, `rknn.api`) |
| rknn-toolkit-lite2 | v2.3.2 cp38 arm64 (온디바이스 추론용, `rknnlite.api`) |

**중요**: `rknn.api`와 `rknnlite.api`는 같은 프로세스에서 동시 임포트 불가.
변환 스크립트와 추론 스크립트를 **분리**해야 함.

모든 명령은:
```bash
conda run -n RKNN-Toolkit2 python <script>.py
```

---

## 3. 모델 정보

- **원본**: `BCResNet-t2-Focal-ep110.onnx`
- **입력**: `(1, 1, 40, 151)` NCHW — LogMel spectrogram (40 mel bins, 151 time frames)
- **출력**: `(1, 2)` logits — [비웨이크, 웨이크]
- **클래스**: 0 = 비웨이크워드, 1 = 웨이크워드
- **ONNX 정확도**: 92.3% (1897/1897 테스트 샘플)

### BCResNet 아키텍처 요약

```
Input(1,1,40,151)
  → cnn_head: Conv(stride=2) + ReLU → (1,32,20,151) → Conv(32→16) + ReLU → (1,16,20,151)
  → BCBlock 그룹 0 (×2): 채널=16, H=20
  → BCBlock 그룹 1 (×2): 채널=24, H=10
  → BCBlock 그룹 2 (×4): 채널=32, H=5
  → BCBlock 그룹 3 (×4): 채널=40, H=5
  → classifier: Conv(→64) + ReLU + ReduceMean(axes=[2,3]) + Linear → (1,2)
```

#### BCBlock 내부 구조 (f1/f2 두 브랜치)

```
BCBlock(input=(1,C,H,W)):
  ├── f2 branch (spectral):
  │     input → f2.0(depthwise Conv 3×1 + BN + ReLU)
  │           → SubSpectralNorm (Reshape→BN→Reshape)
  │           → f2.1(depthwise Conv 1×W?)
  │           → f2_out (1,C,H,W)
  │
  ├── f1 branch (temporal mean):
  │     input → ReduceMean(axis=2) → (1,C,1,W)  ← 공간 평균
  │           → f1.0(depthwise Conv 1×3, padding) → Swish (sigmoid + mul)
  │           → f1.1(pointwise Conv 1×1)
  │           → f1_out (1,C,1,W)
  │
  └── Add(f1_out broadcast + f2_out) → ReLU → output (1,C,H,W)
```

**SubSpectralNorm(SSN)**: Reshape(C,H,W → C*S,H/S,W) → BN → Reshape back.
BCResNet-t2에 SSN이 12개 존재.

---

## 4. 문제 진단 과정 (요약)

### 4.1 최초 증상

`inference_rknn.py` 실행 시 NPU 출력이 모두 `[0.334, -0.334]` (상수) → 모든 예측이 class 0.

### 4.2 원인 1: ReduceMean 미지원

RKNN이 `ReduceMean` 노드를 "Unknown op target: 0"으로 처리 불가.
→ **해결**: `ReduceMean` → depthwise Conv 교체:
- `ReduceMean(axis=2)` on (1,C,H,W): → depthwise Conv(group=C, kernel=(H,1), weight=1/H)
- `ReduceMean(axes=[2,3])` on (1,C,1,W): → depthwise Conv(group=C, kernel=(1,W), weight=1/W)

### 4.3 원인 2: H=1 중간 텐서에서 Conv 실패

`after_f1_conv: NPU=[0,0] const=True` (ONNX=[0,0.0007])
에러: `"E RKNN: failed to submit!, op id: 8, op name: Conv:.../f1/f1.0/block/block.0/Conv, task number: 3"`

f1 브랜치의 `depthwise Conv(1×3)` 이 *중간* 텐서로 `(1,C,1,W)` (H=1) 에서 동작할 때 NPU 서브밋 실패.
(단, 해당 텐서가 *최종 출력*일 때는 정상 동작 — 서브그래프 테스트로 확인)

→ **해결**: f1 브랜치 전/후에 Pad/Slice 삽입:
- ReduceMean_conv 출력 (1,C,1,W) → Pad (H=1→4) → (1,C,4,W)
- f1.0.Conv ~ f1.1.Conv는 H=4 텐서에서 동작
- f1.1.Conv 출력 (1,C,4,W) → Slice (row 0) → (1,C,1,W)

### 4.4 원인 3: 방송 Add+ReLU 퓨전 버그

서브그래프 테스트 결과:
- `fix_f2_out`: NPU 정확 ✓
- `fix_add_out`: NPU 정확 ✓ (Add만 포함)
- `fix_after_block0`: NPU=[0,0] 오답 ✗ (Add+ReLU 포함)

RKNN이 `Add + ReLU`를 `AddRelu`로 퓨전(fuse)할 때,
브로드캐스트 Add `(1,C,1,W) + (1,C,H,W) → (1,C,H,W)` 가 잘못 처리됨.

→ **해결**: Slice 뒤에 **Expand 노드** 추가 (`(1,C,1,W) → (1,C,H,W)`).
이로써 Add의 두 입력이 같은 shape가 되어 브로드캐스트 없이 동작.

---

## 5. 현재 상태 (✅ NPU 동작 확인됨)

### 최종 테스트 결과

```
ONNX  probs: [0.07738747 0.92261255]  pred=1
NPU   probs: [0.07765744 0.92234260]  pred=1
Constant output (bad)?: False
Match ONNX vs NPU (atol=0.05): True
```

NPU가 올바른 예측(pred=1)을 출력하며 ONNX와 수치적으로 일치.

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `fix_rknn_graph.py` | **핵심** — 3가지 그래프 수정 모두 적용, ONNX 검증 포함 |
| `BCResNet-t2-npu-fixed.onnx` | 수정된 ONNX (ReduceMean→Conv, Pad/Slice/Expand 적용) |
| `BCResNet-t2-npu-fixed.rknn` | 변환 완료된 RKNN 모델 (NPU 동작 확인) |
| `convert_fixed_only.py` | `BCResNet-t2-npu-fixed.onnx` → RKNN 변환 |
| `test_npu_fixed.py` | NPU 단일 샘플 테스트 (ONNX와 비교) |
| `inference_rknn.py` | 전체 테스트셋 정확도 + FAR 평가 |

---

## 6. fix_rknn_graph.py 동작 방식 (상세)

```python
# 입력: BCResNet-t2-Focal-ep110.onnx (원본)
# 출력: BCResNet-t2-npu-fixed.onnx (3가지 수정 적용)

# ── Pass 1: ReduceMean → depthwise Conv ────────────────────────────────────
# BCBlocks 0.0~3.3의 ReduceMean(axis=2): kernel=(H,1), group=C
# classifier의 ReduceMean(axes=[2,3]): kernel=(1,W), group=C
# → RKNN이 처리 못하던 "Unknown op target: 0" 해결

# ── Pass 2: Pad/Slice/Expand 삽입 (12개 BCBlock 각각) ───────────────────────
# 각 BCBlock의 ReduceMean_conv output (1,C,1,W)에 대해:
#   ReduceMean_conv_out → Pad(H=1→4) → f1.0.Conv(depthwise 1×3) → Swish
#                                     → f1.1.Conv(pointwise) → Slice(H=4→1)
#                                     → Expand(H=1→H_f2) → Add
#
# Pad: NCHW pads=[0,0,0,0, 0,0,3,0] (H 아래 3행 zero-pad)
# Slice: axis=2, starts=0, ends=1  (row 0만 취함)
# Expand: target shape=(1,C,H_f2,W) (f2 branch와 같은 H로 확장)
# → H=1 intermediate 실패 해결 + AddRelu broadcast 버그 우회
```

`fix_rknn_graph.py` 마지막에 ONNX 검증이 포함됨:
```
Original ONNX probs: [0.07738747 0.92261255]
Fixed  ONNX probs: [0.07738747 0.92261255]
Match (atol=1e-4): True
Max diff: 0.000000
```

---

## 7. 남은 작업

### 7.1 즉시 해야 할 것 (최우선)

**전체 테스트셋에서 RKNN NPU 정확도 측정**

```bash
conda run -n RKNN-Toolkit2 python inference_rknn.py
```

`inference_rknn.py`는 `BCResNet-t2-npu-fixed.rknn`을 로드해서 테스트셋 전체를 돌림.
예상 결과: ONNX 92.3%와 비슷한 정확도 (fp16 변환으로 약간 낮을 수 있음).

⚠️ `inference_rknn.py` 내부가 어떤 rknn 파일을 로드하는지 확인 필요:
```bash
grep "load_rknn\|rknn_path\|model_path" inference_rknn.py | head -5
```
만약 `BCResNet-t2.rknn` (옛날 파일)을 쓰고 있다면 `BCResNet-t2-npu-fixed.rknn`으로 수정.

### 7.2 성능 측정

단일 추론 레이턴시 측정:
```bash
conda run -n RKNN-Toolkit2 python -c "
from rknnlite.api import RKNNLite
import numpy as np, time
rknn = RKNNLite(verbose=False)
rknn.load_rknn('BCResNet-t2-npu-fixed.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
feat = np.random.randn(1,1,40,151).astype(np.float32)
# warmup
for _ in range(5): rknn.inference(inputs=[feat], data_format='nchw')
# bench
t = time.perf_counter()
for _ in range(100): rknn.inference(inputs=[feat], data_format='nchw')
print(f'NPU latency: {(time.perf_counter()-t)/100*1000:.2f} ms/call')
rknn.release()
"
```

### 7.3 코드 정리

진단용으로 만든 임시 파일들 정리 (선택):
- `diag_*.py` — 진단 스크립트들 (보관하거나 삭제)
- `sub_fix_*.onnx`, `sub_fix_*.rknn` — 서브그래프 테스트 파일들
- `convert_sub.py`, `convert_stack.py` 등 임시 변환 스크립트들

### 7.4 실시간 마이크 연결

현재는 WAV 파일로만 테스트. 마이크 연결 후 `inference_rknn.py`의 스트리밍 부분 검증 필요.

---

## 8. 전체 워크플로우

```
원본 모델 수정:
  BCResNet-t2-Focal-ep110.onnx
       ↓ (fix_rknn_graph.py)
  BCResNet-t2-npu-fixed.onnx  ← ONNX 검증 포함, Max diff=0.000000
       ↓ (convert_fixed_only.py)
  BCResNet-t2-npu-fixed.rknn  ← NPU에서 올바른 출력 확인

추론:
  inference_rknn.py  →  로드 BCResNet-t2-npu-fixed.rknn  →  정확도 측정
  test_npu_fixed.py  →  단일 샘플 빠른 검증
```

---

## 9. 핵심 인사이트 (삽질 안 하려면 꼭 읽기)

### 9.1 RKNN 변환 시 주의

1. `rknn.api`와 `rknnlite.api` 같은 프로세스에서 import 금지
2. RKNN은 ONNX 노드를 위상 정렬(topological order) 요구 — 노드 삽입 시 순서 주의
3. `"Unknown op target: 0"` 경고는 빌드 성공해도 나옴 (무시 가능, rknn 파일 생성됨)
4. INT64 타입의 constant tensor들 (Pad의 pads, Slice의 starts/ends/axes)을 RKNN이 float16으로 캐스팅하는 경고 나옴 → 무시 가능

### 9.2 RKNN NPU 버그 목록

| 버그 | 증상 | 우회법 |
|------|------|--------|
| `ReduceMean` 미지원 | "Unknown op target: 0", CPU fallback | depthwise Conv 교체 |
| H=1 intermediate Conv | "failed to submit!" runtime error | Pad (H=1→4) + Slice |
| AddRelu broadcast fusion 버그 | Add+ReLU가 (1,C,1,W)+(1,C,H,W)를 잘못 처리 | Expand로 명시적 확장 후 Add |

### 9.3 RKNN seperate_large_kernel_conv

RKNN 내부 최적화 `seperate_large_kernel_conv`가 kernel=(H,1) conv를 자동으로 slice+conv+add로 분해함.
이는 의도된 동작이며 수치적으로 올바름 (변환 로그에서 보임).

### 9.4 서브그래프 추출 패턴

```python
onnx.utils.extract_model(
    'BCResNet-t2-npu-fixed.onnx',
    'sub_test.onnx',
    input_names=['input'],
    output_names=['/backbone/BCBlocks.0.0/Relu_output_0']
)
```
특정 레이어까지만 잘라내서 RKNN 변환 후 NPU에서 테스트하는 방식으로 문제 위치 파악함.

---

## 10. 테스트에 사용한 WAV 파일

```
wallpad_HiWonder_251113/lkk/lkk_1_2.wav  ← 웨이크워드 발화, 정답 label=1
```

빠른 검증 시 이 파일로 테스트. `test_npu_fixed.py`가 이 파일 사용.

---

## 11. 파일 목록 (현재 상태)

### 중요 파일

| 파일 | 설명 |
|------|------|
| `BCResNet-t2-Focal-ep110.onnx` | 원본 ONNX (수정 전) |
| `BCResNet-t2-npu-fixed.onnx` | **현재 사용 모델** (3가지 그래프 수정 적용) |
| `BCResNet-t2-npu-fixed.rknn` | **현재 사용 RKNN** (NPU 동작 확인) |
| `fix_rknn_graph.py` | ONNX 그래프 수정 스크립트 (원본→fixed) |
| `convert_fixed_only.py` | fixed.onnx → .rknn 변환 |
| `test_npu_fixed.py` | 단일 샘플 NPU 테스트 |
| `inference_rknn.py` | 전체 정확도 평가 |

### 진단 스크립트 (참고용)

| 파일 | 했던 일 |
|------|---------|
| `diag_extract_sub.py` | BCBlock 레벨 서브그래프 추출 |
| `diag_extract_sub2.py` | BCBlock 내부 세부 서브그래프 추출 |
| `test_sub_npu.py` | 원본 모델 서브그래프 NPU 테스트 |
| `test_sub2_npu.py` | 수정 전 BCBlock 내부 NPU 테스트 |
| `diag_npu_health.py` | NPU 하드웨어 정상 동작 확인 |
| `diag_minimal.py` | 최소 모델로 NPU 기본 동작 확인 |

---

## 12. 빠른 재현 절차

완전히 처음부터 재현하려면:

```bash
# 1. 그래프 수정 (원본 ONNX → npu-fixed ONNX)
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py
# 출력: "Match (atol=1e-4): True, Max diff: 0.000000"

# 2. RKNN 변환
conda run -n RKNN-Toolkit2 python convert_fixed_only.py
# 출력: "Done: BCResNet-t2-npu-fixed.rknn"

# 3. 단일 샘플 NPU 테스트
conda run -n RKNN-Toolkit2 python test_npu_fixed.py
# 기대: "NPU probs: [~0.078, ~0.922]  pred=1, Match=True"

# 4. 전체 정확도 평가
conda run -n RKNN-Toolkit2 python inference_rknn.py
# 기대: 92% 내외 정확도
```

---

## 13. 향후 고려 사항

- **INT8 양자화**: 현재 FP16 변환. INT8 시도 시 calibration dataset 필요
- **멀티코어 NPU**: `NPU_CORE_0_1_2` 사용 시 latency 개선 가능 (단, 3코어 모드는 모델 구조에 따라 부정확 가능 — 테스트 필요)
- **마이크 스트리밍**: sliding window (1.5s, hop=160ms) 방식으로 연속 추론
- **후처리 최적화**: EMA + N-of-M 파라미터 튜닝
