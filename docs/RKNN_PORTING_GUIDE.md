# RKNN NPU 포팅 가이드 — 다음 세션을 위한 인수인계

> **이 문서의 목적**: RK3588 NPU에서 커스텀 모델을 실행하려 할 때 동일한 시행착오를 반복하지 않도록 작성한 인수인계 문서. CLAUDE.md와 함께 읽을 것.

---

## TL;DR — 핵심 교훈 3가지

1. **RKNN 변환 성공 ≠ NPU 정상 동작**: 변환이 경고 없이 끝나도 NPU 런타임에서 틀린 결과를 낼 수 있다. 반드시 실제 NPU에서 실행해 출력값이 입력에 따라 변하는지 확인해야 한다.

2. **상수 출력 = RKNN 지원 안 되는 연산자**: NPU 출력이 입력에 무관하게 항상 같은 값이면, 어딘가에 NPU가 처리 못 하는 연산자가 있다는 뜻이다. 변환 로그의 경고를 꼼꼼히 확인하고, 서브그래프를 잘라내어 어느 레이어에서 깨지는지 찾아야 한다.

3. **ONNX 그래프 수정이 정답**: RKNN 버그를 우회하는 가장 효과적인 방법은 ONNX 그래프를 직접 수정하여 NPU가 지원하는 동등한 연산으로 교체하는 것이다. 모델 재학습이나 RKNN 설정 조정은 효과가 없었다.

---

## 실패한 접근 방식들 (하지 말 것)

### 1. 변환 옵션 조정 시도 — 효과 없음

```python
# 이것들을 다 시도했지만 의미 없었음
rknn.config(optimization_level=0)       # 최적화 끄기
rknn.config(target_platform='rk3588')
rknn.config(mean_values=..., std_values=...)
```

RKNN 변환 옵션은 정상 지원되는 연산의 성능에만 영향을 준다. 지원 안 되는 연산자 문제는 변환 옵션으로 해결되지 않는다.

### 2. 시뮬레이터(CPU sim)와 NPU 결과가 다른 이유를 무시 — 위험

```python
rknn.init_runtime()                     # 시뮬레이터 (CPU에서 시뮬)
rknn.init_runtime(target=rk3588_board)  # 실제 NPU
```

시뮬레이터에서는 정상 동작하지만 실제 NPU에서 틀린 결과가 나왔다. **시뮬레이터 결과를 신뢰하지 말 것**. 반드시 실제 NPU 하드웨어에서 검증해야 한다.

### 3. SubSpectralNorm 제거 시도 — 문제 아님

처음에 SubSpectralNorm(SSN) 레이어가 원인이라고 추측하고 SSN을 제거한 모델을 실험했다. 실제 원인은 SSN이 아니라 ReduceMean이었다. 섣불리 모델 구조를 바꾸지 말고, 먼저 실제 실패 지점을 정확히 찾아야 한다.

### 4. rknn.api + rknnlite.api 동시 import — 충돌

```python
# 이렇게 하면 충돌 발생
from rknn.api import RKNN            # 변환용 (PC)
from rknnlite.api import RKNNLite    # 추론용 (보드)
```

**반드시 별도 스크립트로 분리**: 변환은 `rknn.api`만, 추론은 `rknnlite.api`만 사용하는 스크립트를 따로 만들 것.

---

## 올바른 디버깅 순서

### Step 1. 변환 로그 확인

```python
rknn = RKNN(verbose=True)
rknn.build(do_quantization=False)
```

`Unknown op target` 경고가 있으면 해당 연산자는 NPU에서 지원 안 됨.

### Step 2. 상수 출력 확인

```python
# 서로 다른 입력 2개로 테스트
out1 = rknn.inference(inputs=[zeros_input])
out2 = rknn.inference(inputs=[random_input])
print("constant?", np.allclose(out1, out2))  # True면 문제
```

### Step 3. 서브그래프 격리로 실패 지점 찾기

```python
import onnx.utils
# 의심 레이어 바로 전/후를 output으로 지정해 잘라냄
sub = onnx.utils.extract_model(model, input_names=[...], output_names=[suspect_output])
```

잘라낸 서브그래프를 RKNN으로 변환하고 NPU에서 실행해, 어느 레이어부터 출력이 깨지는지 이진 탐색으로 찾는다.

### Step 4. 해당 연산자를 동등한 연산으로 교체

지원 안 되는 연산자를 찾았으면, ONNX 그래프를 직접 수정해 NPU가 지원하는 연산으로 교체한다.

---

## 이 프로젝트에서 발견한 RKNN 버그 목록

RKNN-Toolkit2 v2.3.2 기준. 상위 버전에서도 비슷한 버그가 있을 수 있다.

### 버그 1: ReduceMean 미지원

| 항목 | 내용 |
|------|------|
| 증상 | `Unknown op target: 0`, NPU 상수 출력 |
| 원인 | ReduceMean 연산자를 NPU 백엔드가 인식 못 함 |
| 해결 | `depthwise Conv`로 교체 (수학적으로 동일) |

```python
# ReduceMean(axis=[2]): (1,C,H,W) → (1,C,1,W)
# ↓ 교체
w = (1.0/H) * np.ones((C,1,H,1), dtype=np.float32)
Conv(group=C, kernel_shape=[H,1], weights=w)
```

### 버그 2: H=1 중간 텐서 Conv 실패

| 항목 | 내용 |
|------|------|
| 증상 | `E RKNN: failed to submit!, task number: N` |
| 원인 | H=1 피처맵이 중간 레이어 입력으로 쓰일 때 NPU 스케줄러 오류 |
| 함정 | H=1이 **최종** 출력이면 정상 동작. 중간 레이어 입력일 때만 실패 |
| 해결 | `Pad(H:1→4)` 후 처리, 이후 `Slice(row 0)` 로 복원 |

```python
# H=1 텐서가 Conv 입력으로 쓰이기 전에
Pad(pads=[0,0,0,0, 0,0,3,0])   # H: 1 → 4 (아래 3행 zero-pad)
# Conv(1×3) 처리 후
Slice(start=0, end=1, axis=2)  # H: 4 → 1 (row 0만 유지)
```

### 버그 3: AddReLU fusion broadcast 버그

| 항목 | 내용 |
|------|------|
| 증상 | `Add` 단독: 정상. `Add+ReLU` 직후: 출력 모두 0 |
| 원인 | RKNN 컴파일러가 `Add→ReLU`를 `AddReLU` 퓨전 op으로 최적화하는데, 이 퓨전 op이 broadcast `(1,C,1,W)+(1,C,H,W)` 처리 시 버그 |
| 함정 | 서브그래프를 `Add` 출력에서 자르면 정상, `ReLU` 출력에서 자르면 0 → 퓨전 op 버그임을 알 수 있음 |
| 해결 | Add 전에 `Expand`로 f1 출력을 f2와 같은 shape으로 명시 확장 → broadcast 없는 same-shape Add로 변경 |

```python
# (1,C,1,W) → (1,C,H,W) 로 명시 확장 후 Add
Expand(shape=[1,C,H,W])
# 이제 (1,C,H,W) + (1,C,H,W) → broadcast 없음 → AddReLU 퓨전 정상
```

---

## ONNX 그래프 수정 시 주의사항

### 위상 정렬(Topological Order) 필수

RKNN 변환기는 ONNX 노드가 위상 정렬 순서로 배열되어 있어야 한다. 새 노드를 삽입할 때 생산자(producer) 이전에 삽입하면 에러가 난다.

```python
# 틀린 방법: 새 노드를 앞에 prepend
graph.node.insert(0, pad_node)  # → KeyError 또는 topological sort 에러

# 올바른 방법: 생산자 노드 바로 뒤에 삽입
insert_after = {}  # node_name → [nodes to insert after]
insert_after[producer_node.name].append(pad_node)

# 최종 노드 리스트 재구성
final_nodes = []
for node in new_nodes:
    final_nodes.append(node)
    for extra in insert_after.get(node.name, []):
        final_nodes.append(extra)
```

### 수정 후 반드시 onnxruntime으로 수치 검증

```python
sess_orig = ort.InferenceSession('original.onnx')
sess_new  = ort.InferenceSession('modified.onnx')
orig_out = sess_orig.run(None, {input_name: feat})[0]
new_out  = sess_new.run(None,  {input_name: feat})[0]
print(f'Max diff: {np.abs(orig_out - new_out).max():.6f}')  # 0.000000 이어야 함
```

---

## 이 프로젝트의 현재 상태 (2026-03-12)

### 완료된 작업

- BCResNet-t2 NPU 포팅 완료 (`BCResNet-t2-npu-fixed.rknn`)
- 정확도: 98.68% (ONNX와 동일)
- E2E 레이턴시: 21.30ms (hop 160ms 기준 7.5배 여유)
- 최적 threshold: 0.55 (FAR=0.00/hr @ EMA, 118분 기준)

### 핵심 파일 위치

```
fix_rknn_graph.py          # ONNX 그래프 수정 (3가지 버그 우회) — 가장 중요
convert_fixed_only.py      # fixed.onnx → .rknn 변환
inference_rknn.py          # NPU 추론 모듈 (다른 스크립트가 import함)
test_npu_fixed.py          # 빠른 NPU 동작 확인
eval/                      # 벤치마크 & FAR 측정
docs/                      # 상세 문서
```

### 다음 작업

1. **마이크 스트리밍 연결**: 실시간 마이크 입력 → sliding window(hop=160ms) → NPU 추론 파이프라인
2. **INT8 양자화**: 현재 FP16. INT8로 변환 시 NPU 속도 추가 개선 가능

---

## 환경 정보

```bash
# 추론 (RK3588 보드)
conda run -n RKNN-Toolkit2 python <script>.py

# rknn.api (변환)와 rknnlite.api (추론)는 같은 프로세스에서 사용 불가
# → 별도 스크립트로 분리되어 있음

# 모델 파일 (gitignore됨, 로컬에만 있음)
BCResNet-t2-Focal-ep110.onnx     # 원본
BCResNet-t2-npu-fixed.onnx       # 수정본
BCResNet-t2-npu-fixed.rknn       # 최종 NPU 모델
```
