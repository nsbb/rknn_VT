# RKNN NPU 포팅 수정 방법

## 전략

RKNN 런타임을 수정하거나 우회하는 대신, **변환 전 ONNX 그래프를 직접 수정**하여 NPU가 지원하는 동등한 연산으로 교체했다. 수정된 ONNX는 원본과 수치적으로 동일(Max diff = 0.000000)하면서 NPU에서 정상 동작한다.

핵심 스크립트: **`fix_rknn_graph.py`**

---

## 수정 1 — ReduceMean → depthwise Conv 교체

### 아이디어

`ReduceMean(axis=[2])`는 H차원 평균이다. 이는 모든 가중치가 `1/H`인 `depthwise Conv(kernel=(H,1))`와 수학적으로 동일하다.

### 구현

```python
# ReduceMean(axis=[2]): (1,C,H,W) → (1,C,1,W)
C, H = in_shape[1], in_shape[2]
w = (1.0 / H) * np.ones((C, 1, H, 1), dtype=np.float32)
conv_node = helper.make_node('Conv', inputs=[in_name, w_name], outputs=[out_name],
    group=C, kernel_shape=[H, 1], pads=[0,0,0,0], strides=[1,1])

# ReduceMean(axis=[2,3]): (1,C,1,W) → (1,C,1,1)
w = (1.0 / W) * np.ones((C, 1, 1, W), dtype=np.float32)
conv_node = helper.make_node('Conv', ..., kernel_shape=[1, W])
```

가중치는 ONNX initializer로 저장되어 모델에 포함된다. 학습 파라미터가 아니므로 재학습 불필요.

---

## 수정 2 — H=1 중간 Conv 우회 (Pad + Slice)

### 아이디어

NPU가 H=1 중간 텐서를 처리하지 못하므로, f1 브랜치를 통과하는 동안만 H를 4로 늘렸다가, 이후 원래 행(row 0)만 잘라낸다.

```
[수정 전]
rm_out (1,C,1,W) → f1.0.Conv(1×3) → [NPU 실패]

[수정 후]
rm_out (1,C,1,W) → Pad(H:1→4) → (1,C,4,W)
                 → f1.0.Conv(1×3) → Sigmoid → Mul
                 → f1.1.Conv → (1,C,4,W)
                 → Slice(axis=2, 0:1) → (1,C,1,W)
```

Pad는 zero-padding이므로 row 0의 연산 결과는 원래 H=1 연산과 동일하다.

### 구현

```python
# Pad: H 아래에 3행 zero-padding
pads_val = np.array([0,0,0,0, 0,0,3,0], dtype=np.int64)
pad_node = helper.make_node('Pad', inputs=[rm_out, pads_name], outputs=[pad_out])

# f1.0.Conv 입력을 rm_out → pad_out으로 교체
f1_0_conv.input[i] = pad_out

# Slice: row 0만 추출
slice_node = helper.make_node('Slice',
    inputs=[f1_1_out, starts(=[0]), ends(=[1]), axes(=[2])],
    outputs=[slice_out])
```

**위상 정렬 유의사항**: Pad 노드는 반드시 rm_conv 노드 직후에, Slice 노드는 f1.1.Conv 노드 직후에 삽입해야 한다. RKNN 변환기가 노드 순서를 위상 정렬로 검증하기 때문이다. `insert_after` 딕셔너리 방식으로 삽입 위치를 관리했다.

---

## 수정 3 — AddReLU broadcast 버그 우회 (Expand)

### 아이디어

`Add + ReLU` 퓨전 연산자의 broadcast 버그를 우회하기 위해, Add 전에 f1 출력을 f2와 같은 shape으로 명시적으로 확장한다. 같은 shape끼리 Add이면 broadcast가 발생하지 않으므로 퓨전 버그를 피할 수 있다.

```
[수정 전]
f1 output: (1,C,1,W) + f2: (1,C,H,W) → AddReLU [broadcast 버그]

[수정 후]
f1 output: (1,C,1,W)
  → Expand(shape=[1,C,H,W]) → (1,C,H,W)
  → Add with f2: (1,C,H,W)  [same-shape, no broadcast]
  → ReLU
```

`Expand`는 메모리를 실제로 복사하므로 RKNN이 정상 처리한다.

### 구현

```python
expand_shape_val = np.array([1, C, H_f2, W], dtype=np.int64)
expand_node = helper.make_node('Expand',
    inputs=[slice_out, exp_shape_name],
    outputs=[expand_out])

# Add 노드 입력 교체: f1_1_out → expand_out
add_input_remap[f1_1_out] = expand_out
```

---

## 전체 수정 흐름

```
원본 ONNX (BCResNet-t2-Focal-ep110.onnx)
  │
  ├─ Pass 1: 모든 ReduceMean → depthwise Conv 교체
  │          + (rm_conv, rm_out) 쌍 목록 수집
  │
  ├─ Pass 2: 각 BCBlock에 대해
  │          rm_out 뒤에 Pad 삽입
  │          f1.1.Conv 뒤에 Slice + Expand 삽입
  │          Add 입력 교체
  │
  ├─ 노드 리스트 재구성 (위상 정렬 유지)
  │
  ├─ shape inference
  │
  ├─ ONNX 수치 검증 (onnxruntime으로 비교)
  │          Max diff = 0.000000 ✓
  │
  └─ BCResNet-t2-npu-fixed.onnx 저장
```

```
BCResNet-t2-npu-fixed.onnx
  │
  └─ convert_fixed_only.py (rknn.api 사용)
       → BCResNet-t2-npu-fixed.rknn
```

---

## 파일 목록

| 파일 | 역할 |
|------|------|
| `fix_rknn_graph.py` | ONNX 그래프 수정 (3가지 버그 우회) |
| `convert_fixed_only.py` | fixed.onnx → .rknn 변환 |
| `BCResNet-t2-npu-fixed.onnx` | 수정된 ONNX (NPU 호환) |
| `BCResNet-t2-npu-fixed.rknn` | 최종 NPU 모델 |
| `inference_rknn.py` | NPU 추론 및 전체 평가 |

## 실행 순서

```bash
# 1. ONNX 그래프 수정
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py

# 2. RKNN 변환
conda run -n RKNN-Toolkit2 python convert_fixed_only.py

# 3. 단일 샘플 테스트
conda run -n RKNN-Toolkit2 python test_npu_fixed.py

# 4. 전체 평가
conda run -n RKNN-Toolkit2 python inference_rknn.py
```

> 주의: `rknn.api`와 `rknnlite.api`를 같은 프로세스에서 동시에 import하면 충돌한다. 변환(`rknn.api`)과 추론(`rknnlite.api`)은 별도 스크립트로 분리되어 있다.
