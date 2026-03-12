# RKNN NPU 포팅 실패 원인 분석

## 배경

BCResNet-t2 (Broadcast Residual Network, τ=2) 웨이크워드 모델을 ONNX에서 RKNN으로 변환하여 RK3588 NPU에서 실행하려 했으나, 변환은 성공해도 NPU 실행 결과가 **항상 동일한 상수값**을 출력하는 문제가 발생했다.

- ONNX (CPU): 정상 동작 (98.68% 정확도)
- RKNN NPU: 입력에 무관하게 `logits = [0.334, -0.334]` 고정 출력

---

## 버그 1 — ReduceMean 미지원

### 증상

```
E RKNN: Unknown op target: 0
```

변환 시 경고 없이 통과하지만, 런타임에서 `ReduceMean` 연산자를 인식하지 못해 해당 레이어의 출력이 0으로 고정된다.

### 원인

RKNN-Toolkit2 v2.3.2의 NPU 백엔드가 `ReduceMean` 연산자를 지원하지 않는다. BCResNet의 BCBlock 구조에서 f1 브랜치는 temporal mean을 계산하기 위해 `ReduceMean(axis=[2])`를 사용한다:

```
BCBlock f1 branch:
  input (1,C,H,W)
    → ReduceMean(axis=[2])   ← 여기서 실패
    → (1,C,1,W)
    → depthwise Conv(1×3) → Sigmoid → Mul → pointwise Conv
```

`ReduceMean`이 0을 반환하면 이후 f1 브랜치 전체가 0이 되어, f1+f2 덧셈 결과가 f2만 남게 된다. 그 결과 모든 BCBlock이 손상되어 최종 출력이 상수로 고정된다.

### 영향 범위

모델에 `ReduceMean(axis=[2])` 4개, `ReduceMean(axis=[2,3])` 1개 존재 — 모든 BCBlock에 영향.

---

## 버그 2 — H=1 중간 텐서 Conv 실패

### 증상

```
E RKNN: failed to submit!, task number: 3
```

버그 1을 우회하여 `ReduceMean`을 `depthwise Conv`로 교체하면, NPU 실행 시 위 에러가 발생하고 출력이 0으로 고정된다.

### 원인

`ReduceMean(axis=[2])`를 `depthwise Conv(kernel=(H,1))`로 교체하면 출력 shape이 `(1,C,1,W)` (H=1)가 된다. 이 H=1 텐서가 이후 `depthwise Conv(kernel=(1,3))`의 입력으로 사용될 때 NPU가 처리하지 못한다.

```
rm_out: (1,C,1,W)
  → f1.0.Conv (1×3 depthwise)  ← H=1 중간 텐서, NPU 실패
  → Sigmoid → Mul
  → f1.1.Conv
```

**중요한 점**: H=1 텐서가 최종 출력이면 정상 동작한다. 중간 연산의 입력으로 사용될 때만 실패한다. RKNN NPU의 내부 스케줄러가 H=1 피처맵을 중간 버퍼로 처리하지 못하는 것으로 추정된다.

---

## 버그 3 — AddReLU Fusion의 Broadcast 버그

### 증상

버그 1, 2를 모두 우회해도 NPU 출력이 여전히 상수 `[0.334, -0.334]`.

서브그래프 격리 실험 결과:
- `Add` 단독 출력: 정상 ✓
- `Add + ReLU` 출력: 모두 0 ✗

### 원인

RKNN 컴파일러가 `Add → ReLU` 패턴을 `AddReLU` 퓨전 연산자로 최적화한다. 이 퓨전 연산자가 **broadcast 덧셈**(`(1,C,1,W) + (1,C,H,W)`)을 올바르게 처리하지 못한다.

```
f1 branch output:  (1,C,1,W)   ← broadcast 대상
f2 branch output:  (1,C,H,W)
         Add → ReLU             ← RKNN이 AddReLU로 퓨전
                                  broadcast 시 AddReLU 버그 발생
```

- `Add`만 단독으로는 broadcast 정상
- `AddReLU`(퓨전) 상태에서는 broadcast 오류 → 출력 0

---

## 정리

| 버그 | 연산 | 증상 | 발견 방법 |
|------|------|------|-----------|
| 1 | `ReduceMean` 미지원 | 상수 출력 | 에러 로그 `Unknown op target` |
| 2 | H=1 중간 Conv 실패 | `failed to submit` | 서브그래프 격리 테스트 |
| 3 | `AddReLU` broadcast 버그 | 상수 출력 (0) | Add vs Add+ReLU 서브그래프 비교 |

세 버그 모두 RKNN-Toolkit2 v2.3.2의 NPU 백엔드 제한/버그이며, 상위 레벨 API(ONNX 변환)에서는 오류 없이 통과한다. 모두 **ONNX 그래프 수준에서 우회**하는 방식으로 해결했다.
