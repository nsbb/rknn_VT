# BCResNet-t2 추론 성능 보고서 (RK3588)

## 환경

- **하드웨어**: RK3588 (ARM Cortex-A76 + Mali GPU + NPU 6TOPS)
- **모델**: BCResNet-t2-Focal-ep110 (2-class 웨이크워드)
- **날짜**: 2026-03-12
- **RKNN-Toolkit2**: v2.3.2 / rknn-toolkit-lite2: v2.3.2

---

## 정확도 (test.csv, N=1897샘플)

| 런타임 | Micro Accuracy | Wake Recall | Non-Wake Specificity |
|--------|---------------|-------------|----------------------|
| ONNX (CPU, onnxruntime) | 98.68% | 93.95% (264/281) | 99.50% (1608/1616) |
| **RKNN NPU (npu-fixed)** | **98.68%** | **93.95% (264/281)** | **99.50% (1608/1616)** |

**Confusion Matrix (RKNN NPU):**
```
         Pred 0  Pred 1
Actual 0   1608       8
Actual 1     17     264
```

> ONNX와 동일한 정확도 유지 — NPU 포팅 성공

---

## 레이턴시 (단위: ms, N=200~300회 평균)

### ONNX (CPU)
| 항목 | 시간 |
|------|------|
| ONNX 추론 only | 2.48 ms |
| LogMel(numpy) only | ~15.4 ms |
| E2E (LogMel + ONNX) | 17.90 ms |

### RKNN NPU
| 항목 | 시간 |
|------|------|
| NPU 추론 only (Core 0) | 6.21 ms |
| NPU 추론 only (Core 0+1) | 5.59 ms |
| LogMel(numpy) only | 13.36 ms |
| **E2E (LogMel + NPU, Core 0)** | **21.30 ms** |
| 처리 속도 | 47 calls/sec |
| Hop 160ms 기준 실시간 여유 | **7.5배** |

> 슬라이딩 윈도우 hop=160ms 기준: E2E 21ms → hop 예산의 **13%** 사용

---

## RKNN 그래프 수정 내역 (NPU 포팅)

원본 ONNX를 그대로 변환하면 NPU에서 상수 출력 발생. 3가지 수정 필요:

| 문제 | 원인 | 우회법 |
|------|------|--------|
| "Unknown op target" | ReduceMean 미지원 | depthwise Conv 교체 |
| "failed to submit" | H=1 중간 텐서 Conv | Pad(H=1→4) + Slice(row 0) |
| AddRelu 오류 | broadcast fuse 버그 | Expand로 명시적 확장 |

수정 스크립트: `fix_rknn_graph.py`
수정 후 ONNX 일치: `Max diff = 0.000000`

---

## FAR 평가 — 정밀 측정 (measure_FA/ TV 뉴스 배경음, 총 118.3분)

> 비웨이크워드 전용 오디오 2파일 (60.9분 + 57.4분 = 1.97시간)
> prob range: [0.006, 0.482] / [0.012, 0.621]
> 파라미터: EMA α=0.3, N-of-M (3/5), Refractory=2.0s

| Threshold | Raw FA/hr | Refr FA/hr | EMA FA/hr | NoM FA/hr |
|-----------|-----------|------------|-----------|-----------|
| 0.40 | 30.42 | 10.65 | 2.03 | **0.00** |
| 0.45 | 13.69 | 6.59 | 1.01 | **0.00** |
| 0.50 | 5.58 | 4.06 | **0.00** | **0.00** |
| **0.55** | **1.52** | **1.01** | **0.00** | **0.00** |
| 0.60 | 1.01 | 1.01 | **0.00** | **0.00** |
| 0.65 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |

> **결론**: threshold=0.55 + EMA(α=0.3) or N-of-M(3/5) → **FAR = 0.00/hr** (118분 기준)
> threshold=0.55 raw 기준: 3 FA / 1.97hr = 1.52/hr

---

## FAR 참고 (test set 1897파일, 총 77분 — wake word 포함이라 의미없음)

> ⚠️ 아래는 웨이크워드 포함 test set 전체 기준 — 정확한 FAR가 아님 (참고용)

| 설정 | FAR/hour |
|------|----------|
| Raw (threshold=0.5) | 584/hr |
| Refractory | 217/hr |
| Refrac + EMA | 212/hr |
| Refrac + EMA + N-of-M | 186/hr |

ONNX CPU 기준 참고 (vad_cropped/ 200파일, 10.3분):

| 설정 | FAR/hour |
|------|----------|
| Raw | 11.63 |
| Refrac + EMA | 0.00 |
| Refrac + EMA + N-of-M | 0.00 |

---

## Threshold 최적화 (test.csv NPU 추론 결과)

| Threshold | Acc% | Prec% | Recall% | F1% | FP | FN |
|-----------|------|-------|---------|-----|-----|-----|
| 0.30 | 98.05% | 89.87% | 97.86% | 93.70% | 31 | 6 |
| 0.40 | 98.58% | 94.10% | 96.44% | 95.25% | 17 | 10 |
| 0.45 | 98.73% | 96.06% | 95.37% | 95.71% | 11 | 13 |
| **0.50 (default)** | **98.68%** | 97.22% | 93.59% | 95.37% | 8 | 18 |
| **0.55 (best F1)** | **98.79%** | **99.24%** | 92.53% | **95.76%** | **2** | 21 |
| 0.70 | 98.68% | 99.62% | 88.96% | 93.99% | 1 | 31 |

> 추천 threshold: **0.55** (F1 최대, FP=2)

**실제 FAR 추정** (label=0 클립 1616개, 총 ~64분):

| Threshold | FP 수 | 추정 FAR/hr |
|-----------|-------|------------|
| 0.50 | 8 | ~7.4/hr |
| **0.55** | **2** | **~1.9/hr** |

> ⚠️ 정밀 FAR는 measure_FA/ (장시간 배경음) 기준 재측정 필요

---

## 결론

- **NPU 포팅 완료**: 정확도 98.68% (ONNX와 동일)
- **E2E 레이턴시**: 21.3ms (hop 160ms 기준 7.5배 여유)
- **최적 threshold**: 0.55 (F1=95.76%, FP=2)
- **실측 FAR** (118분 TV 뉴스): threshold=0.55 + EMA → **0.00/hr**, raw → 1.52/hr
- ONNX CPU 추론(2.5ms) vs NPU(6.2ms): NPU가 느리지만 CPU 부하 오프로드 이점
- 다음 작업: 마이크 스트리밍 연결
