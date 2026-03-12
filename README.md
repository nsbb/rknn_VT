# BCResNet-t2 Wake Word Detection on RK3588 NPU

BCResNet-t2 웨이크워드 감지 모델을 RK3588 NPU에서 실행하기 위한 포팅 작업 결과물.

원본 ONNX 모델을 그대로 변환하면 NPU에서 상수 출력이 발생하는 문제를 ONNX 그래프 수준에서 우회하여 해결했다. ONNX CPU와 동일한 98.68% 정확도를 NPU에서 달성했으며, 118분 TV 뉴스 배경음 기준 FAR = 0.00/hr (threshold=0.55 + EMA).

---

## 성능 요약

| 항목 | 값 |
|------|-----|
| 정확도 (test set, N=1,897) | **98.68%** |
| Wake Recall | 93.95% (264/281) |
| Non-Wake Specificity | 99.50% (1608/1616) |
| E2E 레이턴시 (LogMel + NPU) | **21.30 ms** |
| 실시간 여유 (hop 160ms 기준) | **7.5배** |
| FAR @ threshold=0.55 + EMA | **0.00/hr** (118분 기준) |
| 추천 threshold | **0.55** |

---

## 핵심 문제 및 해결

원본 ONNX를 RKNN으로 변환하면 NPU에서 3가지 버그로 인해 상수 출력이 발생한다. 모두 ONNX 그래프 수정으로 우회했다.

| # | 버그 | 원인 | 우회법 |
|---|------|------|--------|
| 1 | `ReduceMean` 미지원 | NPU 백엔드 미구현 | `depthwise Conv`로 교체 |
| 2 | H=1 중간 Conv 실패 | NPU 스케줄러 제한 | `Pad(H=1→4)` + `Slice(row 0)` |
| 3 | `AddReLU` broadcast 버그 | RKNN 퓨전 옵티마이저 버그 | `Expand`로 사전 확장 |

자세한 내용: [`docs/01_rknn_bugs_root_cause.md`](docs/01_rknn_bugs_root_cause.md)

---

## 주요 파일

| 파일 | 설명 |
|------|------|
| `fix_rknn_graph.py` | ONNX 그래프 수정 핵심 스크립트 |
| `convert_fixed_only.py` | fixed.onnx → .rknn 변환 |
| `BCResNet-t2-npu-fixed.rknn` | 최종 NPU 모델 |
| `inference_rknn.py` | 전체 정확도 평가 |
| `threshold_sweep.py` | threshold 최적화 |
| `measure_far_npu.py` | 장시간 배경음 FAR 측정 |
| `bench_e2e.py` | E2E 레이턴시 벤치마크 |

---

## 빠른 시작

```bash
# 1. ONNX 그래프 수정
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py

# 2. RKNN 변환
conda run -n RKNN-Toolkit2 python convert_fixed_only.py

# 3. NPU 동작 확인
conda run -n RKNN-Toolkit2 python test_npu_fixed.py

# 4. 전체 평가
conda run -n RKNN-Toolkit2 python inference_rknn.py
```

환경 설정 및 전체 재현 방법: [`docs/04_environment_setup.md`](docs/04_environment_setup.md)

---

## 문서

| 문서 | 내용 |
|------|------|
| [`docs/01_rknn_bugs_root_cause.md`](docs/01_rknn_bugs_root_cause.md) | NPU 포팅 실패 원인 분석 (3가지 버그) |
| [`docs/02_fix_solution.md`](docs/02_fix_solution.md) | ONNX 그래프 수정 방법 및 구현 |
| [`docs/03_test_results.md`](docs/03_test_results.md) | 수정 전후 테스트 결과 비교 |
| [`docs/04_environment_setup.md`](docs/04_environment_setup.md) | 환경 설정 및 재현 방법 |
| [`benchmark_results.md`](benchmark_results.md) | 전체 성능 수치 요약 |

---

## 환경

- RK3588, RKNN-Toolkit2 v2.3.2, Python 3.8
- conda env: `RKNN-Toolkit2`
