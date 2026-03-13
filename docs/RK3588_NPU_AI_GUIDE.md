# RK3588 NPU AI 포팅 가이드

> **이 문서의 목적**: RK3588 NPU에서 AI 모델(웨이크워드, STT, TTS, LLM 등)을 실행할 때 반복하지 말아야 할 실수와, 처음부터 올바르게 접근하는 방법을 정리한 범용 가이드.
>
> BCResNet-t2 웨이크워드 포팅 과정에서 직접 겪은 실패와 해결에서 얻은 교훈이다. 새 프로젝트를 시작할 때 이 문서를 먼저 읽을 것.

---

## 핵심 원칙 — 어떤 모델이든 공통

### 1. "변환 성공 ≠ NPU 정상 동작"

RKNN 변환이 경고 없이 완료되어도, 실제 NPU에서 상수 출력이 나올 수 있다. 변환 직후 반드시 실제 하드웨어에서 검증하라.

```python
# 가장 먼저 해야 할 검증
out1 = rknn.inference(inputs=[zeros_input])
out2 = rknn.inference(inputs=[random_input])
assert not np.allclose(out1, out2), "상수 출력 — NPU가 모델을 제대로 실행하지 않고 있음"
```

### 2. 시뮬레이터를 믿지 말 것

```python
rknn.init_runtime()                          # CPU 시뮬레이터 — 정상처럼 보임
rknn.init_runtime(core_mask=NPU_CORE_0)     # 실제 NPU — 다른 결과
```

시뮬레이터에서 정상이어도 실제 NPU에서 틀릴 수 있다. **항상 실제 NPU에서 검증** 후 결과를 신뢰할 것.

### 3. 상수 출력 = 지원 안 되는 연산자

출력이 입력에 무관하게 고정되면, 어딘가 NPU가 처리 못 하는 레이어가 있다는 신호다. 변환 로그에서 `Unknown op target` 경고를 찾고, 서브그래프 격리로 실패 지점을 이진 탐색한다.

### 4. ONNX 그래프 수정이 가장 확실한 해결책

RKNN 설정(optimization_level, quantization 옵션 등)을 조작하는 것은 거의 효과가 없다. 지원 안 되는 연산자를 찾았으면 **ONNX 그래프를 직접 수정**해 NPU가 지원하는 동등한 연산으로 교체하는 게 가장 빠르고 확실하다.

---

## RKNN-Toolkit2 v2.3.2 알려진 버그 / 제한사항

새 모델을 포팅할 때 이 목록을 먼저 확인한다. 상위 버전에서도 유사한 버그가 있을 가능성이 높다.

| # | 연산자 / 패턴 | 증상 | 우회법 |
|---|--------------|------|--------|
| 1 | `ReduceMean` | `Unknown op target: 0`, 상수 출력 | `depthwise Conv`로 교체 |
| 2 | H=1 중간 Conv | `failed to submit!, task number: N` | `Pad(H→4)` + `Slice(row 0)` |
| 3 | `Add→ReLU` broadcast | `AddReLU` 퓨전 시 broadcast 오류 | `Expand`로 사전 확장 |
| 4 | `rknn.api` + `rknnlite.api` 동시 import | 프로세스 충돌 | 변환/추론 스크립트 분리 |

### 버그 1: ReduceMean → depthwise Conv

```python
# ReduceMean(axis=[2]): (1,C,H,W) → (1,C,1,W)
w = (1.0/H) * np.ones((C, 1, H, 1), dtype=np.float32)
Conv(group=C, kernel_shape=[H,1], weights=w)

# ReduceMean(axis=[2,3]): (1,C,H,W) → (1,C,1,1)
w = (1.0/W) * np.ones((C, 1, 1, W), dtype=np.float32)
Conv(group=C, kernel_shape=[1,W], weights=w)
```

### 버그 2: H=1 중간 텐서

```python
# 처리 전: H를 늘림
Pad(pads=[0,0,0,0, 0,0,3,0])   # H: 1 → 4
# ... Conv 처리 ...
# 처리 후: 원래 row만 추출
Slice(start=0, end=1, axis=2)   # H: 4 → 1
```

### 버그 3: AddReLU broadcast

```python
# f1: (1,C,1,W) → Add에 앞서 명시적 확장
Expand(shape=[1, C, H_f2, W])   # (1,C,1,W) → (1,C,H,W)
# 이제 same-shape Add → broadcast 없음 → AddReLU 퓨전 정상
```

---

## ONNX 그래프 수정 공통 패턴

### 노드 삽입 시 위상 정렬 유지

RKNN은 노드 순서가 위상 정렬이어야 한다. 새 노드는 반드시 생산자(producer) 바로 뒤에 삽입한다.

```python
# 올바른 방법
insert_after = {}  # producer_node_name → [nodes_to_insert]
insert_after[producer.name].append(new_node)

final_nodes = []
for node in all_nodes:
    final_nodes.append(node)
    for extra in insert_after.get(node.name, []):
        final_nodes.append(extra)
```

### 수정 후 수치 검증 (필수)

```python
orig = ort.InferenceSession('original.onnx').run(None, {name: x})[0]
fixed = ort.InferenceSession('fixed.onnx').run(None, {name: x})[0]
print(f'Max diff: {np.abs(orig - fixed).max():.8f}')  # 반드시 0.000000
```

### 서브그래프 격리로 실패 지점 찾기

```python
import onnx.utils
sub = onnx.utils.extract_model(
    model,
    input_names=['input_tensor_name'],
    output_names=['suspect_output_name']
)
# 이 서브그래프만 RKNN 변환 후 NPU에서 테스트
```

---

## 음성 AI 태스크별 주의사항

### 웨이크워드 / KWS (완료 — BCResNet-t2 기준)

- **전처리**: LogMel spectrogram을 numpy로 구현해야 함 (torchaudio는 보드에서 무거움)
- **입력 shape**: `(1, 1, 40, 151)` NCHW, `data_format='nchw'` 필수
- **슬라이딩 윈도우**: hop=160ms, E2E 21ms → 7.5배 실시간 여유
- **후처리**: EMA(α=0.3) + Refractory(2s) 조합으로 FAR 대폭 감소
- **포팅 완료 파일**: `BCResNet-t2-npu-fixed.rknn`

### STT — Zipformer (진행 예정, `/home/rk3588/travail/rk3588/zipformer/`)

한국어 Streaming Transducer ASR. encoder + decoder + joiner 3개 모델로 구성.

**이미 준비된 것:**
- ONNX 모델 3개 (fp32 + int8): `encoder/decoder/joiner-epoch-99-avg-1.onnx`
- CPU 추론 스크립트: `zipformer_onnx_test.py` (sherpa-onnx 기반)
- RKNN Model Zoo에 동일 아키텍처 예제: `rknn_model_zoo/examples/zipformer/`

**예상 이슈:**
- `ReduceMean` encoder에 존재 → depthwise Conv 교체 (이미 아는 방법)
- `CumSum`, `Where`, `ConstantOfShape`, `Range` → NPU 미지원 가능
- Dynamic shape (입력 `?`) → RKNN은 고정 shape 필요 → Model Zoo 예제처럼 청크 단위 고정 (x: [1, 103, 80])
- encoder 입력 32개 (x + cached 상태 31개) → RKNN 캐시 상태 NCHW→NHWC 변환 필요

**시작점:** `rknn_model_zoo/examples/zipformer/python/zipformer.py` — RKNN 추론 파이프라인 전체 구현 참조

**STT 일반 주의사항 (Whisper 등에도 적용):**
- Attention/LayerNorm → 분해 또는 교체 필요
- Whisper: encoder만 NPU, decoder는 CPU 하이브리드 권장
- 참고: `rknn_model_zoo/examples/whisper/`, `rknn_model_zoo/examples/wav2vec2/`

### TTS (미래 작업)

- **아키텍처 종류**: VITS, Tacotron2, VALL-E 계열
- **예상 이슈**: Conv1d (RKNN은 Conv2d 중심), Flow 기반 연산, 가변 길이 출력
- **VITS**: 생성 모델이라 NPU 이점이 STT보다 적을 수 있음 — 프로파일링 먼저
- **참고**: RKNN Model Zoo에 mms_tts 예제 있음 (`rknn_model_zoo-main/examples/mms_tts/`)

### LLM (미래 작업)

- **현실적 한계**: RK3588 NPU 6TOPS는 대형 LLM 실행에 부족. 7B 모델은 INT4에서도 ~14GB 메모리 필요
- **실용적 범위**: 소형 모델 (1B 이하) + INT4/INT8 양자화
- **추천 도구**: RKNN-LLM (별도 툴킷, RKNN-Toolkit2와 다름), llama.cpp with RK3588 backend
- **NPU 활용**: LLM은 주로 행렬 곱셈 → NPU보다 RK3588의 CPU NEON 또는 GPU가 더 효율적일 수 있음
- **주의**: RKNN-Toolkit2로 LLM 변환하지 말 것. RKNN-LLM 별도 사용

---

## 디버깅 체크리스트

새 모델 포팅 시 순서대로 진행:

```
□ 1. ONNX 모델 구조 분석 — 사용된 연산자 목록 확인
     python -c "import onnx; m=onnx.load('model.onnx'); print(set(n.op_type for n in m.graph.node))"

□ 2. RKNN 변환 (verbose=True로 경고 확인)
     rknn = RKNN(verbose=True); rknn.load_onnx(...); rknn.build(do_quantization=False)

□ 3. 시뮬레이터 vs 실제 NPU 비교
     sim_out = ... (init_runtime())
     npu_out = ... (init_runtime(core_mask=NPU_CORE_0))
     print("sim==npu:", np.allclose(sim_out, npu_out))

□ 4. 상수 출력 확인
     out1 = inference(zeros_input)
     out2 = inference(random_input)
     print("constant:", np.allclose(out1, out2))  # True면 문제

□ 5. 서브그래프 격리 (문제 있을 때)
     — ONNX를 레이어별로 잘라서 어느 지점부터 깨지는지 이진 탐색

□ 6. ONNX 그래프 수정
     — 위 알려진 버그 목록 참조
     — 수정 후 onnxruntime으로 Max diff = 0.000000 검증

□ 7. 재변환 + NPU 재검증

□ 8. 정확도/레이턴시 측정
```

---

## 환경 정보

```
하드웨어:  RK3588 (Cortex-A76×4 + A55×4, Mali-G610, NPU 6TOPS)
OS:        Linux 5.10 (Rockchip BSP)
Python:    3.8 (conda env: RKNN-Toolkit2)
Toolkit:   RKNN-Toolkit2 v2.3.2 (변환, PC)
           rknn-toolkit-lite2 v2.3.2 cp38 aarch64 (추론, 보드)

실행 명령: conda run -n RKNN-Toolkit2 python <script>.py

NPU 코어:
  NPU_CORE_0       6.21ms (단독)
  NPU_CORE_0_1     5.59ms (듀얼, 소폭 개선)
  NPU_CORE_AUTO    6.40ms (자동)
```

---

## 참고 자료

- `rknn_model_zoo-main/` — Rockchip 공식 예제 (whisper, wav2vec2, yamnet 등 음성 모델 포함)
- `docs/RKNN_PORTING_GUIDE.md` — BCResNet-t2 포팅 상세 히스토리
- `docs/01~04_*.md` — 이 프로젝트 상세 문서
