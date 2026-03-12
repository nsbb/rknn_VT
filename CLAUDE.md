# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wake word / keyword spotting pipeline targeting the Rockchip RK3588 SoC. The model (BCResNet-t2, tau=2, Focal loss) is trained in PyTorch, exported to ONNX, then converted to RKNN format for on-device inference.

## Common Commands

```bash
# Evaluate ONNX model on test set + FAR analysis (requires PyTorch + torchaudio)
python inference.py

# Convert ONNX → RKNN with validation report (run on PC with rknn-toolkit2)
python convert_to_rknn.py

# Generic ONNX → RKNN conversion
python convert.py BCResNet-t2-Focal-ep110.onnx rk3588 fp

# Run RKNN inference on RK3588 board (uses rknn-toolkit-lite2)
python inference_rknn.py

# Validate ONNX vs RKNN numerical consistency
python compare_onnx_rknn.py

# Validate numpy LogMel (inference_rknn.py) vs torchaudio LogMel (inference.py)
python compare_logmel.py
```

## Architecture

### Model & Tensor Format
- **Model**: BCResNet-t2, 2-class classifier (wake word = class 1, non-wake = class 0)
- **Input shape**: `(1, 1, 40, 151)` — NCHW: batch × channel × mel_bins × time_frames
- **Audio params**: 16 kHz, 1.5 s window → 24000 samples → 151 frames (hop=160, win=480, FFT=512, 40 mel bins, center-padded)
- **RKNN inference always requires `data_format='nchw'`**

### Two-Environment Inference
| Environment | Script | Runtime | Feature extraction |
|---|---|---|---|
| PC / development | `inference.py` | onnxruntime | `torchaudio.transforms.MelSpectrogram` |
| RK3588 board | `inference_rknn.py` | `rknnlite.api.RKNNLite` (preferred) or `rknn.api.RKNN` | Pure-numpy `LogMel` class |

`inference_rknn.py` auto-detects which toolkit is installed (lite2 first, then full toolkit).

### FAR Evaluation Configurations
Both `inference.py` and `inference_rknn.py` evaluate four post-processing configurations:
1. **Raw** — raw probability threshold
2. **Refractory only** — 2 s cooldown after each trigger
3. **Refractory + EMA** — exponential moving average smoothing (α=0.3) + refractory
4. **Refractory + EMA + N-of-M** — N=3 of last M=5 windows must fire + EMA + refractory

### Key Files
- `inference.py` — ONNX evaluation: accuracy on `test.csv` + FAR on `measure_FA/` audio
- `inference_rknn.py` — RKNN evaluation: same logic, numpy-only, runs on board
- `convert_to_rknn.py` — `ModelConverter` class: config → load → build → export → validate → JSON report
- `convert.py` — Minimal RKNN converter (used as a utility template)
- `compare_onnx_rknn.py` — Quick numerical sanity check between ONNX and RKNN outputs
- `compare_logmel.py` — Checks that numpy and torchaudio LogMel outputs match
- `pad_check*.py` — Iterative scripts investigating padding/frame-count edge cases

### Data Layout
- `test.csv` — columns: `path`, `label` (0 or 1)
- `wallpad_HiWonder_251113/<speaker>/` — per-speaker WAV clips (`<speaker>_<label>_<idx>.wav`)
- `vad_cropped/` — VAD-segmented clips for FAR evaluation
- `measure_FA/` — long recordings (TV news, etc.) for false alarm rate measurement

### RKNN Conversion Notes
- Target platform: `rk3588`
- Default quantization: `fp16` (pass `do_quantization=False` for FP32/FP16, `True` for INT8)
- `mean_values=[[0]], std_values=[[1]]` are set explicitly because the input is already a log-mel spectrogram (not raw pixels)
- **Do NOT import `rknn.api` and `rknnlite.api` in the same process** — they conflict

### RKNN Graph Fix (NPU Porting)

The original ONNX model requires 3 graph transformations before RKNN conversion:

1. **ReduceMean → depthwise Conv** (all 12+1 ReduceMean nodes):
   - `ReduceMean(axis=2)` on `(1,C,H,W)` → depthwise Conv `(group=C, kernel=(H,1), weight=1/H)`
   - `ReduceMean(axes=[2,3])` → depthwise Conv `(group=C, kernel=(1,W), weight=1/W)`

2. **Pad H=1→4 before f1 branch** (each BCBlock):
   - `(1,C,1,W)` → Pad `[0,0,0,0, 0,0,3,0]` → `(1,C,4,W)`
   - Needed because RKNN NPU fails to run depthwise Conv(1×3) on H=1 intermediate tensors

3. **Slice + Expand after f1 branch** (each BCBlock):
   - After f1.1.Conv: `(1,C,4,W)` → Slice(axis=2, rows 0:1) → `(1,C,1,W)` → Expand → `(1,C,H_f2,W)`
   - Expand avoids RKNN's fused AddRelu broadcast bug `(1,C,1,W)+(1,C,H,W)`

**Fixed model files**:
- `BCResNet-t2-npu-fixed.onnx` — modified ONNX (ONNX output unchanged: `Max diff: 0.000000`)
- `BCResNet-t2-npu-fixed.rknn` — converted RKNN (NPU pred=1 matches ONNX pred=1 ✓)

**Workflow**:
```bash
# Regenerate fixed ONNX + validate
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py

# Convert to RKNN
conda run -n RKNN-Toolkit2 python convert_fixed_only.py

# Quick NPU sanity check
conda run -n RKNN-Toolkit2 python test_npu_fixed.py
```

**필독**: 새 세션에서 이 프로젝트를 이어받을 때 반드시 먼저 읽을 것:
- `docs/RKNN_PORTING_GUIDE.md` — 실패한 접근 방식, RKNN 버그 목록, 올바른 디버깅 순서
- `docs/HANDOVER.md` — 전체 진단 과정 히스토리
