"""엔드투엔드 레이턴시 측정 (LogMel + NPU)"""
import sys, time, wave
import numpy as np
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
# 1.5초 window (24000 samples)
audio_15s = audio[:24000] if len(audio) >= 24000 else np.pad(audio, (0, 24000 - len(audio)))

logmel = LogMel(apply_preemph=False)
rknn = RKNNLite(verbose=False)
rknn.load_rknn('BCResNet-t2-npu-fixed.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# warmup
for _ in range(10):
    feat = logmel(audio_15s)[np.newaxis, np.newaxis, ...]
    rknn.inference(inputs=[feat], data_format='nchw')

N = 300

# LogMel only
t0 = time.perf_counter()
for _ in range(N):
    feat = logmel(audio_15s)[np.newaxis, np.newaxis, ...]
logmel_ms = (time.perf_counter() - t0) / N * 1000

# NPU only (pre-computed feat)
feat = logmel(audio_15s)[np.newaxis, np.newaxis, ...]
t0 = time.perf_counter()
for _ in range(N):
    rknn.inference(inputs=[feat], data_format='nchw')
npu_ms = (time.perf_counter() - t0) / N * 1000

# End-to-end
t0 = time.perf_counter()
for _ in range(N):
    feat = logmel(audio_15s)[np.newaxis, np.newaxis, ...]
    rknn.inference(inputs=[feat], data_format='nchw')
e2e_ms = (time.perf_counter() - t0) / N * 1000

rknn.release()

print(f'LogMel only:   {logmel_ms:.2f} ms')
print(f'NPU only:      {npu_ms:.2f} ms')
print(f'End-to-End:    {e2e_ms:.2f} ms')
print(f'Max rate:      {1000/e2e_ms:.0f} calls/sec')
print(f'Hop 160ms -> {160/e2e_ms:.1f}x realtime margin')
