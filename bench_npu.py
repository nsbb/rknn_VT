"""NPU 레이턴시 벤치마크 (단일/멀티코어)"""
import sys, time, wave
import numpy as np
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel(apply_preemph=False)(audio)[np.newaxis, np.newaxis, ...]

cores = [
    ('NPU_CORE_0',    RKNNLite.NPU_CORE_0),
    ('NPU_CORE_AUTO', RKNNLite.NPU_CORE_AUTO),
    ('NPU_CORE_0_1',  RKNNLite.NPU_CORE_0_1),
]

print(f'Input shape: {feat.shape}')
for name, mask in cores:
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn('BCResNet-t2-npu-fixed.rknn')
    rknn.init_runtime(core_mask=mask)
    # warmup
    for _ in range(10):
        rknn.inference(inputs=[feat], data_format='nchw')
    # bench
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        rknn.inference(inputs=[feat], data_format='nchw')
    ms = (time.perf_counter() - t0) / N * 1000
    print(f'{name}: {ms:.2f} ms/call  ({1000/ms:.0f} infer/sec)')
    rknn.release()
