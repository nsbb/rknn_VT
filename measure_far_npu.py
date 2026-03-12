"""
measure_FA/ 기준 정밀 FAR 측정 (비웨이크워드 배경음, 총 ~118분)
threshold 여러 값에서 FAR/hr 계산
"""
import sys, os, wave
import numpy as np
sys.path.insert(0, '.')
from inference_rknn import LogMel, AudioPreprocessor, SlidingWindowProcessor
from rknnlite.api import RKNNLite

FA_DIR = 'measure_FA'
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
EMA_ALPHA = 0.3
REFRAC_SEC = 2.0
N_N, N_M = 3, 5

preprocessor = AudioPreprocessor()
logmel = LogMel(apply_preemph=False)
rknn = RKNNLite(verbose=False)
rknn.load_rknn('BCResNet-t2-npu-fixed.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
processor = SlidingWindowProcessor(sr=16000)

files = [os.path.join(FA_DIR, f) for f in os.listdir(FA_DIR) if f.endswith('.wav')]
files.sort()

total_duration = 0.0
# threshold → FA count
fa_raw   = {t: 0 for t in THRESHOLDS}
fa_refr  = {t: 0 for t in THRESHOLDS}
fa_ema   = {t: 0 for t in THRESHOLDS}
fa_nom   = {t: 0 for t in THRESHOLDS}

for fpath in files:
    print(f'Processing: {os.path.basename(fpath)[:60]}')
    audio, sr = preprocessor.load_audio(fpath)
    audio = preprocessor.convert_to_mono(audio)
    audio = preprocessor.resample(audio, sr)
    dur = len(audio) / 16000.0
    total_duration += dur
    print(f'  Duration: {dur/60:.1f} min, Computing windows...')

    # 모든 윈도우 inference (1회)
    windows = list(processor.sliding_windows(audio))
    probs = []
    for i, (st, et, chunk) in enumerate(windows):
        if i % 5000 == 0:
            print(f'  Window {i}/{len(windows)}')
        feat = logmel(chunk)[np.newaxis, np.newaxis, ...]
        out = rknn.inference(inputs=[feat], data_format='nchw')[0].squeeze()
        p = float(np.exp(out[1]) / (np.exp(out[0]) + np.exp(out[1])))
        probs.append(p)
    probs = np.array(probs)
    print(f'  Done. prob range=[{probs.min():.3f},{probs.max():.3f}]')

    # 각 threshold × 설정 조합으로 FA 집계
    for t in THRESHOLDS:
        # 1. Raw
        fa_raw[t] += int((probs >= t).sum())

        # 2. Refractory
        cooldown = -1e9
        for i, (st, et, _) in enumerate(windows):
            if probs[i] >= t and st >= cooldown:
                fa_refr[t] += 1
                cooldown = st + REFRAC_SEC

        # 3. Refractory + EMA
        cooldown = -1e9; p_ema = None
        for i, (st, et, _) in enumerate(windows):
            p_ema = EMA_ALPHA * probs[i] + (1-EMA_ALPHA) * (p_ema if p_ema is not None else probs[i])
            if p_ema >= t and st >= cooldown:
                fa_ema[t] += 1
                cooldown = st + REFRAC_SEC

        # 4. Refrac + EMA + N-of-M
        from collections import deque
        cooldown = -1e9; p_ema = None; buf = deque(maxlen=N_M)
        for i, (st, et, _) in enumerate(windows):
            p_ema = EMA_ALPHA * probs[i] + (1-EMA_ALPHA) * (p_ema if p_ema is not None else probs[i])
            buf.append(1 if p_ema >= t else 0)
            if len(buf) == N_M and sum(buf) >= N_N and st >= cooldown:
                fa_nom[t] += 1
                cooldown = st + REFRAC_SEC

    del audio, probs

rknn.release()

total_hr = total_duration / 3600.0
print(f'\nTotal duration: {total_duration/60:.1f} min ({total_hr:.2f} hr)')
print(f'\n{"Thresh":>7} {"Raw FA/hr":>10} {"Refr FA/hr":>11} {"EMA FA/hr":>10} {"NoM FA/hr":>10}')
print('-'*55)
for t in THRESHOLDS:
    print(f'{t:>7.2f} {fa_raw[t]/total_hr:>10.2f} {fa_refr[t]/total_hr:>11.2f} {fa_ema[t]/total_hr:>10.2f} {fa_nom[t]/total_hr:>10.2f}')
