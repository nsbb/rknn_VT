"""
Threshold 스윕: NPU raw probs 저장 후 최적 threshold 분석
Step1: test.csv 전체 NPU inference → probs 저장
Step2: threshold 0.1~0.99 스윕 → accuracy/precision/recall/F1
"""
import sys, os, wave, csv
import numpy as np
sys.path.insert(0, '.')
from inference_rknn import LogMel, AudioPreprocessor
from rknnlite.api import RKNNLite

PROBS_CACHE = 'npu_probs_cache.npz'

def run_inference():
    preprocessor = AudioPreprocessor()
    logmel = LogMel(apply_preemph=False)
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn('BCResNet-t2-npu-fixed.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    paths, labels, probs = [], [], []
    with open('test.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'Running NPU inference on {len(rows)} samples...')
    for i, row in enumerate(rows):
        if i % 200 == 0:
            print(f'  {i}/{len(rows)}')
        try:
            audio, sr = preprocessor.load_audio(row['path'])
            audio = preprocessor.convert_to_mono(audio)
            audio = preprocessor.resample(audio, sr)
            feat = logmel(audio)[np.newaxis, np.newaxis, ...]
            out = rknn.inference(inputs=[feat], data_format='nchw')[0]
            p = float(np.exp(out[0,1]) / (np.exp(out[0,0]) + np.exp(out[0,1])))
            probs.append(p)
            labels.append(int(row['label']))
            paths.append(row['path'])
        except Exception as e:
            print(f'  Error {row["path"]}: {e}')
            probs.append(0.5)
            labels.append(int(row['label']))
            paths.append(row['path'])
    rknn.release()

    probs = np.array(probs)
    labels = np.array(labels)
    np.savez(PROBS_CACHE, probs=probs, labels=labels)
    print(f'Saved {PROBS_CACHE}')
    return probs, labels

def sweep(probs, labels):
    thresholds = np.arange(0.05, 1.0, 0.01)
    best_f1 = 0; best_t = 0.5
    print(f'\n{"Thresh":>7} {"Acc%":>7} {"Prec%":>7} {"Rec%":>7} {"F1%":>7} {"TN":>5} {"FP":>5} {"FN":>5} {"TP":>5}')
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn = int(((pred==0)&(labels==0)).sum())
        fp = int(((pred==1)&(labels==0)).sum())
        fn = int(((pred==0)&(labels==1)).sum())
        tp = int(((pred==1)&(labels==1)).sum())
        acc = (tn+tp)/(tn+fp+fn+tp)*100
        prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
        rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        if f1 > best_f1:
            best_f1 = f1; best_t = t
        if t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] or abs(t-best_t)<0.015:
            print(f'{t:>7.2f} {acc:>7.2f} {prec:>7.2f} {rec:>7.2f} {f1:>7.2f} {tn:>5} {fp:>5} {fn:>5} {tp:>5}')
    print(f'\nBest F1={best_f1:.2f}% at threshold={best_t:.2f}')

if os.path.exists(PROBS_CACHE):
    data = np.load(PROBS_CACHE)
    probs, labels = data['probs'], data['labels']
    print(f'Loaded cached probs ({len(probs)} samples)')
else:
    probs, labels = run_inference()

sweep(probs, labels)
