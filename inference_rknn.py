import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import deque
import time
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import gc
import wave
import glob
from typing import Iterator, Tuple, Optional, Dict, Any, Callable, List

# RKNN Runtime
# 실기기(RK3588)에서는 rknn-toolkit-lite2 사용이 권장됩니다.
try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    try:
        from rknn.api import RKNN
        is_lite = False
    except ImportError:
        print("ERROR: RKNN-Toolkit2 or RKNN-Toolkit-Lite2 not found.")
        sys.exit(1)

class AudioPreprocessor:
    """Design Spec 3: Audio Preprocessor"""
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        with wave.open(audio_path, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 1:
                dtype = np.uint8
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            audio = np.frombuffer(data, dtype=dtype)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128.0) / 128.0
                
        return audio, sr

    def convert_to_mono(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        return waveform

    def resample(self, waveform: np.ndarray, source_sr: int) -> np.ndarray:
        if source_sr != self.target_sr:
            # [수정] scipy.signal.resample 대신 numpy.interp (선형 보간) 사용
            duration = len(waveform) / source_sr
            num_samples = int(duration * self.target_sr)
            old_indices = np.linspace(0, duration, len(waveform))
            new_indices = np.linspace(0, duration, num_samples)
            waveform = np.interp(new_indices, old_indices, waveform)
        return waveform

    def pad_or_truncate(self, waveform: np.ndarray, target_length: int = 24000) -> np.ndarray:
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
        return waveform

    def load_and_preprocess(self, audio_path: str) -> np.ndarray:
        waveform, sr = self.load_audio(audio_path)
        waveform = self.convert_to_mono(waveform)
        waveform = self.resample(waveform, sr)
        return waveform

class LogMel:
    """Design Spec 4: LogMel Feature Extractor"""
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        win_length: int = 480,
        n_fft: int = 512,
        n_mels: int = 40,
        apply_preemph: bool = False
    ):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.apply_preemph = apply_preemph
        self.mel_basis = self._create_mel_filterbank()
        self.window = np.hanning(win_length)

    def _create_mel_filterbank(self):
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)
        all_freqs = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sr / 2), self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left, center, right = hz_points[i:i+3]
            for j, f in enumerate(all_freqs):
                if left < f < center:
                    filterbank[i, j] = (f - left) / (center - left)
                elif center <= f < right:
                    filterbank[i, j] = (right - f) / (right - center)
        return filterbank

    def apply_preemphasis(self, waveform: np.ndarray) -> np.ndarray:
        return np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        # torchaudio.transforms.MelSpectrogram(center=True)와 동일하게 패딩
        # center=True일 경우 토치오디오는 양 끝에 각각 n_fft // 2 만큼 패딩합니다.
        pad_size = self.n_fft // 2
        waveform = np.pad(waveform, (pad_size, pad_size), mode='reflect')
        
        frames = []
        for i in range(0, len(waveform) - self.n_fft + 1, self.hop_length):
            chunk = waveform[i:i+self.win_length]
            if len(chunk) < self.n_fft:
                chunk = np.pad(chunk, (0, self.n_fft - len(chunk)), mode='constant')
            
            chunk = chunk * np.pad(self.window, (0, self.n_fft - len(self.window)), mode='constant')
            spec = np.abs(np.fft.rfft(chunk, n=self.n_fft))**2
            frames.append(spec)
            
        spectrogram = np.array(frames).T
        mel_spec = np.dot(self.mel_basis, spectrogram)
        return mel_spec

    def apply_log_transform(self, mel_spec: np.ndarray) -> np.ndarray:
        # Pytorch (inference.py) uses .log() which is natural logarithm
        return np.log(mel_spec + 1e-6)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        if self.apply_preemph:
            waveform = self.apply_preemphasis(waveform)
        
        # [수정] 입력 길이를 모델 기대치(24000샘플 = 1.5초)로 고정합니다.
        # 이렇게 해야 항상 151프레임(1, 1, 40, 151)이 생성됩니다.
        target_length = 24000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')

        mel_spec = self.compute_mel_spectrogram(waveform)
        log_mel = self.apply_log_transform(mel_spec)
        return log_mel.astype(np.float32)

class SlidingWindowProcessor:
    """Design Spec 5: Sliding Window Processor"""
    def __init__(self, win_sec: float = 1.5, shift_sec: float = 0.2, sr: int = 16000):
        self.win_samples = int(win_sec * sr)
        self.hop_samples = int(shift_sec * sr)
        self.sr = sr

    def sliding_windows(self, audio: np.ndarray) -> Iterator[Tuple[float, float, np.ndarray]]:
        for start in range(0, len(audio), self.hop_samples):
            end = start + self.win_samples
            if end > len(audio):
                chunk = np.pad(audio[start:], (0, end - len(audio)), mode='constant')
            else:
                chunk = audio[start:end]
            yield start/self.sr, end/self.sr, chunk

class RKNNInferenceEngine:
    """Design Spec 2: RKNN Inference Engine"""
    def __init__(self, model_path: str, target: Optional[str] = None):
        self.rknn = RKNN()
        self.model_path = model_path
        self.target = target # 'rk3588' 등을 지정하면 ADB를 통해 기기 연결 시도

    def load_model(self) -> bool:
        print(f"--> Loading RKNN model: {self.model_path}")
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print("ERROR: load_rknn failed.")
            return False
            
        # [수정] 
        # rknnlite(is_lite=True)는 target 인자가 없음.
        # rknn-toolkit2(is_lite=False)는 실기기에서 돌릴 때 target을 명시하거나 설정을 맞춰야 함.
        print(f"--> Initializing runtime (is_lite={is_lite})")
        if is_lite:
            ret = self.rknn.init_runtime()
        else:
            # Full Toolkit을 보드에서 돌릴 때는 target='rk3588' 또는 None을 명시적으로 처리
            try:
                ret = self.rknn.init_runtime(target=self.target)
            except:
                ret = self.rknn.init_runtime()
                
        if ret != 0:
            print(f"ERROR: init_runtime failed with ret={ret}")
            return False
        return True

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.rknn.inference(inputs=[input_tensor], data_format='nchw')
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def batch_infer(self, input_tensors: List[np.ndarray]) -> List[np.ndarray]:
        results = []
        for tensor in input_tensors:
            results.append(self.infer(tensor))
        return results

    def release(self):
        self.rknn.release()

class FAREvaluator:
    """Design Spec 6: FAR Evaluator"""
    def __init__(self, threshold=0.5, refrac_sec=2.0, ema_alpha=0.3, n_n=3, n_m=5):
        self.threshold = threshold
        self.refrac_sec = refrac_sec
        self.ema_alpha = ema_alpha
        self.n_n = n_n
        self.n_m = n_m

    def update_ema(self, p: float, p_ema: Optional[float]) -> float:
        if p_ema is None: return p
        return self.ema_alpha * p + (1 - self.ema_alpha) * p_ema

    def check_n_of_m(self, trigbuf: deque) -> bool:
        return len(trigbuf) == self.n_m and sum(trigbuf) >= self.n_n

    def check_trigger(self, p: float) -> bool:
        return p >= self.threshold

    def reset_state(self):
        return None, deque(maxlen=self.n_m)

    def evaluate_all_configs(self, audio: np.ndarray, engine: RKNNInferenceEngine, logmel: LogMel, configs: List[Dict[str, Any]]):
        processor = SlidingWindowProcessor(sr=16000)
        
        # 1. 모든 윈도우에 대해 추론을 먼저 수행 (1회만 진행)
        all_raw_probs = []
        windows = list(processor.sliding_windows(audio))
        for start_t, end_t, chunk in windows:
            feat = logmel(chunk)[np.newaxis, np.newaxis, ...]
            probs = engine.infer(feat)
            all_raw_probs.append(float(probs[0, 1]))
        
        all_raw_probs = np.array(all_raw_probs)
        results = []

        # 2. 각 설정별로 스무딩 및 트리거 로직만 빠르게 적용
        for cfg in configs:
            smooth_probs = []
            triggers = []
            p_ema, trigbuf = self.reset_state()
            
            refrac_sec = cfg.get("REFRACTORY_SEC", self.refrac_sec)
            cooldown_until = -1e9
            
            for i, (start_t, end_t, _) in enumerate(windows):
                p = all_raw_probs[i]
                
                p_s = self.update_ema(p, p_ema) if cfg.get("USE_EMA", False) else p
                if cfg.get("USE_EMA", False): p_ema = p_s
                smooth_probs.append(p_s)
                
                hit = self.check_trigger(p_s)
                fired = self.check_n_of_m(trigbuf) if cfg.get("USE_N_OF_M", False) else hit
                if cfg.get("USE_N_OF_M", False):
                    trigbuf.append(1 if hit else 0)
                    fired = self.check_n_of_m(trigbuf)

                if start_t >= cooldown_until and fired:
                    triggers.append((start_t, end_t, p_s))
                    cooldown_until = start_t + refrac_sec
                    p_ema, trigbuf = self.reset_state()
            
            results.append((all_raw_probs, np.array(smooth_probs), triggers))
            
        return results

class AccuracyEvaluator:
    """Design Spec 7: Accuracy Evaluator"""
    def evaluate_test_set(self, test_csv_path: str, engine: RKNNInferenceEngine, logmel: LogMel, preprocessor: AudioPreprocessor):
        df = self.load_test_data(test_csv_path)
        predictions, labels = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval Accuracy"):
            audio = preprocessor.load_and_preprocess(row['path'])
            audio = preprocessor.pad_or_truncate(audio)
            feat = logmel(audio)[np.newaxis, np.newaxis, ...]
            probs = engine.infer(feat)
            pred = np.argmax(probs)
            predictions.append(pred)
            labels.append(row['label'])
            
        metrics = self.calculate_metrics(np.array(labels), np.array(predictions))
        return metrics, df.assign(pred=predictions)

    def load_test_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def calculate_metrics(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred)
        cm = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Pred'))
        return {"accuracy": acc, "confusion_matrix": cm}

    def save_results(self, metrics, predictions_df, path):
        predictions_df.to_csv(path, index=False)

class VisualizationGenerator:
    """Design Spec 8: Visualization Generator"""
    def plot_far_result(self, audio: np.ndarray, sr: int, far_result: Tuple, config: Dict[str, Any], save_path: str):
        raw_p, smooth_p, triggers = far_result
        times = np.linspace(0, len(audio) / sr, len(raw_p))
        audio_times = np.linspace(0, len(audio) / sr, len(audio))

        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax1.plot(audio_times, audio, color='gray', alpha=0.3, label='Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        ax2 = ax1.twinx()
        ax2.plot(times, raw_p, color='red', alpha=0.5, linestyle=':', label='Raw Probability')
        if config.get("USE_EMA", False):
            ax2.plot(times, smooth_p, color='orange', linewidth=2, label='Smoothed (EMA) Probability')
            
        ax2.axhline(0.5, color='blue', linestyle='--', linewidth=1, label='Threshold (0.5)')
        ax2.set_ylabel('Probability', color='black')
        ax2.set_ylim(-0.05, 1.05)

        for i, (t_start, t_end, p_val) in enumerate(triggers):
            label = 'Trigger' if i == 0 else ""
            ax2.plot(t_start, p_val, 'g^', markersize=12, label=label)

        plt.title(f"Configuration: {config.get('name', 'Unknown')} - Triggers Detected: {len(triggers)}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

def main():
    model_path = 'BCResNet-t2-npu-fixed.rknn'
    test_csv = 'test.csv'
    
    # [수정] RK3588 보드에서 full toolkit을 사용할 경우 target='rk3588' 명시가 필요할 수 있습니다.
    engine = RKNNInferenceEngine(model_path, target='rk3588') 
    if not engine.load_model():
        return
        
    preprocessor = AudioPreprocessor()
    logmel = LogMel(apply_preemph=False)
    
    # 1. Accuracy Evaluation
    if os.path.exists(test_csv):
        evaluator = AccuracyEvaluator()
        metrics, results_df = evaluator.evaluate_test_set(test_csv, engine, logmel, preprocessor)
        print(f"\n📊 Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print("Confusion Matrix:\n", metrics['confusion_matrix'])
    
    # 2. FAR Evaluation (Multi-directory support)
    # [수정] 대용량 파일이 포함된 'measure_FA' 제외
    far_dirs = ['vad_cropped', 'wallpad_HiWonder_251113']
    audio_path_list = []
    for d in far_dirs:
        if os.path.exists(d):
            found_files = glob.glob(os.path.join(d, '**', '*.wav'), recursive=True)
            audio_path_list.extend(found_files)
    
    # Add root files if exist
    if os.path.exists('test_long_audio.wav'):
        audio_path_list.append('test_long_audio.wav')

    if not audio_path_list:
        print("NOTE: No audio files found for FAR evaluation. Skipping.")
        engine.release()
        return

    print(f"\n--> Running FAR Evaluation on {len(audio_path_list)} files...")
    far_evaluator = FAREvaluator()
    viz_gen = VisualizationGenerator()
    
    # 4가지 설정 옵션 (inference.py와 동일)
    configs = [
        {"name": "Raw (no processing)", "REFRACTORY_SEC": 0.0, "USE_EMA": False, "USE_N_OF_M": False},
        {"name": "Refractory only", "REFRACTORY_SEC": 2.0, "USE_EMA": False, "USE_N_OF_M": False},
        {"name": "Refractory + EMA", "REFRACTORY_SEC": 2.0, "USE_EMA": True, "USE_N_OF_M": False},
        {"name": "Refractory + EMA + N-of-M", "REFRACTORY_SEC": 2.0, "USE_EMA": True, "USE_N_OF_M": True}
    ]
    
    total_results = {i: {"false_alarms": 0, "duration": 0.0} for i in range(len(configs))}
    
    for audio_path in tqdm(audio_path_list, desc="FAR Evaluating Files"):
        try:
            audio_data, sr = preprocessor.load_audio(audio_path)
            audio_data = preprocessor.convert_to_mono(audio_data)
            audio_data = preprocessor.resample(audio_data, sr)
            file_duration = len(audio_data) / 16000.0
            
            # 모든 설정을 한 번의 추론 스트림으로 평가
            results = far_evaluator.evaluate_all_configs(audio_data, engine, logmel, configs)
            
            for idx, (raw_p, smooth_p, triggers) in enumerate(results):
                # 통계 누적
                total_results[idx]["false_alarms"] += len(triggers)
                total_results[idx]["duration"] += file_duration
                
                # 큰 파일(예: measure_FA)만 플롯 생성하여 리소스 절약
                if file_duration > 10.0 and HAS_MATPLOTLIB:
                    cfg = configs[idx]
                    folder_name = os.path.dirname(audio_path).replace(os.path.sep, '_')
                    file_name = os.path.basename(audio_path)
                    save_path = f"far_{idx+1}_{cfg['name']}_{folder_name}_{file_name}.png"
                    viz_gen.plot_far_result(audio_data, 16000, (raw_p, smooth_p, triggers), cfg, save_path)
                elif file_duration > 10.0 and not HAS_MATPLOTLIB:
                    if idx == 0: # 한 번만 출력
                        print("\nNOTE: matplotlib not found. Skipping visualization.")
                    
            del audio_data
            gc.collect()
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")

    # 3. FAR Summary (per hour)
    print(f"\n{'='*60}")
    print("📊 FAR Summary (per hour) for each configuration:")
    print(f"{'='*60}")
    for idx, cfg in enumerate(configs):
        fa = total_results[idx]["false_alarms"]
        dur = total_results[idx]["duration"]
        far_hr = (fa / dur) * 3600.0 if dur > 0 else 0
        print(f"  [{idx+1}] {cfg['name']}")
        print(f"      False Alarms: {fa}, Duration: {dur/60:.1f} min, FAR: {far_hr:.2f}/hour")
    print(f"{'='*60}")
        
    engine.release()

if __name__ == '__main__':
    main()
