"""
features/audio.py - AUDIO branch: noise level (noise floor, flatness, dynamic range),
dominant pitch (pYIN f0, spectral peak, pitch class), bandwidth and MFCCs with LibROSA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import log

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LTAS_FREQS = np.geomspace(50, 20000, 120)  # long-term average spectrum grid (Hz)
AUDIO_COLS = ["loudness_db", "noise_floor_db", "dynamic_range_db", "flatness", "centroid_hz",
              "bandwidth_hz", "rolloff85_hz", "rolloff99_hz", "zcr", "hf_ratio", "voiced_ratio",
              "f0_median_hz", "peak_freq_hz"] + [f"mfcc{i}" for i in range(1, 14)]


# features that also respond to a commentator's voice (used for the "audio without voice" ablation)
VOICE_COLS = ["voiced_ratio", "f0_median_hz"]


def audio_features_from_signal(y: np.ndarray, sr: int) -> tuple[dict, np.ndarray]:
    import librosa
    nan = {k: np.nan for k in AUDIO_COLS}
    if y is None or len(y) < sr // 2 or np.abs(y).max() < 1e-6:
        return {**nan, "pitch_class": ""}, np.full(len(LTAS_FREQS), np.nan)
    eps = 1e-10
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    rms_db = 20 * np.log10(rms + eps)
    noise_floor = np.percentile(rms_db, 10)
    peak = np.percentile(rms_db, 95)
    spec = P.mean(axis=1)
    band = (freqs >= 60) & (freqs <= 4000)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y16, fmin=65, fmax=600, sr=16000, frame_length=1024)
    chroma = librosa.feature.chroma_stft(S=P, sr=sr).mean(axis=1)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=sr)),
                                n_mfcc=13).mean(axis=1)
    feats = {
        "loudness_db": 20 * np.log10(rms.mean() + eps),
        "noise_floor_db": noise_floor, "dynamic_range_db": peak - noise_floor,
        "flatness": librosa.feature.spectral_flatness(S=S).mean(),
        "centroid_hz": librosa.feature.spectral_centroid(S=S, sr=sr).mean(),
        "bandwidth_hz": librosa.feature.spectral_bandwidth(S=S, sr=sr).mean(),
        "rolloff85_hz": np.median(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)),
        "rolloff99_hz": np.median(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.99)),
        "zcr": librosa.feature.zero_crossing_rate(y)[0].mean(),
        "hf_ratio": spec[freqs > 4000].sum() / (spec.sum() + eps),
        "voiced_ratio": float(np.mean(voiced)),
        "f0_median_hz": float(np.nanmedian(f0[voiced])) if voiced.any() else np.nan,
        "peak_freq_hz": float(freqs[band][np.argmax(spec[band])]),
        **{f"mfcc{i + 1}": float(m) for i, m in enumerate(mfcc)},
    }
    ltas = np.interp(LTAS_FREQS, freqs, 10 * np.log10(spec + eps))
    ltas[LTAS_FREQS > sr / 2] = np.nan
    return {**feats, "pitch_class": PITCH_CLASSES[int(np.argmax(chroma))]}, ltas


def audio_features(path: str):
    import librosa
    if not path or not Path(path).exists():
        return audio_features_from_signal(None, 44100)
    y, sr = librosa.load(path, sr=None, mono=True)
    return audio_features_from_signal(y, sr)


def extract_audio_features(man: pd.DataFrame, cache, force: bool = False, n_jobs: int = -1):
    """Audio features + long-term average spectra for every clip (cached in cache/)."""
    f_aud, f_lt = cache / "audio_features.csv", cache / "ltas.npz"
    if f_aud.exists() and f_lt.exists() and not force and set(man.clip_id) <= set(pd.read_csv(f_aud).clip_id):
        aud = pd.read_csv(f_aud)
        ltas = dict(np.load(f_lt))
    else:
        log("Extracting AUDIO features (LibROSA, pYIN pitch tracking)...")
        res = Parallel(n_jobs=n_jobs)(delayed(audio_features)(p) for p in man.audio_path)
        aud = pd.DataFrame([r[0] for r in res]).assign(clip_id=man.clip_id.values)
        ltas = {c: r[1] for c, r in zip(man.clip_id, res)}
        aud.to_csv(f_aud, index=False)
        np.savez_compressed(f_lt, **ltas)
    log(f"  audio features: {len(AUDIO_COLS)} per clip ({aud.noise_floor_db.notna().sum()} clips with sound)")
    return aud, ltas
