"""
degradation.py - critical test (b): does a degraded MODERN clip look "older" to the model?

If yes, the model partly learns recording quality (resolution, grain, bandwidth) and not
football history. Images: low resolution, blur, faded colour, grain. Audio: 4 kHz low-pass + hiss.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfilt
from sklearn.base import clone

from config import RNG
from features.audio import AUDIO_COLS, audio_features_from_signal
from features.image import IMAGE_COLS, image_features_from_frames
from models import lr
from utils import log, read_frames


def degrade_frames(frames):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        g = cv2.resize(cv2.resize(f, (w // 3, h // 3), interpolation=cv2.INTER_AREA), (w, h))
        g = cv2.GaussianBlur(g, (3, 3), 0)
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 0.45
        g = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        g += RNG.normal(0, 10, g.shape)
        out.append(g.clip(0, 255).astype(np.uint8))
    return out


def degrade_audio(y, sr):
    sos = butter(6, 4000, btype="low", fs=sr, output="sos")
    z = sosfilt(sos, y)
    noise = RNG.normal(0, 1, len(z))
    p = np.mean(z ** 2) + 1e-12
    return z + noise * np.sqrt(p / 10 ** (20 / 10))  # hiss at SNR 20 dB


def degraded_clip_features(r) -> dict:
    import librosa
    frames, tiny, fps = read_frames(r.video_path)
    tiny_d = np.array([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)) for f in degrade_frames(frames)],
                      dtype=np.float32)
    img = image_features_from_frames(degrade_frames(frames), tiny if len(tiny) else tiny_d, fps)
    if r.audio_path and Path(r.audio_path).exists():
        y, sr = librosa.load(r.audio_path, sr=None, mono=True)
        aud, _ = audio_features_from_signal(degrade_audio(y, sr), sr)
    else:
        aud, _ = audio_features_from_signal(None, 44100)
    return {"clip_id": r.clip_id, **img, **aud}


def run_degradation_test(df: pd.DataFrame, eras: list[str], reg, yr, y, cache, tabdir,
                         force: bool = False, n_jobs: int = -1):
    """Degrade the clips of the most recent era and ask models trained WITHOUT that World Cup
    to date original vs degraded versions. Returns (per-clip table, summary dict)."""
    modern_era = eras[-1]
    mod = df[df.era == modern_era]
    log(f"Degradation test on {len(mod)} clips of era {modern_era}...")
    f_dg = cache / "degraded_features.csv"
    if f_dg.exists() and not force and set(mod.clip_id) <= set(pd.read_csv(f_dg).clip_id):
        dfe = pd.read_csv(f_dg)
    else:
        dfe = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(degraded_clip_features)(r)
                                                   for r in mod.itertuples()))
        dfe.to_csv(f_dg, index=False)
    dfe = dfe.set_index("clip_id").loc[mod.clip_id]
    ia = IMAGE_COLS + AUDIO_COLS
    out = []
    for yy, g in mod.groupby("edition_year"):
        trn = df.edition_year != yy
        r_m = clone(reg).fit(df.loc[trn, ia], yr[trn.values])
        c_m = clone(lr()).fit(df.loc[trn, ia], y[trn.values])
        Xo, Xd = g[ia], dfe.loc[g.clip_id, ia]
        out.append(pd.DataFrame({"clip_id": g.clip_id.values, "edition_year": yy,
                                 "pred_original": r_m.predict(Xo), "pred_degraded": r_m.predict(Xd),
                                 "era_original": c_m.predict(Xo), "era_degraded": c_m.predict(Xd)}))
    dg = pd.concat(out)
    dg.to_csv(tabdir / "degradation_test.csv", index=False)
    summary = {
        "era": modern_era, "n_clips": len(dg),
        "median_year_shift": round(float((dg.pred_degraded - dg.pred_original).median()), 1),
        "share_reclassified_older": round(float((dg.era_degraded != modern_era).mean()), 3),
        "share_original_correct": round(float((dg.era_original == modern_era).mean()), 3)}
    return dg, summary
