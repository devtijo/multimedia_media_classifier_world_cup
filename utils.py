"""
utils.py - small helpers used by several modules: logging, video reading, FDR, plot style.
"""
from __future__ import annotations

import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no window: figures are only saved to disk
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from config import RC_PARAMS  # noqa: E402


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(RC_PARAMS)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_frames(path: str, sample_fps: float = 2.0, width: int = 320):
    """Sampled frames (BGR, fixed width) + tiny grayscale copies of EVERY frame."""
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native / sample_fps)))
    frames, tiny, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        tiny.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36),
                               interpolation=cv2.INTER_AREA))
        if i % step == 0:
            h, w = f.shape[:2]
            frames.append(cv2.resize(f, (width, max(1, int(round(h * width / w)))),
                                     interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    return frames, np.array(tiny, dtype=np.float32), float(native)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (q-values)."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out
