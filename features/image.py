"""
features/image.py - IMAGE branch: colour (saturation, colourfulness, palette, pitch green)
and patterns (sharpness, grain, edges, texture, frame format, duplicated frames, editing pace).
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import RNG
from utils import log, read_frames

IMAGE_COLS = ["saturation", "brightness", "contrast", "colorfulness", "hue_entropy", "green_ratio",
              "sharpness_log", "grain", "edge_density", "orientation_entropy", "hf_energy",
              "pillarbox", "letterbox", "motion", "dup_frames", "cut_rate"]


def content_box(frames):
    """Detect black pillar/letter-box bars (old 4:3 footage inside a 16:9 frame)."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    H, W = g.shape[1:]
    cols = np.percentile(g, 98, axis=(0, 1)) > 24
    rows = np.percentile(g, 98, axis=(0, 2)) > 24
    if cols.sum() < W * 0.2 or rows.sum() < H * 0.2:
        return 0, H, 0, W, 0.0, 0.0
    x = np.flatnonzero(cols); y = np.flatnonzero(rows)
    x0, x1, y0, y1 = x[0], x[-1] + 1, y[0], y[-1] + 1
    return y0, y1, x0, x1, 1 - (x1 - x0) / W, 1 - (y1 - y0) / H


def _entropy(hist):
    p = hist / hist.sum() if hist.sum() > 0 else hist
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(len(hist))) if len(p) else 0.0


def frame_features(f) -> dict:
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    b, g, r = [c.astype(np.float32) for c in cv2.split(f)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    colored = s > 40
    hue_ent = _entropy(np.histogram(h[colored], bins=18, range=(0, 180))[0].astype(float)) \
        if colored.mean() > 0.02 else 0.0
    green = ((h >= 35) & (h <= 85) & (s > 60) & (v > 40)).mean()
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    grain = (gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)).std()
    edges = (cv2.Canny(gray, 100, 200) > 0).mean()
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    orient_ent = _entropy(np.histogram(ang % np.pi, bins=18, range=(0, np.pi), weights=mag)[0])
    F = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean()))) ** 2
    yy, xx = np.indices(F.shape)
    rad = np.hypot((yy - F.shape[0] / 2) / (F.shape[0] / 2), (xx - F.shape[1] / 2) / (F.shape[1] / 2))
    hf = F[rad > 0.5].sum() / (F.sum() + 1e-9)
    return {
        "saturation": s.mean() / 255, "brightness": v.mean() / 255,
        "contrast": gray.std() / 255, "colorfulness": colorful,
        "hue_entropy": hue_ent, "green_ratio": green,
        "sharpness_log": np.log1p(lap), "grain": grain, "edge_density": edges,
        "orientation_entropy": orient_ent, "hf_energy": hf,
    }


def palette(f, k: int = 4):
    """Dominant colours of one frame (k-means): [(hex, share), ...] sorted by share."""
    px = f.reshape(-1, 3).astype(np.float32)
    px = px[RNG.choice(len(px), size=min(4000, len(px)), replace=False)]
    _, lab, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=k) / len(lab)
    order = np.argsort(-share)
    return [("#%02x%02x%02x" % tuple(int(c) for c in cen[i][::-1]), float(share[i])) for i in order]


def image_features_from_frames(frames, tiny, native_fps) -> dict:
    if not frames:
        return {}
    y0, y1, x0, x1, side, topbot = content_box(frames)
    crops = [f[y0:y1, x0:x1] for f in frames]
    per = pd.DataFrame([frame_features(c) for c in crops]).median().to_dict()
    d = np.abs(np.diff(tiny, axis=0)).mean(axis=(1, 2)) / 255 if len(tiny) > 1 else np.array([0.0])
    med = np.median(d)
    dur = len(tiny) / native_fps
    per.update({
        "pillarbox": side, "letterbox": topbot,
        "motion": float(med), "dup_frames": float((d < 0.002).mean()),
        "cut_rate": float(((d > 0.12) & (d > 4 * (med + 1e-4))).sum() / max(dur, 1e-6)),
        "palette": json.dumps(palette(crops[len(crops) // 2])),
    })
    return per


def image_features(path: str) -> dict:
    frames, tiny, fps = read_frames(path)
    return image_features_from_frames(frames, tiny, fps)


def extract_image_features(man: pd.DataFrame, cache, force: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    """Image features for every clip in the manifest (cached in cache/image_features.csv)."""
    f_img = cache / "image_features.csv"
    if f_img.exists() and not force and set(man.clip_id) <= set(pd.read_csv(f_img).clip_id):
        img = pd.read_csv(f_img)
    else:
        log("Extracting IMAGE features (OpenCV)...")
        res = Parallel(n_jobs=n_jobs)(delayed(image_features)(p) for p in man.video_path)
        img = pd.DataFrame(res).assign(clip_id=man.clip_id.values)
        img.to_csv(f_img, index=False)
    log(f"  image features: {len(IMAGE_COLS)} per clip")
    return img
