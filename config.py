"""
config.py - constants shared by every module: seed, paths, colours, plot style, labels.

Nothing here does any work; it is safe to import from anywhere.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------- reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)  # one shared generator (k-means sampling, bootstraps, noise)

# ----------------------------------------------------------------------- default paths
# Relative to the folder you run the script from (where clips_manifest.csv lives).
DEFAULT_MANIFEST = Path("clips_manifest.csv")
DEFAULT_VIDEOS_DIR = Path("videos")
DEFAULT_OUT = Path("outputs")
DEFAULT_CACHE = Path("cache")

# ----------------------------------------------------------------------- colours
# Eras are ORDERED, so they use one sequential hue (light = old, dark = recent);
# era names are always on the axis or in a legend, never colour alone.
ERA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MODALITY_COLORS = {"Image": "#2a78d6", "Audio": "#eb6834", "Text": "#1baf7a",
                   "Fusion": "#6250d6", "Baseline": "#a3a29b"}
INK, INK2 = "#0b0b0b", "#52514e"

# ----------------------------------------------------------------------- matplotlib style
RC_PARAMS = {
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "grid.color": "#ecebe7",
    "grid.linewidth": 0.8, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "legend.frameon": False,
}

# ----------------------------------------------------------------------- readable feature names
FEATURE_LABELS = {
    "saturation": "Saturation (0-1)", "colorfulness": "Colourfulness (Hasler)",
    "green_ratio": "Pitch green share", "sharpness_log": "Sharpness (log Laplacian var.)",
    "grain": "Film grain / noise", "edge_density": "Edge density",
    "hue_entropy": "Hue diversity (entropy)", "pillarbox": "Side black bars (4:3 format)",
    "dup_frames": "Duplicated frames share", "cut_rate": "Cuts per second",
    "hf_energy": "High-freq. texture energy", "contrast": "Contrast",
    "noise_floor_db": "Noise floor (dBFS)", "flatness": "Spectral flatness (noisiness)",
    "dynamic_range_db": "Dynamic range (dB)", "rolloff99_hz": "Bandwidth (99% roll-off, Hz)",
    "centroid_hz": "Spectral centroid (Hz)", "f0_median_hz": "Dominant pitch f0 (Hz)",
    "voiced_ratio": "Voiced share", "words_per_sec": "Words per second",
    "speech_ratio": "Share of time with speech",
}
