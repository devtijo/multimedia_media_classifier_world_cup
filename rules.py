"""
rules.py - automatic rule-based labels per modality (no training involved):
colour mode, texture, noise level, pitch register, speech density.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency


def rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["color_mode"] = pd.cut(df.saturation, [-1, 0.08, 0.25, 2], labels=["monochrome", "muted", "vivid"])
    out["texture"] = pd.qcut(df.edge_density.rank(method="first"), 3, labels=["smooth", "medium", "detailed"])
    if df.noise_floor_db.notna().sum() > 3:
        out["noise_level"] = pd.qcut(df.noise_floor_db.rank(method="first"), 3, labels=["low", "medium", "high"])
        out["pitch_register"] = pd.cut(df.f0_median_hz, [0, 150, 300, 1e4], labels=["low", "mid", "high"])
        out["pitch_register"] = out.pitch_register.cat.add_categories("unvoiced").fillna("unvoiced")
    if "words_per_sec" in df:
        out["speech"] = pd.cut(df.words_per_sec, [-1, 0.01, 1.5, 99], labels=["silent", "sparse", "dense"])
    return out


def rule_tables(rules: pd.DataFrame, df: pd.DataFrame, eras: list[str]) -> dict:
    """Label distribution per era + chi-square test (clip level, indicative only)."""
    tables = {}
    for c in rules:
        ct = pd.crosstab(df.era, rules[c]).reindex(eras).fillna(0)
        try:
            chi2, p, _, _ = chi2_contingency(ct.loc[:, ct.sum() > 0])
        except ValueError:
            chi2, p = float("nan"), float("nan")
        tables[c] = {"chi2": round(float(chi2), 2), "p": float(p),
                     "distribution": (ct.div(ct.sum(axis=1).replace(0, 1), axis=0).round(2)).to_dict("index")}
    return tables
