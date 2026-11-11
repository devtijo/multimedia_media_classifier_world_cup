"""
stats.py - descriptive statistics and tests at EDITION level (n = World Cups).

Clips of one video are not independent, so every test uses one value (the median) per
World Cup: Kruskal-Wallis across eras (+ epsilon^2 effect size), Spearman vs year,
Benjamini-Hochberg FDR across features.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats as sps  # aliased: this module is itself called "stats"

from features.audio import AUDIO_COLS
from features.image import IMAGE_COLS
from utils import bh_fdr


def modality_of(c: str) -> str:
    return "Image" if c in IMAGE_COLS else "Audio" if c in AUDIO_COLS else "Text"


def feature_tests(ed: pd.DataFrame, cols: list[str], eras: list[str]) -> pd.DataFrame:
    """Edition-level tests: Kruskal-Wallis across eras (+ epsilon^2) and Spearman vs year."""
    rows = []
    for c in cols:
        x = ed[[c, "era", "edition_year"]].dropna()
        groups = [x.loc[x.era == e, c].values for e in eras if (x.era == e).sum() > 0]
        if len(groups) < 2 or x[c].nunique() < 2:
            continue
        H, p = sps.kruskal(*groups)
        n, k = len(x), len(groups)
        rho, ps = sps.spearmanr(x["edition_year"], x[c])
        rows.append({"feature": c, "modality": modality_of(c), "kruskal_H": H, "p_kruskal": p,
                     "epsilon2": max(0.0, (H - k + 1) / (n - k)), "spearman_rho": rho,
                     "p_spearman": ps, "n_editions": n})
    t = pd.DataFrame(rows)
    if len(t):
        t["q_kruskal"] = bh_fdr(t.p_kruskal); t["q_spearman"] = bh_fdr(t.p_spearman)
    return t.sort_values("epsilon2", ascending=False)


def run_statistics(df: pd.DataFrame, ed: pd.DataFrame, num_cols: list[str], eras: list[str], tabdir):
    """Summary per era, edition-level tests and source-quality table. Returns (tests, results)."""
    summ = df.groupby("era")[num_cols].agg(["mean", "std", "median"]).reindex(eras)
    summ.to_csv(tabdir / "summary_by_era.csv")
    tests = feature_tests(ed, num_cols, eras)
    tests.to_csv(tabdir / "feature_tests_edition_level.csv", index=False)
    info = {
        "n_significant_fdr": int((tests.q_kruskal < 0.05).sum()),
        "top_features": tests.head(8)[["feature", "modality", "epsilon2", "spearman_rho", "q_kruskal"]]
        .round(3).to_dict("records"),
    }
    src = df.groupby("era")[["src_height", "src_fps"]].agg(lambda s: s.value_counts().index[0]).reindex(eras)
    src.to_csv(tabdir / "source_quality_by_era.csv")
    return tests, info
