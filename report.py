"""
report.py - summary.md: every number needed to write the PDF report, in one place.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary(path: Path, R: dict, res: pd.DataFrame, tests: pd.DataFrame):
    m = res.copy()
    lines = ["# Multimodal Media Classifier - key results\n",
             f"Dataset: {R['n_clips']} clips from {R['n_editions']} World Cups, {len(R['eras'])} eras "
             f"({', '.join(f'{e}: {n}' for e, n in R['clips_per_era'].items())}).",
             "Validation: leave-one-World-Cup-out (no clip of a tested video is ever seen in training).\n",
             "## Era classification\n", "| scenario | model | clip acc [95% CI] | macro-F1 | World-Cup-level acc |",
             "|---|---|---|---|---|"]
    for _, r in m.iterrows():
        lines.append(f"| {r.scenario} | {r.model} | {r.clip_accuracy:.2f} [{r.clip_acc_ci_low:.2f}-{r.clip_acc_ci_high:.2f}] "
                     f"| {r.macro_f1:.2f} | {r.edition_accuracy:.2f} |")
    pt = R["permutation_test"]
    lines += [f"\nPermutation test (early fusion): observed {pt['observed_accuracy']}, null mean {pt['null_mean']}, "
              f"p = {pt['p_value']} ({pt['n_perm']} shuffles).",
              f"\nYear regression: MAE {R['year_regression']['mae_years_edition']} years per World Cup "
              f"(baseline {R['year_regression']['baseline_mae']}).\n", "## Features that change most with the era (edition level)\n",
              "| feature | modality | ε² | Spearman ρ vs year | FDR q |", "|---|---|---|---|---|"]
    for _, r in tests.head(10).iterrows():
        lines.append(f"| {r.feature} | {r.modality} | {r.epsilon2:.2f} | {r.spearman_rho:+.2f} | {r.q_kruskal:.3g} |")
    lines.append(f"\n{R['n_significant_fdr']} features differ between eras at FDR q < 0.05.\n")
    lines += ["## Importance\n", f"Accuracy lost when a whole modality is shuffled (held-out World Cups): {R['modality_importance']}\n",
              "Top features (random-forest impurity importance):"]
    lines += [f"- {d['feature']} ({d['modality']}): {d['importance']:.3f}" for d in R["top_importance"]]
    if "degradation_test" in R:
        d = R["degradation_test"]
        lines += ["\n## Degradation test (is the model learning quality instead of history?)\n",
                  f"Modern clips ({d['era']}) were artificially degraded. Median predicted-year shift: "
                  f"{d['median_year_shift']} years; {d['share_reclassified_older']:.0%} of degraded clips leave their era "
                  f"(vs {1 - d['share_original_correct']:.0%} of originals)."]
    if "keywords" in R:
        lines += [f"\n## Text keywords per {R.get('text_group_type', 'era')} (TF-IDF)\n"] + [f"- {e}: {', '.join(k)}" for e, k in R["keywords"].items()]
        lines.append(f"\nShare of clips with speech per era: {R['speech_share_by_era']}")
        lines.append(f"Commentary language detected by Whisper: {R.get('commentary_language')}")
    lines += ["\n## Rule-based labels (chi-square vs era, clip level - indicative only, clips are not independent)\n"]
    lines += [f"- {k}: chi2 = {v['chi2']}, p = {v['p']:.3g}" for k, v in R["rule_tables"].items()]
    lines += ["\n## Figures\n"] + [f"- {f}" for f in R["figures"]]
    path.write_text("\n".join(lines), encoding="utf-8")
