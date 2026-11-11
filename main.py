#!/usr/bin/env python3
"""
main.py - Multimodal Media Classifier (Python for Research, final project). Orchestrator.

Question: can we tell WHEN a football clip was filmed (which World Cup era) from its
images, its sound and its words?  Data: official FIFA highlight videos of every World
Cup 1930-2026, cut into short clips by build_dataset.py (clips_manifest.csv).

Pipeline (one command runs everything; features are cached, so re-runs are fast):
  1. IMAGE  - features/image.py : colour and patterns with OpenCV.
  2. AUDIO  - features/audio.py : noise level and dominant pitch with LibROSA.
  3. TEXT   - features/text.py  : Whisper transcripts, word frequencies, TF-IDF keywords.
  4. STATS  - stats.py          : per-era summaries, Kruskal-Wallis + epsilon^2, Spearman, FDR
                                  (EDITION level, n = 23: clips of one video are not independent).
  5. RULES  - rules.py          : automatic per-modality labels.
  6. MODELS - models.py         : era classification per modality + fusion, year regression,
                                  leave-one-World-Cup-out validation, bootstrap CIs, permutation test.
  7. CRITICAL TESTS - (a) colour eras only (no black-and-white shortcut) in models.py;
                      (b) degradation test in degradation.py: do degraded modern clips get "older"?
  8. FIGURES + TABLES + summary.md (figures.py, report.py).

Usage (from the folder that holds clips_manifest.csv, clips/ and videos/):
  python multimodal_classifier/main.py                       # full pipeline
  python multimodal_classifier/main.py --no-text             # skip Whisper / text branch
  python multimodal_classifier/main.py --whisper-model small # better transcription (slower)
  python multimodal_classifier/main.py --force               # recompute cached features
Optional: pip install faster-whisper   (text branch; first run downloads the model)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
from config import FEATURE_LABELS as L
from degradation import run_degradation_test
from features.audio import AUDIO_COLS, extract_audio_features
from features.image import IMAGE_COLS, extract_image_features
from features.text import TEXT_COLS, describe_text, has_enough_speech, text_features
from figures import (era_palette, fig_box_grid, fig_confusion, fig_corr, fig_degradation, fig_effects,
                     fig_importance, fig_ltas, fig_models, fig_montage, fig_overview, fig_palettes,
                     fig_pca, fig_perm, fig_pitch, fig_rules, fig_trends, fig_words, fig_year)
from models import (evaluate_classifiers, feature_importance, permutation_test, pick_best_fusion,
                    year_regression)
from report import write_summary
from rules import rule_labels, rule_tables
from stats import run_statistics
from utils import log, set_style


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=config.DEFAULT_MANIFEST)
    ap.add_argument("--videos-dir", type=Path, default=config.DEFAULT_VIDEOS_DIR)
    ap.add_argument("--out", type=Path, default=config.DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=config.DEFAULT_CACHE)
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--whisper-task", default="translate", choices=["translate", "transcribe"],
                    help="translate = everything to English (uniform vocabulary)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-degradation", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    return ap.parse_args()


def main():
    args = parse_args()

    set_style()
    figdir, tabdir = args.out / "figures", args.out / "tables"
    for d in (figdir, tabdir, args.cache):
        d.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    R: dict = {}

    # ---------------------------------------------------------------- load
    man = pd.read_csv(args.manifest, dtype={"audio_path": str}).fillna({"audio_path": ""})
    man = man[man.video_path.map(lambda p: Path(p).exists())].reset_index(drop=True)
    if man.empty:
        sys.exit("No clips found. Run build_dataset.py first (from the same folder).")
    eras = (man.groupby("era").edition_year.min().sort_values().index.tolist())
    pal = era_palette(eras)
    log(f"Loaded {len(man)} clips, {man.edition_year.nunique()} World Cups, eras: {', '.join(eras)}")

    # ---------------------------------------------------------------- 1-3. features
    img = extract_image_features(man, args.cache, args.force, args.n_jobs)
    aud, ltas = extract_audio_features(man, args.cache, args.force, args.n_jobs)

    txt = None
    if not args.no_text:
        log(f"TEXT: Whisper '{args.whisper_model}' ({args.whisper_task}), once per World Cup video...")
        txt = text_features(man, args.videos_dir, args.cache, args.whisper_model, args.whisper_task)
    text_ok = has_enough_speech(txt, man)
    if txt is not None:
        n_sp = txt.merge(man[["clip_id", "edition_year"]], on="clip_id").groupby("edition_year") \
            .text_clean.apply(lambda s: (s.str.len() > 0).any()).sum()
        log(f"  text: commentary found in {int(n_sp)} of {man.edition_year.nunique()} World Cups")
    if txt is not None and not text_ok:
        log("  text: commentary too sparse across editions -> text kept for description only, "
            "excluded from models (it would just detect 'has commentary', not the era)")

    df = man.merge(img, on="clip_id").merge(aud, on="clip_id")
    if txt is not None:
        df = df.merge(txt, on="clip_id", how="left")
        df["text_clean"] = df.text_clean.fillna("")
    tcols = TEXT_COLS if txt is not None else []
    # Text features enter the statistics only if the text branch is usable. If commentary exists
    # in a few editions only, "words per second" just flags those editions (n_words = 0 elsewhere)
    # and would show up as a huge but meaningless era effect.
    num_cols = IMAGE_COLS + AUDIO_COLS + (tcols if text_ok else [])
    df.drop(columns=[c for c in ["text_raw"] if c in df]).to_csv(tabdir / "features_clip_level.csv", index=False)
    ed = df.groupby(["edition_year", "era"])[num_cols].median().reset_index()
    ed.to_csv(tabdir / "features_edition_level.csv", index=False)
    if txt is not None and not text_ok:
        df.groupby("era")[tcols].agg(["mean", "median"]).reindex(eras).to_csv(tabdir / "text_descriptives_by_era.csv")

    # ---------------------------------------------------------------- 4. statistics
    log("Statistics...")
    tests, info = run_statistics(df, ed, num_cols, eras, tabdir)
    R.update(info)

    # ---------------------------------------------------------------- 5. rule-based labels
    rules = rule_labels(df)
    rules.assign(clip_id=df.clip_id, era=df.era).to_csv(tabdir / "rule_labels.csv", index=False)
    R["rule_tables"] = rule_tables(rules, df, eras)

    # ---------------------------------------------------------------- text descriptives
    kw, wf, groups = {}, {}, []
    if txt is not None:
        kw, wf, groups, info = describe_text(df, eras, tabdir)
        R.update(info)

    # ---------------------------------------------------------------- 6. models
    log("Classification (leave-one-World-Cup-out)...")
    out = evaluate_classifiers(df, eras, tcols, text_ok)
    res, Xia, y, groups, classes = out["res"], out["Xia"], out["y"], out["groups"], out["classes"]
    res.round(4).to_csv(tabdir / "model_results.csv", index=False)
    R["models"] = res.round(3).to_dict("records")

    best, Pb, pb = pick_best_fusion(out)
    R["best_model"] = best.model
    pd.DataFrame({"clip_id": df.clip_id, "edition_year": groups, "true_era": y, "pred_era": pb,
                  **{f"p_{c}": Pb[:, i] for i, c in enumerate(classes)}}).to_csv(
        tabdir / "predictions_clip_level.csv", index=False)
    ed_pred = pd.DataFrame(Pb, columns=classes).assign(g=groups, true=y).groupby("g")
    pd.DataFrame({"true_era": ed_pred["true"].first(), "pred_era": ed_pred[list(classes)].mean().idxmax(axis=1)}
                 ).to_csv(tabdir / "predictions_edition_level.csv")

    log(f"Permutation test ({args.n_perm} shuffles)...")
    obs = res.query("scenario=='all eras' and model=='Early fusion (logistic)'").clip_accuracy.iloc[0]
    null, p_perm = permutation_test(df, Xia, groups, classes, obs, args.n_perm, args.n_jobs)
    R["permutation_test"] = {"observed_accuracy": round(float(obs), 3), "null_mean": round(float(null.mean()), 3),
                             "p_value": round(float(p_perm), 4), "n_perm": args.n_perm}

    log("Year regression...")
    yrg = year_regression(df, Xia, groups, y)
    edp, mae, base_mae = yrg["edp"], yrg["mae"], yrg["base_mae"]
    edp.to_csv(tabdir / "year_predictions_edition_level.csv", index=False)
    R["year_regression"] = yrg["summary"]

    log("Feature importance...")
    imp, gimp = feature_importance(Xia, y, groups)
    imp.to_csv(tabdir / "feature_importance.csv", index=False)
    gimp.rename("accuracy_drop").to_csv(tabdir / "modality_importance.csv")
    R["top_importance"] = imp.head(8)[["feature", "modality", "importance"]].round(4).to_dict("records")
    R["modality_importance"] = gimp.round(3).to_dict()

    # ---------------------------------------------------------------- 7. critical test (b): degradation
    dg = None
    if not args.skip_degradation:
        dg, R["degradation_test"] = run_degradation_test(
            df, eras, yrg["reg"], yrg["yr"], y, args.cache, tabdir, args.force, args.n_jobs)

    # ---------------------------------------------------------------- 8. figures
    log("Figures...")
    fig_overview(df, ed, eras, pal, figdir, saved)
    fig_montage(df, figdir, saved)
    fig_palettes(df, figdir, saved)
    fig_box_grid(df, ["saturation", "colorfulness", "green_ratio", "sharpness_log", "grain", "edge_density",
                      "pillarbox", "dup_frames", "cut_rate"], L, eras, pal,
                 "IMAGE features by era (colour and patterns)", "fig04_image_features_by_era", figdir, saved)
    fig_trends(ed, ["saturation", "sharpness_log", "grain", "pillarbox"], L, pal, "fig05_image_trends",
               "Image trends across World Cups", figdir, saved)
    fig_box_grid(df, ["noise_floor_db", "flatness", "dynamic_range_db", "rolloff99_hz", "centroid_hz", "f0_median_hz"],
                 L, eras, pal, "AUDIO features by era (noise level and pitch)", "fig06_audio_features_by_era", figdir, saved)
    fig_ltas(ltas, df, eras, pal, figdir, saved)
    fig_pitch(df, eras, pal, figdir, saved)
    fig_rules(rules, df, eras, figdir, saved)
    if txt is not None:
        fig_words(kw, wf, groups, figdir, saved, info.get("text_group_type", "era"))
        fig_box_grid(df, ["words_per_sec", "speech_ratio"], L, eras, pal, "TEXT: how much commentary per clip",
                     "fig10b_speech_by_era", figdir, saved, ncols=2)
    fig_effects(tests, figdir, saved)
    fig_corr(df, [c for c in IMAGE_COLS + AUDIO_COLS[:13] if df[c].nunique() > 1], figdir, saved)
    fig_pca(df[IMAGE_COLS + AUDIO_COLS], df, eras, pal, figdir, saved)
    fig_models(res, figdir, saved)
    fig_confusion(y, pb, eras, f"Confusion matrix - {best.model}\n(leave-one-World-Cup-out, clip level)",
                  "fig15_confusion_matrix", figdir, saved)
    fig_year(edp, mae, base_mae, figdir, saved, pal)
    fig_importance(imp, R["modality_importance"], figdir, saved)
    if dg is not None:
        fig_degradation(dg, figdir, saved)
    fig_perm(null, obs, p_perm, figdir, saved)

    # ---------------------------------------------------------------- summary
    R.update({"n_clips": len(df), "n_editions": int(df.edition_year.nunique()), "eras": eras,
              "clips_per_era": df.era.value_counts().reindex(eras).to_dict(), "text_used_in_models": bool(text_ok),
              "figures": saved})
    (args.out / "results.json").write_text(
        json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_summary(args.out / "summary.md", R, res, tests)
    log(f"Done. {len(saved)} figures in {figdir}, tables in {tabdir}, key numbers in {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
