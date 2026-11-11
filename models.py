"""
models.py - era classification, year regression and their validation.

Validation is ALWAYS leave-one-World-Cup-out: clips of one video share the same source, so a
random split would leak the recording into training and inflate every score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RNG, SEED
from features.audio import AUDIO_COLS, VOICE_COLS
from features.image import IMAGE_COLS
from features.text import TEXT_COLS
from stats import modality_of
from utils import log


# ----------------------------------------------------------------------------- estimators
def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))


def rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1))


def text_model():
    return make_pipeline(TfidfVectorizer(min_df=2, max_features=3000, token_pattern=r"(?u)\b[a-z]{3,}\b"),
                         LogisticRegression(C=2.0, max_iter=5000, class_weight="balanced"))


def year_regressor():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1))


# ----------------------------------------------------------------------------- validation
def logo_proba(model, X, y, groups, classes) -> np.ndarray:
    """Out-of-fold class probabilities, leaving ONE WORLD CUP out each time."""
    P = np.zeros((len(y), len(classes)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr], y[tr])
        p = m.predict_proba(X.iloc[te] if hasattr(X, "iloc") else X[te])
        idx = [list(classes).index(c) for c in m.classes_]
        P[np.ix_(te, idx)] = p
    return P


def scores(y, pred, groups, P=None, classes=None) -> dict:
    d = {"clip_accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro")}
    if P is not None:
        dfp = pd.DataFrame(P, columns=classes).assign(g=groups, y=y).groupby("g")
        ed_pred = dfp[list(classes)].mean().idxmax(axis=1)
        ed_true = dfp["y"].first()
        d["edition_accuracy"] = float((ed_pred == ed_true.loc[ed_pred.index]).mean())
        # bootstrap over editions (the real unit of independence)
        ok = pd.Series(np.asarray(y) == np.asarray(pred)).groupby(np.asarray(groups)).agg(["sum", "count"])
        bs = []
        for _ in range(1000):
            s = ok.iloc[RNG.integers(0, len(ok), len(ok))]
            bs.append(s["sum"].sum() / s["count"].sum())
        d["clip_acc_ci_low"], d["clip_acc_ci_high"] = np.percentile(bs, [2.5, 97.5])
    return d


# ----------------------------------------------------------------------------- era classification
def evaluate_classifiers(df: pd.DataFrame, eras: list[str], tcols: list[str], text_ok: bool) -> dict:
    """Every model of the study, leave-one-World-Cup-out.

    Scenarios: "all eras" and "colour-era World Cups only" (critical test: no black-and-white shortcut).
    Returns {"res": results table, "preds": {(scenario, model): (P, pred, classes, mask)},
             "late"/"pred_late": late-fusion probabilities / labels, "Xia", "y", "groups", "classes"}.
    """
    y = df.era.values; groups = df.edition_year.values; classes = np.array(eras)
    Xi, Xa = df[IMAGE_COLS], df[AUDIO_COLS]
    Xa_nv = df[[c for c in AUDIO_COLS if c not in VOICE_COLS]]  # ablation: no voice/pitch features
    Xia = df[IMAGE_COLS + AUDIO_COLS + (tcols if text_ok else [])]
    rows, preds = [], {}

    def run(name, modality, model, X, scen="all eras", mask=None):
        m = np.ones(len(df), bool) if mask is None else mask
        cls = np.array([e for e in eras if e in set(y[m])])
        P = logo_proba(model, X[m], y[m], groups[m], cls)
        pred = cls[P.argmax(1)]
        d = {"scenario": scen, "model": name, "modality": modality, "n_classes": len(cls),
             **scores(y[m], pred, groups[m], P, cls)}
        rows.append(d); preds[(scen, name)] = (P, pred, cls, m)
        log(f"  [{scen}] {name:32s} acc={d['clip_accuracy']:.2f}  F1={d['macro_f1']:.2f}  WC-level={d['edition_accuracy']:.2f}")
        return P

    run("Baseline: random guess", "Baseline", DummyClassifier(strategy="stratified", random_state=SEED), Xi)
    Pi = run("Image (logistic)", "Image", lr(), Xi)
    Pa = run("Audio (logistic)", "Audio", lr(), Xa)
    run("Audio w/o voice features (logistic)", "Audio", lr(), Xa_nv)
    Pt = None
    if text_ok:
        Pt = run("Text (TF-IDF + logistic)", "Text", text_model(), df.text_clean.values)
    run("Early fusion (logistic)", "Fusion", lr(), Xia)
    run("Early fusion (random forest)", "Fusion", rf(), Xia)
    late = np.mean([p for p in (Pi, Pa, Pt) if p is not None], axis=0)
    pred_late = classes[late.argmax(1)]
    rows.append({"scenario": "all eras", "model": "Late fusion (avg. probabilities)", "modality": "Fusion",
                 "n_classes": len(classes), **scores(y, pred_late, groups, late, classes)})
    log(f"  [all eras] {'Late fusion':32s} acc={rows[-1]['clip_accuracy']:.2f}")

    # critical test (a): remove the black-and-white shortcut
    colour = df.saturation.groupby(df.edition_year).transform("median") > 0.08
    if colour.sum() and df.loc[colour, "era"].nunique() >= 2 and df.loc[colour, "edition_year"].nunique() >= 4:
        scen = "colour-era World Cups only"
        run("Baseline: random guess", "Baseline", DummyClassifier(strategy="stratified", random_state=SEED), Xi, scen, colour.values)
        run("Image (logistic)", "Image", lr(), Xi, scen, colour.values)
        run("Audio (logistic)", "Audio", lr(), Xa, scen, colour.values)
        run("Audio w/o voice features (logistic)", "Audio", lr(), Xa_nv, scen, colour.values)
        if text_ok:
            run("Text (TF-IDF + logistic)", "Text", text_model(), df.text_clean.values, scen, colour.values)
        run("Early fusion (logistic)", "Fusion", lr(), Xia, scen, colour.values)
    return {"res": pd.DataFrame(rows), "preds": preds, "late": late, "pred_late": pred_late,
            "Xia": Xia, "y": y, "groups": groups, "classes": classes}


def pick_best_fusion(out: dict):
    """Best fusion model (all eras) by clip accuracy -> (table row, probabilities, predicted labels)."""
    res = out["res"]
    best = res[(res.scenario == "all eras") & (res.modality == "Fusion")].sort_values("clip_accuracy").iloc[-1]
    if best.model.startswith("Late"):
        return best, out["late"], out["pred_late"]
    return (best, *out["preds"][("all eras", best.model)][:2])


# ----------------------------------------------------------------------------- permutation test
def _perm_once(seed, X, years, ed_era, groups, classes):
    r = np.random.default_rng(seed)
    mp = dict(zip(ed_era.index, r.permutation(ed_era.values)))
    yp = years.map(mp).values
    P = logo_proba(lr(), X, yp, groups, classes)
    return accuracy_score(yp, classes[P.argmax(1)])


def permutation_test(df: pd.DataFrame, X, groups, classes, obs: float, n_perm: int, n_jobs: int = -1):
    """Early-fusion logistic model with era labels shuffled BETWEEN World Cups. Returns (null, p)."""
    ed_era = df.groupby("edition_year").era.first()
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_perm_once)(s, X, df.edition_year, ed_era, groups, classes) for s in range(n_perm)))
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    return null, p


# ----------------------------------------------------------------------------- year regression
def year_regression(df: pd.DataFrame, X: pd.DataFrame, groups, y) -> dict:
    """Predict the edition year, leave-one-World-Cup-out. MAE is measured per World Cup."""
    reg = year_regressor()
    yr = df.edition_year.values.astype(float); yhat = np.zeros(len(df)); ybase = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, yr, groups):
        yhat[te] = clone(reg).fit(X.iloc[tr], yr[tr]).predict(X.iloc[te])
        ybase[te] = np.mean(np.unique(yr[tr]))
    edp = pd.DataFrame({"true_year": yr, "pred": yhat, "era": y}).groupby("true_year").agg(
        pred_year=("pred", "median"), q25=("pred", lambda s: s.quantile(.25)), q75=("pred", lambda s: s.quantile(.75)),
        era=("era", "first")).reset_index()
    mae = mean_absolute_error(edp.true_year, edp.pred_year)
    base_mae = mean_absolute_error(edp.true_year, pd.Series(ybase).groupby(yr).first().values)
    summary = {"mae_years_edition": round(mae, 1), "baseline_mae": round(base_mae, 1),
               "mae_years_clip": round(mean_absolute_error(yr, yhat), 1)}
    return {"reg": reg, "yr": yr, "edp": edp, "mae": mae, "base_mae": base_mae, "summary": summary}


# ----------------------------------------------------------------------------- feature importance
def feature_importance(X: pd.DataFrame, y, groups):
    """(1) random-forest MDI averaged over folds (per feature);
    (2) GROUP permutation on held-out World Cups: shuffle a whole modality at once, which is
        robust to the many correlated features (shuffling one alone barely matters).
    Returns (per-feature table, accuracy drop per modality)."""
    fams = {"Image": [c for c in X.columns if c in IMAGE_COLS], "Audio": [c for c in X.columns if c in AUDIO_COLS],
            "Text": [c for c in X.columns if c in TEXT_COLS]}
    fams = {k: v for k, v in fams.items() if v}
    mdi, grp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(rf()).fit(X.iloc[tr], y[tr])
        m[-1].set_params(n_jobs=1)  # tiny held-out sets: parallel overhead dominates
        mdi.append(m[-1].feature_importances_)
        base_acc = accuracy_score(y[te], m.predict(X.iloc[te]))
        for fam, cols in fams.items():
            for _ in range(5):
                Xp = X.iloc[te].copy()
                rows_ = RNG.choice(len(X), len(te), replace=False)   # values from OTHER clips
                Xp[cols] = X.iloc[rows_][cols].values
                grp.append({"modality": fam, "drop": base_acc - accuracy_score(y[te], m.predict(Xp)), "n": len(te)})
    imp = pd.DataFrame({"feature": X.columns, "importance": np.mean(mdi, axis=0), "std": np.std(mdi, axis=0)})
    imp = imp.sort_values("importance", ascending=False); imp["modality"] = imp.feature.map(modality_of)
    g = pd.DataFrame(grp)
    gimp = g.groupby("modality").apply(lambda d: np.average(d["drop"], weights=d.n), include_groups=False)
    return imp, gimp
