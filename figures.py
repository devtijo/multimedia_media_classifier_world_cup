"""
figures.py - every figure of the report (fig01 ... fig19), one function each.

Style rules: eras are ordered -> one sequential hue; modalities keep a fixed colour everywhere;
thin marks; direct labels instead of dense legends.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import ERA_RAMP, INK, INK2, MODALITY_COLORS, SEED  # noqa: E402
from features.audio import LTAS_FREQS, PITCH_CLASSES  # noqa: E402


def savefig(fig, figdir: Path, name: str, saved: list):
    fig.savefig(figdir / f"{name}.png"); plt.close(fig); saved.append(f"{name}.png")


def era_palette(eras):
    return dict(zip(eras, ERA_RAMP[:len(eras)] if len(eras) <= 5 else sns.color_palette("Blues", len(eras))))


# ----------------------------------------------------------------------------- dataset / image
def fig_overview(df, ed, eras, pal, figdir, saved):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    c = df.groupby("era").size().reindex(eras)
    ax[0].bar(eras, c.values, color=[pal[e] for e in eras], width=0.6)
    for i, v in enumerate(c.values):
        ax[0].text(i, v, f"{v}\n({ed[ed.era == eras[i]].shape[0]} WCs)", ha="center", va="bottom", fontsize=8, color=INK2)
    ax[0].set(title="Clips per era", ylabel="clips"); ax[0].set_ylim(0, c.max() * 1.3)
    cc = df.groupby("edition_year").size()
    ax[1].bar(cc.index, cc.values, width=2.6, color=[pal[ed.set_index("edition_year").era[y]] for y in cc.index])
    ax[1].set(title="Clips per World Cup edition", xlabel="edition", ylabel="clips")
    ax[1].set_xticks(cc.index); ax[1].tick_params(axis="x", rotation=60)
    savefig(fig, figdir, "fig01_dataset_overview", saved)


def fig_montage(df, figdir, saved):
    eds = sorted(df.edition_year.unique()); n = len(eds); cols = 6; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.65))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, y in zip(axes.ravel(), eds):
        g = df[df.edition_year == y].sort_values("start_s")
        r = g.iloc[len(g) // 2]
        cap = cv2.VideoCapture(r.video_path); n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_f // 2)); ok, f = cap.read(); cap.release()
        if ok:
            ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        ax.set_title(str(y), fontsize=10, color=INK)
    fig.suptitle("One frame per World Cup (middle clip of each video)", fontweight="bold")
    fig.tight_layout(); savefig(fig, figdir, "fig02_frame_montage", saved)


def fig_palettes(df, figdir, saved):
    eds = sorted(df.edition_year.unique())
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(eds) + 1))
    for i, y in enumerate(eds):
        cols = Counter()
        for p in df.loc[df.edition_year == y, "palette"]:
            for hx, sh in json.loads(p):
                cols[hx] += sh
        tot = sum(cols.values()); left = 0
        # merge similar colors crudely by quantizing to 32 levels
        q = Counter()
        for hx, sh in cols.items():
            rgb = tuple(int(hx[k:k + 2], 16) // 32 * 32 + 16 for k in (1, 3, 5))
            q[rgb] += sh / tot
        for rgb, sh in q.most_common(8):
            ax.barh(i, sh, left=left, color=np.array(rgb) / 255, height=0.82, edgecolor="white", linewidth=1)
            left += sh
    ax.set_yticks(range(len(eds))); ax.set_yticklabels(eds); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of pixels (8 dominant colours, k-means)"); ax.grid(False)
    ax.set_title("Dominant colour palette of each World Cup")
    savefig(fig, figdir, "fig03_dominant_palettes", saved)


def fig_box_grid(df, feats, labels, eras, pal, title, name, figdir, saved, ncols=3):
    feats = [f for f in feats if f in df and df[f].notna().any()]
    n = len(feats); rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(4.2 * ncols, 3.1 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="era", y=f, order=eras, hue="era", palette=pal, legend=False,
                    ax=ax, width=0.55, fliersize=1.5, linewidth=0.9)
        ax.set(title=labels.get(f, f), xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    savefig(fig, figdir, name, saved)


def fig_trends(ed, feats, labels, pal, name, title, figdir, saved):
    feats = [f for f in feats if f in ed and ed[f].notna().any()]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.6 * len(feats), 3.3), squeeze=False)
    for ax, f in zip(axes[0], feats):
        x = ed[["edition_year", f, "era"]].dropna()
        ax.plot(x.edition_year, x[f], color="#c9c8c2", lw=1.2, zorder=1)
        ax.scatter(x.edition_year, x[f], c=[pal[e] for e in x.era], s=46, edgecolor="white", lw=1.5, zorder=2)
        rho, p = spearmanr(x.edition_year, x[f])
        ax.set_title(labels.get(f, f), fontsize=10)
        ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}\np = {p:.3g}", transform=ax.transAxes, va="top",
                fontsize=8, color=INK2)
        ax.set_xlabel("edition")
    fig.suptitle(title + "  (one dot = median of the clips of one World Cup)", fontweight="bold")
    fig.tight_layout(); savefig(fig, figdir, name, saved)


# ----------------------------------------------------------------------------- audio
def fig_ltas(ltas, df, eras, pal, figdir, saved):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for e in eras:
        ids = df.loc[df.era == e, "clip_id"]
        m = np.array([ltas[i] for i in ids if i in ltas])
        if not len(m) or np.all(np.isnan(m)):
            continue
        curve = np.nanmean(m, axis=0); curve -= np.nanmax(curve)
        ax.plot(LTAS_FREQS, curve, color=pal[e], lw=2, label=e)
    ax.set_xscale("log"); ax.set(xlabel="frequency (Hz)", ylabel="relative power (dB)",
                                 title="Average spectrum per era: how much high-frequency detail survives")
    ax.set_ylim(bottom=-80); ax.legend(title="era")
    savefig(fig, figdir, "fig07_audio_spectrum_by_era", saved)


def fig_pitch(df, eras, pal, figdir, saved):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    sns.stripplot(data=df, x="era", y="f0_median_hz", order=eras, hue="era", palette=pal, legend=False,
                  ax=ax[0], size=3.5, alpha=0.75, jitter=0.25)
    sns.pointplot(data=df, x="era", y="f0_median_hz", order=eras, color=INK, estimator=np.nanmedian,
                  errorbar=None, markers="D", linestyles="none", ax=ax[0])
    ax[0].set(title="Dominant pitch (pYIN f0) per clip; ◆ = era median", ylabel="Hz", xlabel="")
    ct = pd.crosstab(df.era, df.pitch_class).reindex(index=eras, columns=PITCH_CLASSES, fill_value=0)
    ct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    sns.heatmap(ct, cmap="Blues", ax=ax[1], cbar_kws={"label": "share of clips"}, linewidths=1, linecolor="white")
    ax[1].set(title="Dominant pitch class (chroma) per era", xlabel="pitch class", ylabel="")
    fig.tight_layout(); savefig(fig, figdir, "fig08_dominant_pitch", saved)


# ----------------------------------------------------------------------------- rules / text
def fig_rules(rules, df, eras, figdir, saved):
    cols = [c for c in ["color_mode", "texture", "noise_level", "pitch_register", "speech"] if c in rules]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6), squeeze=False)
    for ax, c in zip(axes[0], cols):
        ct = pd.crosstab(df.era, rules[c]).reindex(eras).fillna(0)
        ct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
        cats = list(ct.columns)
        shades = sns.color_palette("Blues", len(cats) + 1)[1:]
        left = np.zeros(len(eras))
        for k, cat in enumerate(cats):
            ax.barh(eras, ct[cat].values, left=left, color=shades[k], edgecolor="white", linewidth=2, label=str(cat))
            left += ct[cat].values
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_title(c.replace("_", " ")); ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(4, len(cats)), fontsize=8)
    fig.suptitle("Automatic rule-based labels per modality (share of clips in each era)", fontweight="bold")
    fig.tight_layout(); savefig(fig, figdir, "fig09_rule_based_labels", saved)


def fig_words(kw, wf, groups, figdir, saved, kind="era"):
    eras = [e for e in groups if e in wf and len(wf[e])]
    if not eras:
        return
    fig, axes = plt.subplots(1, len(eras), figsize=(3.2 * len(eras), 4.2), squeeze=False)
    for ax, e in zip(axes[0], eras):
        words, counts = zip(*wf[e][:12])
        ax.barh(words[::-1], counts[::-1], color=MODALITY_COLORS["Text"], height=0.6)
        ax.set_title(e); ax.set_xlabel("count")
        k = ", ".join(kw.get(e, [])[:5])
        ax.text(0, -0.16, f"TF-IDF: {k}", transform=ax.transAxes, fontsize=7.5, color=INK2, wrap=True)
    fig.suptitle(f"Most frequent commentary words per {kind} (stop-words and numbers removed)", fontweight="bold")
    fig.tight_layout(); savefig(fig, figdir, "fig10_word_frequencies", saved)


# ----------------------------------------------------------------------------- statistics
def fig_effects(tests, figdir, saved, top=20):
    t = tests.head(top).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(t) + 1.5), sharey=True)
    ax[0].barh(t.feature, t.epsilon2, color=[MODALITY_COLORS[m] for m in t.modality], height=0.65)
    for i, (e, q) in enumerate(zip(t.epsilon2, t.q_kruskal)):
        ax[0].text(e, i, "  *" if q < 0.05 else "", va="center", color=INK)
    ax[0].set(title="Separation between eras (Kruskal-Wallis ε²)", xlabel="ε²  (* = FDR q < 0.05)")
    ax[1].barh(t.feature, t.spearman_rho, color=[MODALITY_COLORS[m] for m in t.modality], height=0.65)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set(title="Monotonic trend with year (Spearman ρ)", xlabel="ρ", xlim=(-1, 1))
    mods = [m for m in ["Image", "Audio", "Text"] if m in set(t.modality)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS[m]) for m in mods]
    fig.legend(handles, mods, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which features change with the era? (edition-level, n = World Cups)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); savefig(fig, figdir, "fig11_feature_effects", saved)


def fig_corr(df, cols, figdir, saved):
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 2, 0.42 * len(cols) + 1.5))
    sns.heatmap(c, cmap=sns.diverging_palette(250, 15, s=75, l=45, as_cmap=True), center=0, vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Spearman ρ", "shrink": 0.6}, linewidths=0.5, linecolor="white")
    ax.set_title("Correlation between image and audio features (clip level)")
    savefig(fig, figdir, "fig12_feature_correlation", saved)


def fig_pca(X, df, eras, pal, figdir, saved):
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(2, random_state=SEED)).fit(X)
    pca = Z[-1]; P = Z.transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for e in eras:
        m = (df.era == e).values
        ax.scatter(P[m, 0], P[m, 1], s=18, color=pal[e], alpha=0.55, label=e, edgecolor="none")
    cen = pd.DataFrame(P, columns=["a", "b"]).assign(y=df.edition_year.values).groupby("y").median()
    for y, r in cen.iterrows():
        ax.annotate(str(y), (r.a, r.b), fontsize=8, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#c9c8c2", lw=0.6))
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
           title="PCA of image + audio features (labels = median position of each World Cup)")
    ax.legend(title="era", markerscale=1.6)
    savefig(fig, figdir, "fig13_pca", saved)


# ----------------------------------------------------------------------------- models
def fig_models(res, figdir, saved):
    r = res[res.scenario == "all eras"].copy()
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(r) + 1.2))
    y = np.arange(len(r))[::-1]
    cols = [MODALITY_COLORS[m] for m in r.modality]
    ax.barh(y, r.clip_accuracy, color=cols, height=0.6)
    err = np.vstack([r.clip_accuracy - r.clip_acc_ci_low, r.clip_acc_ci_high - r.clip_accuracy]).clip(0)
    ax.errorbar(r.clip_accuracy, y, xerr=err, fmt="none", ecolor=INK2, capsize=3, lw=1)
    for yi, (a, f1, ea) in zip(y, zip(r.clip_accuracy, r.macro_f1, r.edition_accuracy)):
        ax.text(1.04, yi, f"acc {a:.2f} · F1 {f1:.2f} · WC {ea:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(r.model); ax.set_xlim(0, 1.5); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.axvline(1.0, color="#c9c8c2", lw=0.8)
    ax.axvline(1 / r.n_classes.iloc[0], color=INK2, ls=":", lw=1)
    ax.set(xlabel="clip accuracy, leave-one-World-Cup-out (bars: 95% bootstrap CI over editions; dotted = chance)",
           title="Era classification: which modality knows the era?")
    savefig(fig, figdir, "fig14_model_comparison", saved)


def fig_confusion(y, pred, eras, title, name, figdir, saved):
    cm = confusion_matrix(y, pred, labels=eras)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cmn, annot=np.array([[f"{p:.0%}\n({n})" for p, n in zip(rp, rn)] for rp, rn in zip(cmn, cm)]),
                fmt="", cmap="Blues", vmin=0, vmax=1, xticklabels=eras, yticklabels=eras, ax=ax,
                cbar_kws={"label": "share of true era"}, linewidths=1, linecolor="white", annot_kws={"fontsize": 8})
    ax.set(xlabel="predicted era", ylabel="true era", title=title)
    savefig(fig, figdir, name, saved)


def fig_year(edp, mae, base, figdir, saved, pal):
    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = edp.true_year.min() - 6, edp.true_year.max() + 6
    ax.plot([lo, hi], [lo, hi], color="#c9c8c2", lw=1.2, ls="--", label="perfect")
    ax.vlines(edp.true_year, edp.q25, edp.q75, color="#c9c8c2", lw=2)
    ax.scatter(edp.true_year, edp.pred_year, c=[pal[e] for e in edp.era], s=60, edgecolor="white", lw=1.5, zorder=3)
    for _, r in edp.iterrows():
        ax.annotate(str(int(r.true_year)), (r.true_year, r.pred_year), xytext=(5, -3), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="true year", ylabel="predicted year (median of clips, IQR bar)",
           title=f"Year regression, leave-one-World-Cup-out\nMAE = {mae:.1f} years (baseline {base:.1f})")
    ax.legend(loc="upper left")
    savefig(fig, figdir, "fig16_year_prediction", saved)


def fig_importance(imp, gimp, figdir, saved, top=15):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.38 * len(t) + 1.4), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].barh(t.feature, t.importance, xerr=t["std"], color=[MODALITY_COLORS[m] for m in t.modality],
               height=0.6, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS[m]) for m in ["Image", "Audio", "Text"]]
    ax[0].legend(handles, ["Image", "Audio", "Text"], loc="lower right")
    ax[0].set(xlabel="mean decrease in impurity (avg. over folds, ±sd)", title="Top features (random forest)")
    gi = pd.Series(gimp).sort_values()
    ax[1].barh(gi.index, gi.values, color=[MODALITY_COLORS[m] for m in gi.index], height=0.5)
    for i, v in enumerate(gi.values):
        ax[1].text(v, i, f" {v:.2f}", va="center", fontsize=9, color=INK2)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set(xlabel="accuracy lost when the whole modality is shuffled", title="Which modality matters?")
    fig.suptitle("What gives the era away?", fontweight="bold"); fig.tight_layout()
    savefig(fig, figdir, "fig17_feature_importance", saved)


# ----------------------------------------------------------------------------- critical tests
def fig_degradation(dg, figdir, saved):
    g = dg.groupby("edition_year")[["pred_original", "pred_degraded"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(g) + 1.5))
    for i, r in g.iterrows():
        ax.plot([r.pred_degraded, r.pred_original], [i, i], color="#c9c8c2", lw=2.5, zorder=1)
    ax.scatter(g.pred_original, g.index, color=MODALITY_COLORS["Image"], s=70, zorder=2, label="original clip")
    ax.scatter(g.pred_degraded, g.index, color=MODALITY_COLORS["Audio"], s=70, zorder=2,
               label="same clip, degraded (low-res, grain, faded colour, 4 kHz audio + hiss)")
    ax.scatter(g.edition_year, g.index, marker="|", color=INK, s=200, zorder=2, label="true year")
    ax.set_yticks(g.index); ax.set_yticklabels(g.edition_year)
    shift = (dg.pred_degraded - dg.pred_original).median()
    ax.set(xlabel="predicted year", title=f"Degradation test: degraded clips look {abs(shift):.0f} years "
                                          f"{'older' if shift < 0 else 'newer'} (median)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    savefig(fig, figdir, "fig18_degradation_test", saved)


def fig_perm(null, obs, p, figdir, saved):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(null, bins=25, color="#c9c8c2", edgecolor="white")
    ax.axvline(obs, color=MODALITY_COLORS["Fusion"], lw=2.5)
    ax.text(obs, ax.get_ylim()[1] * 0.92, f"  observed {obs:.2f}\n  p = {p:.3f}", color=INK, fontsize=9, va="top")
    ax.set(xlabel="clip accuracy with era labels shuffled between World Cups", ylabel="permutations",
           title="Permutation test: is the fusion model better than luck?")
    savefig(fig, figdir, "fig19_permutation_test", saved)
