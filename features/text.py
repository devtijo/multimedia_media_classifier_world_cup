"""
features/text.py - TEXT branch: commentary transcribed with Whisper (once per World Cup video),
word frequencies (NLTK) and TF-IDF keywords per era (scikit-learn).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from utils import log

TEXT_COLS = ["n_words", "words_per_sec", "speech_ratio", "lexical_diversity"]

NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty
ninety hundred thousand first second third fourth fifth""".split())
HALLUCINATIONS = ("thank you for watching", "subscribe", "thanks for watching", "amara.org",
                  "subtitles by", "[music]", "(music)", "♪")


def stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords as sw
        words = set(sw.words("english"))
    except Exception:
        words = set(ENGLISH_STOP_WORDS)
    # numbers and years are removed so the text model cannot "read" the date
    return words | NUMBER_WORDS | {"world", "cup", "fifa"}


STOP = stopwords()


def clean_text(t: str) -> list[str]:
    t = re.sub(r"[^a-z' ]", " ", t.lower().replace("’", "'"))
    return [w.strip("'") for w in t.split() if len(w.strip("'")) >= 3 and w.strip("'") not in STOP]


def decode_audio_16k(video: Path):
    """Mono 16 kHz float32 waveform via ffmpeg, or None if ffmpeg is unavailable.

    faster-whisper normally decodes the file itself with PyAV, but recent PyAV versions broke
    that call (TypeError: open() got an unexpected keyword argument 'metadata_errors').
    Passing an already-decoded array sidesteps PyAV completely.
    """
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return None
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32)


def transcribe(video: Path, out: Path, model_size: str, task: str) -> dict | None:
    if out.exists():
        return json.loads(out.read_text())
    try:
        from faster_whisper import WhisperModel
        if not hasattr(transcribe, "_m"):
            transcribe._m = WhisperModel(model_size, device="cpu", compute_type="int8")
        audio = decode_audio_16k(video)
        segs, info = transcribe._m.transcribe(audio if audio is not None else str(video),
                                              task=task, vad_filter=True, beam_size=5)
        res = {"language": info.language, "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "no_speech_prob": s.no_speech_prob,
             "avg_logprob": s.avg_logprob} for s in segs]}
    except ImportError:
        try:
            import whisper
            if not hasattr(transcribe, "_w"):
                transcribe._w = whisper.load_model(model_size)
            r = transcribe._w.transcribe(str(video), task=task, fp16=False)
            res = {"language": r.get("language", ""), "segments": [
                {k: s.get(k) for k in ("start", "end", "text", "no_speech_prob", "avg_logprob")}
                for s in r["segments"]]}
        except ImportError:
            return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def clip_text(segments: list[dict], start: float, dur: float) -> tuple[str, float]:
    """Words spoken inside one clip's time window (drops low-confidence / hallucinated segments)."""
    keep, spoken = [], 0.0
    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt or (s.get("no_speech_prob") or 0) > 0.6 or (s.get("avg_logprob") or 0) < -1.2 \
                or any(h in txt.lower() for h in HALLUCINATIONS):
            continue
        ov = min(s["end"], start + dur) - max(s["start"], start)
        if ov > min(1.0, 0.3 * (s["end"] - s["start"])):
            keep.append(txt)
            spoken += ov
    return " ".join(keep), min(spoken, dur)


def text_features(man: pd.DataFrame, videos_dir: Path, cache: Path, model: str, task: str) -> pd.DataFrame | None:
    langs, rows = {}, []
    for year, g in man.groupby("edition_year"):
        vids = sorted(videos_dir.glob(f"wc{year}.*"))
        tr = cache / "transcripts" / f"wc{year}.json"
        if not tr.exists() and not vids:
            log(f"  text: no video for {year} in {videos_dir}; its clips get empty text")
        res = transcribe(vids[0], tr, model, task) if (vids or tr.exists()) else {"language": "", "segments": []}
        if res is None:
            log("  text: no Whisper installed (pip install faster-whisper). Text branch skipped.")
            return None
        langs[year] = res.get("language", "")
        log(f"  transcribed {year} ({langs[year] or '?'}): {len(res['segments'])} segments")
        for _, r in g.iterrows():
            raw, spoken = clip_text(res["segments"], float(r.start_s), float(r.dur_s))
            toks = clean_text(raw)
            n = len(raw.split())
            rows.append({"clip_id": r.clip_id, "text_raw": raw, "text_clean": " ".join(toks),
                         "n_words": n, "words_per_sec": n / float(r.dur_s),
                         "speech_ratio": spoken / float(r.dur_s),
                         "lexical_diversity": len(set(toks)) / len(toks) if toks else 0.0,
                         "commentary_lang": langs[year]})
    return pd.DataFrame(rows)


def has_enough_speech(txt: pd.DataFrame | None, man: pd.DataFrame | None = None,
                      min_edition_share: float = 0.5) -> bool:
    """Is there enough commentary for the text branch to be used in the models?

    Needs (a) a minimum amount of words and (b) speech in at least `min_edition_share` of the
    World Cups. Without (b) the "text" signal collapses into "does this edition have commentary
    at all?", a production-style shortcut, not a property of the era's language.
    """
    if txt is None:
        return False
    enough = bool((txt.text_clean.str.len() > 0).sum() >= 20
                  and len(set(" ".join(txt.text_clean).split())) >= 30)
    if man is None:
        return enough
    t = txt.merge(man[["clip_id", "edition_year"]], on="clip_id")
    share = (t.groupby("edition_year").text_clean.apply(lambda s: (s.str.len() > 0).any())).mean()
    return enough and bool(share >= min_edition_share)


def describe_text(df: pd.DataFrame, eras: list[str], tabdir: Path):
    """Word frequencies (NLTK FreqDist, Counter fallback) and TF-IDF keywords.

    Groups are ERAS when at least two eras have commentary. Otherwise (e.g. only the recent
    World Cups have a commentator) groups fall back to the individual EDITIONS that do, because
    TF-IDF needs at least two documents to say what is distinctive.
    Returns (keywords, word_freqs, groups, extra_results); writes text_keywords.csv.
    """
    info = {"commentary_language": df.groupby("edition_year").commentary_lang.first().to_dict()}
    era_docs = {e: " ".join(df.loc[df.era == e, "text_clean"]) for e in eras}
    if sum(bool(d.strip()) for d in era_docs.values()) >= 2:
        kind, docs = "era", era_docs
    else:
        kind = "edition"
        docs = {str(y): " ".join(g.text_clean) for y, g in df.groupby("edition_year")}
        docs = {k: d for k, d in docs.items() if d.strip()}
    groups = list(docs)
    kw, wf = {}, {}
    for g in groups:
        toks = docs[g].split()
        try:
            import nltk
            wf[g] = nltk.FreqDist(toks).most_common(15)
        except ImportError:
            wf[g] = Counter(toks).most_common(15)
    if len(groups) >= 2:
        tv = TfidfVectorizer(token_pattern=r"(?u)\b[a-z]{3,}\b", sublinear_tf=True)
        M = tv.fit_transform([docs[g] for g in groups]).toarray()
        vocab = np.array(tv.get_feature_names_out())
        kw = {g: vocab[np.argsort(-M[i])[:10]].tolist() for i, g in enumerate(groups)}
    pd.DataFrame([{"group": g, "group_type": kind,
                   "top_words": ", ".join(f"{w} ({n})" for w, n in wf.get(g, [])),
                   "tfidf_keywords": ", ".join(kw.get(g, []))} for g in groups]).to_csv(
        tabdir / "text_keywords.csv", index=False)
    info["text_group_type"] = kind
    info["keywords"] = kw
    info["speech_share_by_era"] = (df.groupby("era").n_words.apply(lambda s: (s > 0).mean())
                                   .reindex(eras).round(2).to_dict())
    return kw, wf, groups, info
