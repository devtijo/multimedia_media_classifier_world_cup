#!/usr/bin/env python3
"""
build_dataset.py - Dataset de clips para el Multimodal Media Classifier
(clasificador de la época de clips de fútbol: un video por Mundial, 1930-2026).

Flujo:
  1. Lee editions.csv (un video por edición del Mundial, etiquetado con el AÑO de la edición).
  2. Busca videos/wc<año>.<ext>; si no existe, lo descarga con yt-dlp desde la URL
     (uso académico; revisa los términos de la fuente). También puedes dejar tú el archivo.
  3. Quita los primeros --head segundos (intro FIFA) y los últimos --tail (outro) y parte
     lo que queda en clips consecutivos de --clip-seconds. Cada clip da un .mp4 y, si el
     video tiene audio, un .wav mono a 44.1 kHz cortado de la fuente.
  4. Escribe clips_manifest.csv con etiquetas (año, década, era) y metadatos de la FUENTE
     (resolución, fps, códec, sample rate...) para controlar el confusor "calidad de fuente".

Uso:
  python build_dataset.py --dry-run      # qué videos hay y cuáles faltan
  python build_dataset.py                # descarga + recorta
  python build_dataset.py --stats        # resumen del manifiesto existente

IMPORTANTE para el modelado: todos los clips de un video comparten fuente. Separa
entrenamiento/prueba por 'edition_year' (GroupKFold / leave-one-edition-out), nunca por clip,
o el modelo reconocerá el video y no la época.

Requiere ffmpeg y ffprobe; yt-dlp solo para descargar.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".ts")
MANIFEST_FIELDS = [
    "clip_id", "edition_year", "decade", "era", "start_s", "dur_s",
    "video_path", "audio_path", "has_audio",
    "src_width", "src_height", "src_fps", "src_vcodec",
    "src_audio_sr", "src_audio_ch", "src_bitrate_kbps", "src_duration_s",
]


def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"ERROR: no encuentro '{tool}' en el PATH. Instálalo y vuelve a correr.")


def read_editions(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            r = {k: (v or "").strip() for k, v in r.items()}
            if not r["edition_year"]:
                continue
            if f"{int(r['edition_year']) // 10 * 10}s" != r["decade"]:
                sys.exit(f"ERROR editions.csv línea {i}: {r['edition_year']} no corresponde a {r['decade']}")
            if not r["era"]:
                sys.exit(f"ERROR editions.csv línea {i}: falta 'era'")
            rows.append(r)
    years = [r["edition_year"] for r in rows]
    if len(years) != len(set(years)):
        sys.exit("ERROR: años duplicados en editions.csv")
    return rows


def find_video(videos_dir: Path, name: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


YTDLP_HINT = (
    "Si los errores son 'HTTP Error 403', casi siempre es yt-dlp desactualizado o sin runtime JS:\n"
    "    python3 -m pip install -U yt-dlp yt-dlp-ejs\n"
    "    brew install deno          # runtime de JavaScript que YouTube exige ahora\n"
    "  y vuelve a correr (los clips ya cortados no se repiten). Otras opciones:\n"
    "    --cookies-from-browser chrome     # o safari/firefox, usa tu sesión del navegador\n"
    "    python3 -m pip install -U --pre yt-dlp   # versión de desarrollo\n"
    "  Si nada funciona, descarga el video a mano y guárdalo como videos/wc<año>.mp4."
)


def _cleanup_partial(videos_dir: Path, name: str) -> None:
    """Borra restos de descargas fallidas (.part, .f398.mp4, .ytdl...) de ese video."""
    for p in videos_dir.glob(f"{name}.*"):
        if p.name.endswith((".part", ".ytdl")) or re.match(rf"^{re.escape(name)}\.f\d+", p.name):
            p.unlink(missing_ok=True)


def download(url: str, name: str, videos_dir: Path, cookies_browser: str | None = None) -> Path | None:
    if shutil.which("yt-dlp") is None:
        print(f"  [aviso] yt-dlp no instalado; descarga a mano en videos/{name}.mp4")
        return None
    # intento 1: video+audio separados (<=720p); intento 2: un solo archivo ya combinado
    for fmt in ("bv*[height<=720]+ba/b[height<=720]/b", "b[height<=720]/b"):
        cmd = ["yt-dlp", "--no-update", "--no-playlist", "--retries", "3",
               "-f", fmt, "--merge-output-format", "mp4",
               "-o", str(videos_dir / f"{name}.%(ext)s")]
        if cookies_browser:
            cmd += ["--cookies-from-browser", cookies_browser]
        r = subprocess.run(cmd + [url])
        video = find_video(videos_dir, name) if r.returncode == 0 else None
        if video:
            return video
        _cleanup_partial(videos_dir, name)
    print(f"  [aviso] falló la descarga de {name}; descárgalo a mano como videos/{name}.mp4")
    return None


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = round(float(num) / float(den), 2) if float(den or 1) else 0.0
    br = info["format"].get("bit_rate")
    return {
        "duration": float(info["format"]["duration"]),
        "src_width": v.get("width", ""), "src_height": v.get("height", ""),
        "src_fps": fps, "src_vcodec": v.get("codec_name", ""),
        "src_audio_sr": a.get("sample_rate", ""), "src_audio_ch": a.get("channels", ""),
        "src_bitrate_kbps": round(int(br) / 1000) if br else "",
        "has_audio": bool(a),
    }


def clip_windows(duration: float, head: float, tail: float, clip: float) -> list[float]:
    """Inicios de clips consecutivos entre head y duration-tail; descarta el sobrante final."""
    n = int((duration - head - tail) // clip)
    return [round(head + k * clip, 2) for k in range(max(n, 0))]


def cut(video: Path, start: float, dur: float, out_mp4: Path, out_wav: Path | None) -> None:
    base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(video), "-t", str(dur)]
    audio = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100"] if out_wav else ["-an"]
    subprocess.run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p"] + audio + [str(out_mp4)], check=True)
    if out_wav:  # directo de la fuente: conserva el ancho de banda original
        subprocess.run(base + ["-vn", "-ac", "1", "-ar", "44100", str(out_wav)], check=True)


def stats(manifest: Path) -> None:
    if not manifest.exists():
        sys.exit(f"No existe {manifest}; corre primero el script sin --stats.")
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
    for key in ("era", "decade"):
        g = defaultdict(lambda: {"eds": set(), "clips": 0, "h": [], "aud": 0})
        for r in rows:
            d = g[r[key]]
            d["eds"].add(r["edition_year"]); d["clips"] += 1
            d["aud"] += r["has_audio"] == "True"
            if r["src_height"]:
                d["h"].append(int(r["src_height"]))
        print(f"\n{key:10} {'ediciones':>9} {'clips':>6} {'con audio':>10} {'altura media':>13}")
        for k in sorted(g):
            d = g[k]
            h = sum(d["h"]) / len(d["h"]) if d["h"] else 0
            print(f"{k:10} {len(d['eds']):>9} {d['clips']:>6} {d['aud']:>10} {h:>13.0f}")
        if key == "era":
            low = [k for k in g if len(g[k]["eds"]) < 2]
            if low:
                print(f"[aviso] menos de 2 ediciones en {', '.join(sorted(low))}: no se puede validar por grupos")
    print(f"\nTOTAL: {len(rows)} clips, {len({r['edition_year'] for r in rows})} ediciones")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editions", type=Path, default=Path("editions.csv"))
    ap.add_argument("--videos-dir", type=Path, default=Path("videos"))
    ap.add_argument("--out-dir", type=Path, default=Path("clips"))
    ap.add_argument("--manifest", type=Path, default=Path("clips_manifest.csv"))
    ap.add_argument("--head", type=float, default=10.0, help="segundos iniciales a quitar (intro)")
    ap.add_argument("--tail", type=float, default=10.0, help="segundos finales a quitar (outro)")
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--cookies-from-browser", metavar="NAVEGADOR", default=None,
                    help="p. ej. chrome, safari, firefox (pasa tu sesión a yt-dlp)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        return stats(args.manifest)

    editions = read_editions(args.editions)
    if not args.dry_run:
        need("ffmpeg"); need("ffprobe")
    args.videos_dir.mkdir(exist_ok=True)

    rows, missing, failed_dl = [], [], 0
    for e in editions:
        year, name = e["edition_year"], f"wc{e['edition_year']}"
        video = find_video(args.videos_dir, name)
        if video is None and e["url"] and not args.no_download and not args.dry_run:
            print(f"[{year}] descargando...")
            video = download(e["url"], name, args.videos_dir, args.cookies_from_browser)
            failed_dl += video is None
        if video is None:
            missing.append(year)
            print(f"[{year}] SIN VIDEO" + (" (tiene URL: se descargará)" if e["url"] else " (y sin url)"))
            continue
        if args.dry_run:
            print(f"[{year}] video encontrado: {video.name}")
            continue

        meta = probe(video)
        dur = meta["duration"]
        if dur > 480:
            print(f"[{year}] aviso: dura {dur/60:.0f} min (se esperaban 1-3); ¿es el enlace correcto?")
        starts = clip_windows(dur, args.head, args.tail, args.clip_seconds)
        if not starts:
            print(f"[{year}] dura {dur:.0f} s: tras quitar {args.head+args.tail:.0f} s no cabe ningún clip")
            continue
        if not meta["has_audio"]:
            print(f"[{year}] aviso: sin pista de audio (se generan clips solo de video)")
        out = args.out_dir / e["era"]
        out.mkdir(parents=True, exist_ok=True)
        for k, s in enumerate(starts, start=1):
            cid = f"{name}_c{k:02d}"
            mp4 = out / f"{cid}.mp4"
            wav = out / f"{cid}.wav" if meta["has_audio"] else None
            if not (mp4.exists() and (wav is None or wav.exists())):
                cut(video, s, args.clip_seconds, mp4, wav)
            rows.append({
                "clip_id": cid, "edition_year": year, "decade": e["decade"], "era": e["era"],
                "start_s": s, "dur_s": args.clip_seconds,
                "video_path": str(mp4), "audio_path": str(wav) if wav else "",
                "has_audio": meta["has_audio"],
                **{k2: meta[k2] for k2 in MANIFEST_FIELDS if k2 in meta and k2.startswith("src_")},
                "src_duration_s": round(dur, 1),
            })
        print(f"[{year}] {len(starts)} clips de {args.clip_seconds:.0f} s")

    if args.dry_run:
        print(f"\nEdiciones sin video: {len(missing)}/{len(editions)}")
        return
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"\nListo: {len(rows)} clips -> {args.manifest}")
    if missing:
        print(f"Sin video ({len(missing)}): {', '.join(missing)}")
    if failed_dl:
        print(f"\n{failed_dl} descargas fallaron.\n  {YTDLP_HINT}")
    stats(args.manifest)


if __name__ == "__main__":
    main()