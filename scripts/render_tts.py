"""Populate ``tts_cache/<lang>/<id>.wav`` for every curriculum item.

Tries Piper TTS (``piper-tts``) first — if voice .onnx files are present
under ``~/.local/share/piper-voices/`` — and falls back to silent WAV
placeholders so every curriculum item has a resolvable ``tts_<lang>``
path (matches the schema expected by the demo).

The TTS cache is excluded from the 75 MB on-device budget per the brief.

Run:
    python scripts/render_tts.py --curriculum data/T3.1_Math_Tutor/curriculum.json --out tts_cache/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from pathlib import Path


_VOICE_DIR = Path.home() / ".local" / "share" / "piper-voices"
_VOICE_PATTERNS = {
    "en": "en_US*.onnx",
    "fr": "fr_FR*.onnx",
    "kin": "kin_RW*.onnx",  # unlikely to exist; falls through to silence.
}


def _find_voice(lang: str) -> Path | None:
    if not _VOICE_DIR.exists():
        return None
    matches = list(_VOICE_DIR.rglob(_VOICE_PATTERNS[lang]))
    return matches[0] if matches else None


def _write_silence(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n_samples)


def _piper_render(voice: Path, text: str, out_wav: Path) -> bool:
    try:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "-m", "piper", "-m", str(voice), "-f", str(out_wav)],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=30,
        )
        return out_wav.exists() and out_wav.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def render_one(item: dict, lang: str, out_dir: Path, voice: Path | None) -> Path:
    stem = item.get(f"stem_{lang}") or item.get("stem_en", "")
    out_path = out_dir / lang / f"{item['id']}.wav"
    rendered = False
    if voice is not None and stem:
        rendered = _piper_render(voice, stem, out_path)
    if not rendered:
        _write_silence(out_path, seconds=1.0)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="data/T3.1_Math_Tutor/curriculum.json")
    parser.add_argument("--out", default="tts_cache/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    with open(args.curriculum, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    voices = {lang: _find_voice(lang) for lang in ("en", "fr", "kin")}
    for lang, v in voices.items():
        print(f"  voice[{lang}] = {v or 'silence-placeholder'}")

    n = 0
    for it in items:
        for lang in ("en", "fr", "kin"):
            render_one(it, lang, out_dir, voices[lang])
            n += 1
    print(f"Wrote {n} WAV files under {out_dir}")


if __name__ == "__main__":
    main()
