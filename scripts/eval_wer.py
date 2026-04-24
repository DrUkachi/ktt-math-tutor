"""Compute WER on the child-voice eval manifest.

Usage:
    python scripts/eval_wer.py                                      # baseline: vanilla whisper-tiny
    python scripts/eval_wer.py --model-path tutor/asr_model/ct2     # after LoRA + CT2 export

Prints WER, per-clip predictions, and a JSON summary to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from jiwer import wer as jiwer_wer

from tutor.asr_adapt import ChildASR


def _load_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _normalise(text: str) -> str:
    """WER normalisation: lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/child_utt/manifest_eval.csv")
    parser.add_argument("--model-path", default=None,
                        help="Path to CT2 model dir (post-LoRA).")
    parser.add_argument("--vanilla", action="store_true",
                        help="Force vanilla whisper-tiny from ~/.cache (for baseline reproduction).")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    # Resolve which model ChildASR should actually load.
    from tutor.asr_adapt import _TUNED_MODEL
    if args.vanilla:
        # Bypass tutor/asr_model/ by pointing at a definitely-missing path;
        # ChildASR then falls through to downloading vanilla whisper-tiny.
        import os
        os.environ["WHISPER_CACHE_DIR"] = os.path.expanduser("~/.cache/whisper-tiny-ct2")
        # Trick the resolution by using a sentinel path that doesn't exist.
        asr = ChildASR()
        # Override: replace _TUNED_MODEL check by re-creating without it.
        asr.model_path = None
        # Monkey-patch: stop _load from auto-picking the tuned model.
        import tutor.asr_adapt as _am
        _am._TUNED_MODEL = Path("/nonexistent/force-vanilla")
        label = "whisper-tiny (vanilla int8, ~/.cache)"
    elif args.model_path:
        asr = ChildASR(model_path=Path(args.model_path))
        label = args.model_path
    else:
        asr = ChildASR()
        if _TUNED_MODEL.exists() and (_TUNED_MODEL / "model.bin").exists():
            label = f"{_TUNED_MODEL} (child-voice LoRA, int8 CT2)"
        else:
            label = "whisper-tiny (vanilla int8, ~/.cache)"
    asr._load()

    refs: list[str] = []
    hyps: list[str] = []
    per_clip: list[dict] = []
    t_total = 0.0

    with open(args.manifest, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        wav = _load_16k(Path(row["audio_path"]))
        t0 = time.perf_counter()
        hyp = asr.transcribe(wav, lang=row.get("language", "en"))
        dt = time.perf_counter() - t0
        t_total += dt
        ref_norm = _normalise(row["transcript_en"])
        hyp_norm = _normalise(hyp)
        refs.append(ref_norm)
        hyps.append(hyp_norm)
        per_clip.append({
            "utt_id": row["utt_id"],
            "ref": ref_norm,
            "hyp": hyp_norm,
            "pitch": float(row.get("pitch_semitones", 0)),
            "t_ms": round(dt * 1000, 1),
        })

    wer_score = jiwer_wer(refs, hyps)
    mean_latency = (t_total / len(rows) * 1000) if rows else 0.0

    summary = {
        "model": label,
        "n_clips": len(rows),
        "wer": round(wer_score, 4),
        "mean_transcribe_ms": round(mean_latency, 1),
    }

    print("=" * 60)
    print(f"Model:         {summary['model']}")
    print(f"Clips:         {summary['n_clips']}")
    print(f"WER:           {summary['wer']:.4f}")
    print(f"Mean latency:  {summary['mean_transcribe_ms']:.1f} ms / clip")
    print("=" * 60)
    for r in per_clip[:8]:
        print(f"  {r['utt_id']} (p{r['pitch']}): "
              f"ref={r['ref']!r:32s} hyp={r['hyp']!r}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
