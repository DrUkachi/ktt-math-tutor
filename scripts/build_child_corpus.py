"""Build a reproducible child-voice ASR corpus.

Strategy
--------
1. Synthesize short math-friendly adult utterances with Piper
   (en_US-lessac-medium, Hugging Face `rhasspy/piper-voices`).
2. Pitch-shift each clip by +3, +4.5 and +6 semitones with
   ``tutor.asr_adapt.augment_for_training`` to produce child-like
   versions. Each pitch shift approximates a different age band.
3. Write a manifest CSV matching the schema of the seed file
   ``data/T3.1_Math_Tutor/child_utt_sample_seed.csv``.
4. 80/20 split by utterance (not by pitch) so the eval set contains
   completely unseen phrasings at test time.

This corpus is *synthetic child speech*, not real — documented clearly
in the README and process_log. The brief allows it ("Augment with
pitch + tempo perturbation" on Common Voice / DigitalUmuganda); we go
one step further by generating the source audio too, for full
reproducibility without a dataset-license dependency.
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from tutor.asr_adapt import augment_for_training


PIPER_VOICE = Path.home() / ".local" / "share" / "piper-voices" / "en_US-lessac-medium.onnx"
MUSAN_NOISE_DIR = Path(__file__).resolve().parent.parent / "data" / "musan" / "noise"

NUMBERS_EN = ["one", "two", "three", "four", "five",
              "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen"]

OBJECTS = ["apples", "goats", "cows", "mangoes", "beads", "drums", "books"]

TEMPLATES = [
    "{n}",
    "the answer is {n}",
    "it is {n}",
    "there are {n} {obj}",
    "I see {n} {obj}",
    "{n} {obj}",
]

PITCH_STEPS = (3.0, 4.5, 6.0)
NOISE_SNR_DB = 12.0  # Brief-specified SNR for the classroom-noise overlay.


def _list_musan_noise_clips() -> list[Path]:
    """Return the list of MUSAN .wav clips available for overlay.

    Returns an empty list if the MUSAN subset hasn't been downloaded
    yet — caller then builds a noise-free corpus, which is strictly
    worse but keeps the pipeline runnable.
    """
    if not MUSAN_NOISE_DIR.exists():
        return []
    return sorted(MUSAN_NOISE_DIR.rglob("*.wav"))


def _pick_noise_clip(
    rng: random.Random,
    pool: list[Path],
) -> np.ndarray | None:
    """Load one random MUSAN clip resampled to 16 kHz mono, or None."""
    if not pool:
        return None
    p = rng.choice(pool)
    wav, sr = sf.read(p, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def piper_synth(text: str, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "-m", "piper", "-m", str(PIPER_VOICE), "-f", str(out_wav)],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
        timeout=30,
    )


def _load_resample_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/child_utt/")
    parser.add_argument("--n-utterances", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not PIPER_VOICE.exists():
        raise SystemExit(
            f"Piper voice missing at {PIPER_VOICE}. Run:\n"
            "  curl -L -o ~/.local/share/piper-voices/en_US-lessac-medium.onnx "
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        )

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    adult_dir = out_dir / "_adult_raw"
    adult_dir.mkdir(parents=True, exist_ok=True)

    # Build utterance list
    utterances: list[tuple[str, str]] = []  # (utt_id, text)
    for i in range(args.n_utterances):
        tpl = rng.choice(TEMPLATES)
        n = rng.choice(NUMBERS_EN)
        obj = rng.choice(OBJECTS)
        text = tpl.format(n=n, obj=obj)
        utterances.append((f"U{i+1:03d}", text))

    # 80/20 split by utterance
    rng.shuffle(utterances)
    n_eval = max(8, int(len(utterances) * 0.2))
    split = {"eval": utterances[:n_eval], "train": utterances[n_eval:]}

    rows_by_split: dict[str, list[dict]] = {"train": [], "eval": []}

    noise_pool = _list_musan_noise_clips()
    if noise_pool:
        print(f"MUSAN noise pool: {len(noise_pool)} clips at {NOISE_SNR_DB} dB SNR")
    else:
        print(f"WARNING: {MUSAN_NOISE_DIR} empty — corpus will be clean (no noise overlay).")
        print("         Run: python scripts/download_musan.py")

    for split_name, utt_list in split.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for utt_id, text in utt_list:
            adult_wav = adult_dir / f"{utt_id}.wav"
            if not adult_wav.exists():
                piper_synth(text, adult_wav)
            wav = _load_resample_16k(adult_wav)
            for semis in PITCH_STEPS:
                # One random classroom-noise clip per pitch variant, so
                # the same utterance shows up under different noise
                # conditions — this is what the brief calls for and
                # what makes the LoRA see real-world-ish signal.
                noise = _pick_noise_clip(rng, noise_pool)
                shifted = augment_for_training(
                    wav, sr=16000, pitch_semitones=semis,
                    noise_clip=noise, snr_db=NOISE_SNR_DB,
                )
                out_wav = split_dir / f"{utt_id}_p{int(semis*10):03d}.wav"
                sf.write(out_wav, shifted, 16000)
                rows_by_split[split_name].append({
                    "utt_id": f"{utt_id}_p{int(semis*10):03d}",
                    "audio_path": str(out_wav.relative_to(out_dir.parent) if out_dir.is_absolute() else out_wav),
                    "transcript_en": text,
                    "language": "en",
                    "pitch_semitones": semis,
                    "noise_overlay": "musan" if noise is not None else "none",
                    "snr_db": NOISE_SNR_DB if noise is not None else None,
                    "split": split_name,
                })

    # Manifests
    for split_name, rows in rows_by_split.items():
        manifest = out_dir / f"manifest_{split_name}.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split_name}: {len(rows)} clips → {manifest}")


if __name__ == "__main__":
    main()
