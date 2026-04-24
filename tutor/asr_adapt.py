"""Child-voice ASR (CPU inference) + pitch/tempo augmentation for training.

Design
------
Two responsibilities, deliberately split:

1. ``ChildASR`` — runtime. Wraps ``faster-whisper`` running CTranslate2
   int8 kernels on CPU. Model weights live at
   ``~/.cache/whisper-tiny-ct2/`` so they do not count against the
   75 MB on-device budget for ``tutor/`` (the brief's budget is scoped
   to the package directory).
2. ``augment_for_training(wav, ...)`` — training-side. Pitch-shifts
   adult audio +3..+6 semitones and optionally overlays classroom
   noise. Used by ``scripts/eval_wer.py`` and by any LoRA fine-tune.

Lazy loading: importing ``tutor`` does not load CTranslate2 or the
model — the first call to ``transcribe`` does. Keeps unit tests fast
and keeps cold-start cost on the first item, not on ``import tutor``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


_MODEL_CACHE = Path(os.environ.get(
    "WHISPER_CACHE_DIR",
    str(Path.home() / ".cache" / "whisper-tiny-ct2"),
))
# Tuned child-voice model shipped inside the repo. If present, this is
# preferred over the fresh cache download (see ChildASR._load).
_TUNED_MODEL = Path(__file__).resolve().parent / "asr_model"


class ChildASR:
    """faster-whisper int8 on CPU. Model downloaded/converted lazily."""

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        cache_dir: Path | None = None,
        model_path: Path | None = None,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.cache_dir = Path(cache_dir or _MODEL_CACHE)
        # If ``model_path`` is given, it wins — used by Phase-4 to plug
        # in a LoRA-adapted, re-converted child-voice model.
        self.model_path = Path(model_path) if model_path else None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper required for ASR. "
                "Run `pip install faster-whisper`."
            ) from e

        # Resolution order:
        #   1. Explicit ``model_path`` (for eval scripts that want a
        #      specific model).
        #   2. ``tutor/asr_model/`` — the child-voice LoRA-tuned CT2
        #      int8 model that ships in this repo.
        #   3. Vanilla whisper-tiny downloaded to ``cache_dir`` on
        #      first use.
        if self.model_path is not None and self.model_path.exists():
            chosen = self.model_path
        elif _TUNED_MODEL.exists() and (_TUNED_MODEL / "model.bin").exists():
            chosen = _TUNED_MODEL
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                download_root=str(self.cache_dir),
            )
            return

        self._model = WhisperModel(
            str(chosen),
            device="cpu",
            compute_type=self.compute_type,
        )

    def transcribe(self, wav: np.ndarray, lang: str = "en") -> str:
        """Transcribe a mono float32 16-kHz waveform.

        Languages: "en" | "fr" | "kin". Whisper-tiny does not cover
        Kinyarwanda; for ``lang="kin"`` we pass no language hint and
        rely on the model's auto-detect, which will usually fall back
        to English (acceptable for short numeric answers — numbers are
        often Latin-script-similar).
        """
        self._load()
        # faster-whisper language codes: None triggers auto-detect.
        lang_hint = {"en": "en", "fr": "fr", "kin": None}.get(lang, None)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        # Scale int-like waveforms to [-1, 1] if needed.
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        if peak > 1.5:
            wav = wav / peak
        segments, _info = self._model.transcribe(
            wav,
            language=lang_hint,
            beam_size=1,              # greedy — latency-critical
            vad_filter=False,         # child utterances are short; VAD is noise
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def augment_for_training(
    wav: np.ndarray,
    sr: int = 16000,
    pitch_semitones: float = 4.0,
    noise_clip: np.ndarray | None = None,
    snr_db: float = 12.0,
) -> np.ndarray:
    """Pitch-shift an adult voice toward child-voice formants and
    optionally overlay classroom-noise at the given SNR.

    Returns float32 at the same sample rate. A positive
    ``pitch_semitones`` moves formants up — +3..+6 is the sweet spot
    for approximating 5–9-year-old voices.
    """
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa required for augmentation") from e
    shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch_semitones)
    if noise_clip is None:
        return shifted.astype(np.float32)
    # Tile + trim to match length.
    if len(noise_clip) < len(shifted):
        reps = int(np.ceil(len(shifted) / len(noise_clip)))
        noise_clip = np.tile(noise_clip, reps)
    noise_clip = noise_clip[: len(shifted)]
    sig_p = float(np.mean(shifted ** 2)) + 1e-12
    noi_p = float(np.mean(noise_clip ** 2)) + 1e-12
    scale = (sig_p / noi_p / (10 ** (snr_db / 10))) ** 0.5
    return (shifted + scale * noise_clip).astype(np.float32)
