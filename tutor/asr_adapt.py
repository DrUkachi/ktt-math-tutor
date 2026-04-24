"""Child-voice ASR with whisper-tiny (int8 ONNX) + pitch/tempo augment.

Two responsibilities:

1. ``transcribe(wav, lang)`` — run quantised whisper-tiny on a 16 kHz wav array
   and return the most likely transcript.
2. ``augment_for_training(wav)`` — pitch-shift +3..+6 semitones and overlay
   classroom-noise clips from MUSAN. Used when retraining or fine-tuning.

The ONNX model is loaded lazily so importing ``tutor`` is cheap at startup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_MODEL_PATH = Path(__file__).parent / "whisper_int8.onnx"


class ChildASR:
    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or _MODEL_PATH)
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime required for ASR") from e
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"whisper int8 model missing at {self.model_path}. "
                "Run scripts/quantise_whisper.py first."
            )
        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def transcribe(self, wav: np.ndarray, lang: str = "en") -> str:
        """Return a transcript for ``wav`` (mono float32 @ 16 kHz)."""
        self._load()
        # TODO: pre-process to log-mel, run encoder + decoder, return tokens.
        raise NotImplementedError("phase-5 deliverable")


def augment_for_training(
    wav: np.ndarray,
    sr: int = 16000,
    pitch_semitones: float = 4.0,
    noise_clip: np.ndarray | None = None,
    snr_db: float = 12.0,
) -> np.ndarray:
    """Pitch-shift and (optionally) overlay classroom noise.

    Returns float32 audio at the same sample rate.
    """
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa required for augmentation") from e
    shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch_semitones)
    if noise_clip is None:
        return shifted.astype(np.float32)
    # Match length and SNR
    if len(noise_clip) < len(shifted):
        reps = int(np.ceil(len(shifted) / len(noise_clip)))
        noise_clip = np.tile(noise_clip, reps)
    noise_clip = noise_clip[: len(shifted)]
    sig_p = float(np.mean(shifted ** 2)) + 1e-12
    noi_p = float(np.mean(noise_clip ** 2)) + 1e-12
    scale = (sig_p / noi_p / (10 ** (snr_db / 10))) ** 0.5
    return (shifted + scale * noise_clip).astype(np.float32)
