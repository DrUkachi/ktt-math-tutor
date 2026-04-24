"""Gradio child-facing demo. Mic input + tap fallback.

Run:
    python demo.py            # opens http://127.0.0.1:7860
    python demo.py --share    # public link (Colab)

Notes
-----
- The full inference loop is in :mod:`tutor.inference`. This script only
  wires it to a Gradio UI sized for a 6-year-old: large emoji buttons,
  voice or single-tap response, no text-entry box.
- ASR is best-effort. If whisper-tiny is not available, the script falls
  back to the tap input only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from tutor.inference import Tutor
from tutor.lang_detect import detect


CURRICULUM_PATH = Path("data/T3.1_Math_Tutor/curriculum.json")
SEED_PATH = Path("data/T3.1_Math_Tutor/curriculum_seed.json")


def _load_tutor(learner_id: str) -> Tutor:
    path = CURRICULUM_PATH if CURRICULUM_PATH.exists() else SEED_PATH
    return Tutor(learner_id=learner_id, curriculum_path=path)


def _maybe_transcribe(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    try:
        import numpy as np
        import soundfile as sf
        from tutor.asr_adapt import ChildASR

        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            except ImportError:
                pass
        return ChildASR().transcribe(np.asarray(wav, dtype=np.float32))
    except Exception:
        return ""  # Tap fallback path will still work.


def cycle(audio_path: str | None, tap_response: str, age_band: str, learner_id: str):
    tutor = _load_tutor(learner_id)
    spoken = _maybe_transcribe(audio_path)
    response_text = spoken or tap_response or ""
    result = tutor.step(age_band=age_band, response_text=response_text)
    lang = result.lang_detected if response_text else "kin"
    feedback_text = tutor.feedback(result.item, result.correct, lang)
    return (
        result.item.stem(lang),
        feedback_text,
        f"{result.response_ms} ms · lang={lang} · skill={result.item.skill}",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Math Tutor (T3.1)") as ui:
        gr.Markdown("# 🦁 🐐 🐦  Tap your animal to start")
        with gr.Row():
            learner = gr.Radio(
                choices=["learner_lion", "learner_goat", "learner_bird"],
                value="learner_lion", label="Who is playing?",
            )
            age_band = gr.Radio(
                choices=["5-6", "6-7", "7-8", "8-9"],
                value="6-7", label="Age band",
            )
        audio = gr.Audio(sources=["microphone"], type="filepath",
                         label="Speak your answer (or tap below)")
        tap = gr.Radio(choices=[str(i) for i in range(1, 11)],
                       label="Or tap a number")
        go = gr.Button("Submit", variant="primary", size="lg")
        prompt_box = gr.Textbox(label="Prompt the tutor asked")
        feedback_box = gr.Textbox(label="Feedback")
        meta_box = gr.Textbox(label="Diagnostics")
        go.click(cycle, inputs=[audio, tap, age_band, learner],
                 outputs=[prompt_box, feedback_box, meta_box])
    return ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(share=args.share)


if __name__ == "__main__":
    # Silence the unused-import warning when only the lib is checked.
    _ = detect
    main()
