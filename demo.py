"""Gradio child-facing demo. Mic input + tap fallback.

Flow (as a child sees it):

  1. Pick an avatar + age band.
  2. Hit **Start** -> a math question appears on screen, the tutor
     plays the prompt audio (if TTS cache is populated).
  3. Speak the answer (or tap a number if the mic is blocked or too
     noisy).
  4. Hit **Submit** -> the tutor scores the answer against the *same*
     question the child just saw, shows feedback, and the next
     question is queued.

Run:
    python demo.py              # opens http://127.0.0.1:7860
    python demo.py --share      # public *.gradio.live URL (Colab)
    python demo.py --warm       # pre-warms the ASR model before launching
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from tutor.curriculum_loader import Item
from tutor.inference import Tutor
from tutor.lang_detect import detect


REPO_ROOT = Path(__file__).resolve().parent
CURRICULUM_PATH = REPO_ROOT / "data/T3.1_Math_Tutor/curriculum.json"
SEED_PATH = REPO_ROOT / "data/T3.1_Math_Tutor/curriculum_seed.json"
TTS_CACHE = REPO_ROOT / "tts_cache"
ASSETS_DIR = REPO_ROOT / "assets"

_TUTORS: dict[str, Tutor] = {}


def _load_tutor(learner_id: str) -> Tutor:
    if learner_id not in _TUTORS:
        path = CURRICULUM_PATH if CURRICULUM_PATH.exists() else SEED_PATH
        _TUTORS[learner_id] = Tutor(learner_id=learner_id, curriculum_path=path)
    return _TUTORS[learner_id]


def _prompt_audio_path(item: Item, lang: str = "en") -> str | None:
    # TTS cache is rendered by scripts/render_tts.py using uppercase item IDs.
    p = TTS_CACHE / lang / f"{item.id}.wav"
    return str(p) if p.exists() else None


def _scene_image_path(item: Item) -> str | None:
    """Return the rendered-scene PNG ONLY for skills where it's needed.

    - counting: required — 'How many apples?' is unanswerable without
      the picture.
    - number_sense: helpful for young learners — '4 or 7?' with a
      visual of the two numerals.
    - addition / subtraction / word_problem: text is self-contained;
      a picture of beads is noise, especially at small mobile sizes.
    """
    if item.skill not in ("counting", "number_sense"):
        return None
    if not item.visual:
        return None
    p = ASSETS_DIR / f"{item.visual}.png"
    return str(p) if p.exists() else None


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
        return ""  # Tap fallback always works.


def ask_next(learner_id: str, age_band: str):
    """Pick and display the next item. Returns (prompt_text, scene_image,
    tts_path, item_id, cleared-feedback, cleared-diagnostics)."""
    tutor = _load_tutor(learner_id)
    item = tutor.ask(age_band=age_band)
    return (
        item.stem("en"),
        _scene_image_path(item),          # scene PNG beside the question
        _prompt_audio_path(item, "en"),
        item.id,                          # stashed in gr.State
        "",                               # clear previous feedback
        f"item={item.id} · skill={item.skill} · answer_hidden",
    )


def submit_answer(audio_path: str | None, tap_response: str,
                  age_band: str, learner_id: str, pending_item_id: str):
    """Score the currently-displayed item. Then automatically ask the next
    one so the UX flows like a real tutor session."""
    tutor = _load_tutor(learner_id)
    if not pending_item_id:
        # No active question yet — treat Submit as "please start a session."
        return ask_next(learner_id, age_band)

    item = tutor.curriculum.get(pending_item_id)
    spoken = _maybe_transcribe(audio_path)
    # Precedence: spoken > tapped. If the mic returned a transcript,
    # use it; otherwise the tap value (or empty string if nothing).
    response_text = spoken or tap_response or ""
    cycle = tutor.answer(item, response_text)
    lang = cycle.lang_detected if response_text else "kin"
    feedback_text = tutor.feedback(cycle.item, cycle.correct, lang)

    # Queue up the next question straight away.
    next_item = tutor.ask(age_band=age_band)
    diagnostics = (
        f"Last: {item.id} ({item.skill}) · "
        f"{'correct' if cycle.correct else 'not quite'} · "
        f"{cycle.response_ms} ms · lang={lang}"
    )
    return (
        next_item.stem("en"),
        _scene_image_path(next_item),
        _prompt_audio_path(next_item, "en"),
        next_item.id,
        feedback_text,
        diagnostics,
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

        pending = gr.State(value="")  # current item's id

        with gr.Group():
            prompt_box = gr.Textbox(
                label="Question",
                value="Tap Start to begin.",
                interactive=False, lines=3, max_lines=4,
            )
            with gr.Row():
                prompt_image = gr.Image(
                    label="Look at the picture",
                    interactive=False, height=260, show_label=True,
                )
                prompt_audio = gr.Audio(
                    label="Listen to the question",
                    interactive=False, autoplay=True,
                )
            start_btn = gr.Button("▶ Start / Next question",
                                  variant="secondary", size="lg")

        with gr.Group():
            audio_in = gr.Audio(sources=["microphone"], type="filepath",
                                label="Speak your answer")
            # 0–20 tap covers 89 % of the curriculum (71 of 80 items).
            # The 9 items with answers > 20 are all age band 8–9, and
            # those older learners can use the mic — no need for a
            # second input widget cluttering the UI for 6-year-olds.
            tap = gr.Radio(
                choices=[str(i) for i in range(0, 21)],
                label="Or tap a number (0–20)", value=None,
            )
            submit_btn = gr.Button("Submit", variant="primary", size="lg")

        feedback_box = gr.Textbox(label="Feedback", interactive=False)
        diag_box = gr.Textbox(label="Diagnostics", interactive=False)

        start_btn.click(
            ask_next,
            inputs=[learner, age_band],
            outputs=[prompt_box, prompt_image, prompt_audio, pending,
                     feedback_box, diag_box],
        )
        submit_btn.click(
            submit_answer,
            inputs=[audio_in, tap, age_band, learner, pending],
            outputs=[prompt_box, prompt_image, prompt_audio, pending,
                     feedback_box, diag_box],
        )
    return ui


def _warm_asr() -> None:
    try:
        import numpy as np
        from tutor.asr_adapt import ChildASR
        ChildASR().transcribe(np.zeros(16000, dtype="float32"))
        print("[warm] ASR ready.")
    except Exception as e:  # pragma: no cover
        print(f"[warm] skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--warm", action="store_true",
                        help="Pre-load the ASR model before serving.")
    args = parser.parse_args()
    if args.warm:
        _warm_asr()
    build_ui().launch(share=args.share)


if __name__ == "__main__":
    _ = detect  # silence unused-import warning
    main()
