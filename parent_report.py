"""Generate the weekly 1-page parent report from the local SQLite store.

Output (per learner per week):
- 5 horizontal "filling-cup" bars, one per skill, coloured red/amber/green.
- Smiley / neutral / sad face per skill.
- A QR code → 30-second voiced summary in the parent's chosen language.
- One concrete suggestion icon (e.g. "count goats with your child at home").

Schema is in ``data/T3.1_Math_Tutor/parent_report_schema.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from tutor.curriculum_loader import SKILLS
from tutor.llm_head import LLMHead
from tutor.storage import ProgressStore


_DEFAULT_SUMMARY_FALLBACK = (
    "This week your child practised math. They showed strength on {best}. "
    "At home, try counting {best_words} together."
)

_BEST_WORDS = {
    "counting": "objects like fruits or goats",
    "number_sense": "numbers from 1 to 10",
    "addition": "adding small numbers",
    "subtraction": "taking things away",
    "word_problem": "stories with numbers",
}


def build_summary_text(skills_block: dict, lang: str = "en") -> str:
    """Return the narrative used for the QR-linked voiced summary.

    Tries the optional :class:`LLMHead` first; falls back to a safe
    deterministic template if the GGUF file is missing or generation
    fails. Output is kept short so the TTS rendering stays under
    ~30 seconds.
    """
    scores = {s: float(d["current"]) for s, d in skills_block.items()}
    head = LLMHead()
    out: str | None = None
    if head.available():
        try:
            out = head.weekly_summary(scores, lang=lang)
        except Exception:
            out = None
    if not out:
        best = max(scores, key=lambda s: scores[s])
        out = _DEFAULT_SUMMARY_FALLBACK.format(
            best=best.replace("_", " "),
            best_words=_BEST_WORDS.get(best, "numbers"),
        )

    # Keep at most the first 2 sentences: TinyLlama Q4 tends to drift
    # into meta-commentary after a coherent opener.
    sentences = [s.strip() for s in out.replace("\n", " ").split(".") if s.strip()]
    out = ". ".join(sentences[:2]) + ("." if sentences else "")
    return out.strip()


def render_summary_wav(text: str, out_wav: Path, lang: str = "en") -> bool:
    """Render ``text`` to a local WAV via Piper if a voice is available;
    else write 1 s of silence as a placeholder. Returns True if real
    audio was produced.
    """
    import subprocess
    import wave

    voice_dir = Path.home() / ".local" / "share" / "piper-voices"
    pattern = {"en": "en_*.onnx", "fr": "fr_*.onnx", "kin": "kin_*.onnx"}.get(lang, "en_*.onnx")
    voices = list(voice_dir.rglob(pattern)) if voice_dir.exists() else []

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if voices:
        try:
            subprocess.run(
                ["python", "-m", "piper", "-m", str(voices[0]), "-f", str(out_wav)],
                input=text.encode("utf-8"), check=True, capture_output=True, timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    # Silence fallback
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return False


def _aggregate(store: ProgressStore, learner_id: str, week_start: dt.date) -> dict:
    week_end = week_start + dt.timedelta(days=7)
    week_start_ts = dt.datetime.combine(week_start, dt.time.min).timestamp()
    week_end_ts = dt.datetime.combine(week_end, dt.time.min).timestamp()

    attempts = [
        a for a in store.replay(learner_id)
        if week_start_ts <= a.ts < week_end_ts
    ]
    by_skill: dict[str, list[bool]] = defaultdict(list)
    for a in attempts:
        by_skill[a.skill_id].append(a.correct)

    skills_block: dict[str, dict[str, float]] = {}
    for s in SKILLS:
        results = by_skill.get(s, [])
        current = (sum(results) / len(results)) if results else 0.0
        skills_block[s] = {"current": round(current, 3), "delta": 0.0}

    return {
        "learner_id": learner_id,
        "week_starting": week_start.isoformat(),
        "sessions": len({int(a.ts // 3600) for a in attempts}),
        "skills": skills_block,
        "icons_for_parent": ["overall_arrow", "best_skill", "needs_help"],
        "voiced_summary_audio": f"reports/{learner_id}/{week_start}/summary.wav",
    }


def render_png(report: dict, out_path: Path) -> None:
    """Render the 1-pager as a PNG. Uses Pillow only — no matplotlib bloat."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 800, 1000
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        body_font = ImageFont.truetype("arial.ttf", 28)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((40, 30), f"Week of {report['week_starting']}", font=title_font, fill="black")
    draw.text((40, 80), f"Sessions: {report['sessions']}", font=body_font, fill="black")

    y = 160
    for skill, data in report["skills"].items():
        score = data["current"]
        face = "🙂" if score >= 0.7 else "😐" if score >= 0.4 else "😟"
        draw.text((40, y), f"{face}  {skill.replace('_', ' ').title()}",
                  font=body_font, fill="black")
        bar_x, bar_y, bar_w, bar_h = 360, y + 8, 360, 28
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline="black")
        fill_w = int(bar_w * score)
        colour = "#2e9b3a" if score >= 0.7 else "#e89b1d" if score >= 0.4 else "#c0392b"
        draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=colour)
        y += 70

    try:
        import qrcode
        qr = qrcode.make(report["voiced_summary_audio"]).resize((180, 180))
        img.paste(qr, (W - 220, H - 220))
        draw.text((W - 220, H - 40), "Scan for voice summary",
                  font=body_font, fill="black")
    except ImportError:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-id", required=True)
    parser.add_argument("--week-start", default=None,
                        help="YYYY-MM-DD; defaults to last Monday.")
    parser.add_argument("--out-png", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.week_start:
        week_start = dt.date.fromisoformat(args.week_start)
    else:
        today = dt.date.today()
        week_start = today - dt.timedelta(days=today.weekday())

    store = ProgressStore()
    report = _aggregate(store, args.learner_id, week_start)

    # Generate the voiced-summary narrative and render it to a WAV that
    # the QR code will point at.
    summary_text = build_summary_text(report["skills"], lang="en")
    report["voiced_summary_text"] = summary_text
    out_wav = Path(f"reports/{args.learner_id}/{week_start}/summary.wav")
    rendered = render_summary_wav(summary_text, out_wav, lang="en")
    report["voiced_summary_audio"] = str(out_wav)
    report["voiced_summary_rendered"] = rendered

    out_json = Path(args.out_json or
                    f"reports/{args.learner_id}/{week_start}/report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    out_png = Path(args.out_png or
                   f"reports/{args.learner_id}/{week_start}/report.png")
    render_png(report, out_png)
    print(f"Wrote {out_json} and {out_png}")
    print(f"Summary: {summary_text}")
    print(f"Audio: {out_wav} (real TTS={rendered})")


if __name__ == "__main__":
    main()
