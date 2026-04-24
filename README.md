# AI Math Tutor for Early Learners (T3.1)

> AIMS KTT Hackathon · Tier 3 · EdTech · Edge AI · Child UX
> Adaptive · Offline · Multilingual (KIN / EN / FR + code-switch) · ≤ 75 MB on-device

A child-facing math tutor for ages 5–9 covering counting, number sense,
addition, subtraction, and word problems. Runs fully offline on a low-cost
Android tablet or Colab CPU. Adapts difficulty via knowledge tracing,
listens to child voices via a pitch-augmented Whisper-tiny, and reports
weekly progress to (often non-literate) parents through icons and
QR-to-audio.

---

## Reproduce in ≤ 2 commands (free Colab CPU)

```bash
pip install -r requirements.txt
python generate_curriculum.py --out data/T3.1_Math_Tutor/ && python demo.py
```

---

## Repo layout

```
.
├── tutor/                          # on-device Python package
│   ├── __init__.py
│   ├── inference.py                # stimulus → response → feedback loop (< 2.5 s)
│   ├── curriculum_loader.py        # loads curriculum.json, picks next item
│   ├── adaptive.py                 # BKT + tiny-GRU DKT + Elo baseline
│   ├── asr_adapt.py                # whisper-tiny int8 with child-voice augmentation
│   ├── lang_detect.py              # KIN / EN / FR / mix detection
│   ├── visual_count.py             # owlvit-tiny / blob-counter for "how many X?"
│   └── storage.py                  # encrypted SQLite (SQLCipher) + DP aggregation
├── data/T3.1_Math_Tutor/           # seeds + generator output
│   ├── curriculum_seed.json
│   ├── diagnostic_probes_seed.csv
│   ├── child_utt_sample_seed.csv
│   ├── parent_report_schema.json
│   └── child_utt_index.md
├── notebooks/
│   └── kt_eval.ipynb               # KT vs Elo AUC on held-out replay
├── scripts/                        # helper / one-off scripts
├── assets/                         # icons, rendered scene images for visual items
├── generate_curriculum.py          # ≥ 60-item curriculum generator
├── demo.py                         # Gradio child-facing demo (mic input)
├── parent_report.py                # weekly 1-pager generator from local SQLite
├── footprint_report.md             # `du -sh tutor/` + per-component breakdown
├── process_log.md                  # hour-by-hour timeline + LLM declarations
├── PLAN.md                         # 10-phase build plan
├── SIGNED.md                       # honor-code acknowledgement
├── LICENSE                         # MIT
└── requirements.txt
```

---

## Constraints (live gates)

| Constraint                        | Target          | Where checked            |
|-----------------------------------|-----------------|--------------------------|
| Total on-device footprint         | ≤ 75 MB         | `footprint_report.md`    |
| End-to-end stimulus → feedback    | < 2.5 s on CPU  | `tutor/inference.py`     |
| External calls at inference       | 0               | `tutor/inference.py`     |
| Dark patterns / streak loss       | None            | `tutor/inference.py`     |
| ε-DP budget per learner per week  | 1.0             | `tutor/storage.py`       |

---

## First 90 seconds (child onboarding)

A 6-year-old Kinyarwanda-speaker opens the tablet for the first time:

| t (s)  | What happens                                                                                          |
|--------|--------------------------------------------------------------------------------------------------------|
| 0–3    | A friendly voice in Kinyarwanda: *"Muraho! Reka dukine n'imibare."* ("Hello! Let's play with numbers.") |
| 3–10   | Three large coloured shapes appear; voice asks *"Erekana umuhondo"* ("Tap the yellow one").            |
| 10–20  | Child taps. On correct: cheerful chime + *"Yego! Ni umuhondo."* On miss: gentle re-prompt, no penalty. |
| 20–45  | Voice: *"Reba inka. Bangahe?"* ("Look at the cows. How many?"). Two cow icons.                         |
| 45–80  | Child says or taps the count. ASR + tap fall-through both work.                                        |
| 80–90  | Voice praises: *"Murakoze! Dukomeze."* ("Thank you! Let's continue."). Difficulty bumps based on BKT.  |
| Silent 10s at any point | Re-play current prompt once at 0.9× speed, then offer a tap-only fallback.               |

No login. No password. Pseudo-ID `learner_01..03` selected by tapping one of three animal avatars (see *tablet sharing* below).

---

## Tablet sharing model (3 children at a community centre)

- **Switch learners:** lock screen shows three animal avatars (lion / goat / bird). Child taps theirs → loads their encrypted profile from local SQLCipher DB. No password — the avatar tap is the identifier; this is a deliberate trade-off for 6-year-olds.
- **Privacy:** each learner row in `progress.db` is encrypted with a key derived from a salt stored in Android Keystore (or a file with `chmod 600` on dev). Cross-profile read is blocked at the storage layer.
- **Reboot graceful degradation:** on cold start the tutor loads the last-known curriculum + adapter weights from `tutor/`; if the SQLite DB is missing or corrupt it spins up a fresh profile and replays the first-90-seconds onboarding.
- **Power loss mid-session:** every answered item is `fsync`'d to disk before the next stimulus; worst case the child loses one item.

---

## Parent report (1-pager, 60-second read for non-literate parent)

`parent_report.py` produces a single PNG/PDF page per learner per week containing:

- 5 sub-skill bars (counting, number sense, +, −, word problems) — coloured cups filling left-to-right.
- A smiley / neutral / sad face per sub-skill.
- A QR code → 30-second audio summary in the parent's chosen language (KIN by default).
- One concrete suggestion icon (e.g. count goats together at home).

No numbers, no English text required. Schema in `data/T3.1_Math_Tutor/parent_report_schema.json`.

---

## Models

| Component        | Base                                           | Adaptation                          | Quantised size (target) |
|------------------|------------------------------------------------|-------------------------------------|-------------------------|
| Language head    | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`           | QLoRA on numeracy instructions      | ≤ 30 MB (Q4_K_M GGUF)   |
| ASR              | `openai/whisper-tiny`                          | Pitch +3..+6 semitones augment      | ≤ 25 MB (int8 ONNX)     |
| Visual counter   | `google/owlvit-base-patch32` *(or blob baseline)* | Frozen; only object class prompts | ≤ 15 MB (or 0 MB blob)  |
| KT model         | Tiny GRU (32-dim hidden, 5 skills)             | Trained from scratch on replay      | < 1 MB                  |

**Live `du -sh tutor/`:** see [footprint_report.md](footprint_report.md).

---

## Knowledge tracing

`tutor/adaptive.py` implements two estimators and an Elo baseline:

- **BKT:** standard 4-parameter (p_init, p_transit, p_slip, p_guess) per sub-skill, fit by EM on a held-out replay.
- **DKT:** 1-layer GRU, 32-dim hidden, input = (skill_id one-hot, correct), output = sigmoid over 5 skills.
- **Elo baseline:** student rating + item rating, K = 16.

`notebooks/kt_eval.ipynb` reports next-response AUC on the held-out 20% replay split.

---

## Scoring map (so the evaluator can find each criterion)

| Criterion                   | Weight | Where in repo                                        |
|-----------------------------|--------|------------------------------------------------------|
| Technical Quality & Code    | 20%    | `tutor/`, `demo.py`, `requirements.txt`              |
| Model / Algorithm Perf      | 20%    | `notebooks/kt_eval.ipynb`, `footprint_report.md`     |
| Data Handling & Methodology | 15%    | `generate_curriculum.py`, `data/T3.1_Math_Tutor/`    |
| Product & Business          | 20%    | This README sections above + `parent_report.py`      |
| Communication & Docs        | 15%    | README + `process_log.md`                            |
| Innovation & Problem-Solving| 10%    | Whole repo                                           |

---

## Status (scaffold)

This commit is the scaffold pushed at the start of the 4-hour window. Most modules are
intentional stubs that raise `NotImplementedError` so that `pytest -q` and `du -sh tutor/`
both run from minute one. See [PLAN.md](PLAN.md) for the build order.
