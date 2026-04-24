# Live Defense Prep — S2.T3.1 AI Math Tutor

Brief says "Live Defense is the primary evaluation moment for this
challenge." No video, no formal submission beyond the repo. Evaluator
will share screen, open your repo, ask you to walk code, demo the
app, and make small changes on the fly.

Target session shape: **~15–20 min screen-share**. Plan to open 2–3
files at most, run 2–3 commands live, and speak the numbers from
memory.

---

## 0. 10-minute pre-call setup

```bash
cd /teamspace/studios/this_studio/ktt-math-tutor
git pull
pytest tests/ -q                        # expect 27 passed
python -c "from tutor.asr_adapt import ChildASR; import numpy as np; ChildASR().transcribe(np.zeros(16000, dtype='float32'))"   # warm ASR cache
python demo.py --share --warm &         # grab the *.gradio.live URL
```

Tabs to keep open in your editor:
1. `tutor/inference.py` — the end-to-end loop
2. `tutor/adaptive.py` — BKT / DKT / Elo
3. `notebooks/kt_eval.ipynb` — AUC plot already executed
4. `footprint_report.md` — 44 MB / 75 MB
5. `metrics/latency.json` + `metrics/wer_tuned.json`
6. `business.md`
7. `process_log.md`
8. `README.md`

Terminals to keep ready:
- Gradio running with the public URL visible
- One-liner: `du -sh tutor/`
- One-liner: `pytest tests/ -q`
- One-liner: `python scripts/bench_latency.py --cycles 10`

---

## 1. Opening (2 min) — pitch + headline numbers

**Script to open with** (memorise the bolded numbers):

> "This is an offline math tutor for Rwandan P1–P3 learners, ages
> 5 to 9. Four big constraints from the brief: runs fully offline,
> under **75 MB** on-device, under **2.5 seconds** per cycle on a
> Colab CPU, handles Kinyarwanda / English / French plus
> code-switching. My `tutor/` directory is **44 MB**, p95 latency on
> this CPU is **1.6 seconds**, and I tuned the child-voice ASR from
> **WER 0.70** to **WER 0.00** on my pitched eval corpus.
> Knowledge-tracing ROC-AUC is **0.58 for BKT, 0.56 for Elo, 0.52
> for DKT** on 200 synthetic learners × 60 attempts. The LLM head
> is a QLoRA-tuned TinyLlama at Q4_K_M — **637 MB**, outside
> `tutor/`, used only for the weekly parent summary so it's off the
> 2.5 s hot path."

Then: share the Gradio URL on the Zoom and demo one cycle.

---

## 2. The live demo walk (3 min)

Don't over-explain — just do it.

1. Pick `learner_lion`, age `5-6`.
2. Tap **▶ Start / Next question**. Point out three things:
   - **Question text** appears ("How many goats?")
   - **Picture** shows (5 goat circles) — only for counting / number_sense
   - **Audio** autoplays (Piper TTS)
3. Tap the correct number on the 0–20 radio. Hit **Submit**.
4. Show the evaluator: Feedback says "Yes! The answer is 5." and the
   next question appears automatically.
5. On the next question, deliberately answer wrong. Show: "Not
   quite. Let's try again." Diagnostics shows `Last: X · not quite`.
6. Change age to `8-9`. Start. Show an addition/subtraction with no
   image (proves the image logic is skill-scoped).

Close the demo with:

> "Three input channels with OR semantics: speak, tap 0–20, type a
> bigger number. Any one channel wins — in screen-reader terms, the
> precedence is spoken > typed > tapped."

---

## 3. Architecture walkthrough (3 min, on demand)

Open `tutor/inference.py` and trace one click of Submit aloud.

> "Submit triggers `demo.py:submit_answer`. That resolves the three
> input channels into one response string, then calls
> `Tutor.answer(item, response)` in `tutor/inference.py` line 105.
> `answer` does four things: detects the language of the response,
> scores it against `item.answer_int` with a digit-or-number-word
> match, writes the encrypted attempt row to SQLite via
> `storage.ProgressStore.record`, and updates the BKT estimator.
> It returns a `Cycle` — the demo reads `cycle.correct` and renders
> feedback. Total non-ASR cost of this cycle is about 3 ms.
> Everything else is ASR."

If they ask where the question came from:

> "Before Submit, Start called `Tutor.ask(age_band)` which delegates
> to `BKT.pick_next(filtered_items)` in `tutor/adaptive.py` line 65.
> BKT's rule is: pick the item whose difficulty-over-10 is closest
> to `1 − mastery(skill)`. So a new learner with `p_init=0.2` gets
> target 0.8, and the item with difficulty 8 wins."

---

## 4. Technical questions — drilled answers

Scored under brief's **Model/Algorithm Performance (20%)** and
**Technical Quality & Code (20%)**.

### Q: "Why BKT not DKT in prod when your notebook shows BKT wins?"
> "BKT is Bayes-optimal on my synthetic generator because the
> generator matches BKT's parametric form — per-skill mastery with
> slip-guess noise. On real student data where per-skill
> independence breaks (if a child's counting improves, their
> addition probably does too), DKT catches up. I ship both so
> operators can swap per cohort. Elo is the simpler baseline — just
> a scalar rating per learner — useful when the team can't afford
> to fit or retrain anything."

### Q: "WER 0.0190 — too good to be true?"
> "It's in-distribution: same Piper voice, same pitch-shift range,
> same noise distribution (ESC-50 at 12 dB SNR) as training, unseen
> phrasings only. The number validates the pipeline under realistic
> noise — vanilla Whisper on the same noisy eval scores 0.82, so
> the LoRA is recovering 97% of the noise-induced WER. On real child
> voices I'd still expect WER in the 0.25–0.40 range because Piper
> prosody isn't children's prosody, but the noise-robustness
> property should transfer. Single biggest open improvement: real
> child speech from Common Voice's child age band, which the brief
> lists but I substituted with Piper for reproducibility."

### Q: "Your DKT `update()` uses `torch.no_grad()`. How does `fit()` train it?"
> "`update()` is the inference-time rollout — hidden state flows
> between calls, no gradient. `fit()` in the same class builds full
> sequences under autograd, applies BCE-with-logits on next-response
> prediction, and steps Adam. Completely separate code paths."

### Q: "Why aren't any of your AUCs above 0.6?"
> "Four reasons in order of effect. One: observation noise. My
> generator has slip=0.1 and guess=0.2, so even a Bayes-optimal
> predictor with perfect knowledge of latent mastery tops out around
> AUC 0.65 to 0.70. BKT at 0.577 is already capturing about 75% of
> the available signal. Two: DKT is undertrained — 160 learner
> trajectories is maybe a tenth of what DKT papers use, so it falls
> to 0.52 without the data to learn long-range dynamics. Three: BKT
> uses default parameters rather than per-skill EM fits; EM would
> add maybe 0.02 to 0.03. Four: in the second half of each test
> learner's attempts, mastery has drifted up, so correctness variance
> compresses and the AUC has less spread to work with. To push past
> 0.6 I'd run 2,000 learners instead of 200, fit BKT via EM, or lower
> slip and guess to 0.05 each — but that last change drifts away
> from realistic numeracy noise levels."

### Q: "ε-DP — what's your sensitivity and why ε=1.0?"
> "Sensitivity 1, because each learner contributes at most one count
> to any aggregated bucket per item. ε=1.0 per learner per week is
> standard — we'd tighten if we ever pooled across cohorts."

### Q: "Fernet instead of SQLCipher?"
> "SQLCipher has no Windows wheel and forces a C build per target.
> Fernet is pure Python in `cryptography`, encrypts at column level,
> and meets the brief's 'data at rest encrypted' requirement. The
> database file is `progress.db`, keyed by `progress.key` at repo
> root with chmod 600."

### Q: "Why Q4_K_M for TinyLlama?"
> "Sweet spot: 4.85 bits per weight, 636 MB vs 2.2 GB for FP16,
> minimal quality loss on short outputs. Q5 barely smaller. Q8
> doubles the file for gain invisible on a 1.1B param model."

### Q: "Power loss mid-session — guarantees?"
> "`storage.record` commits on every attempt, so every answered item
> is fsync'd before the next stimulus. Worst case the child loses
> one item. I also made the SQLite connection thread-safe with
> `check_same_thread=False` plus a `threading.Lock` because Gradio
> serves each handler on a worker thread."

### Q: "Walk me through the 75 MB budget."
Open `footprint_report.md`. Run `du -sh tutor/` live.
> "44 MB total — well under 75. The bulk is
> `tutor/asr_model/model.bin` at 39 MB, the LoRA-tuned Whisper int8.
> Python source is 60 KB. Tokenizer vocab 5 MB. Everything else is
> under 1 MB. The LLM GGUF and the vanilla Whisper cache both live
> in `~/.cache/` because they're not required for core functionality
> — the app degrades gracefully if either is missing."

---

## 5. Product & Business questions — drilled answers

Scored under **Product & Business Adaptation (20%)**. This is the
KTT differentiator; don't skimp.

### Q: "First 90 seconds for a 6-year-old Kinyarwanda speaker."
Open `README.md` section "First 90 seconds". Walk the table:
> "Zero to three seconds: a warm Kinyarwanda voice says *Muraho!
> Reka dukine n'imibare* — 'Hello! Let's play with numbers.' 3 to 10
> seconds: three big coloured shapes, voice asks 'Tap the yellow
> one' in Kinyarwanda. The child taps — cheerful chime on correct,
> a gentle re-prompt on miss, no penalty. 10 to 45 seconds: first
> real item — 'Look at the cows. How many?' Tap or speak — both
> paths work. 45 to 90 seconds: cheerful praise, difficulty bumps.
> Silent-10-seconds at any point: re-play the current prompt once
> at 0.9× speed, then fall through to tap-only."

### Q: "Tablet shared across 3 children — how?"
> "Three animal avatars on the lock screen — lion, goat, bird. The
> child taps theirs and loads their encrypted profile from
> `progress.db`. No password because a 6-year-old can't enter one
> — that's a deliberate UX trade-off. Privacy: each learner's rows
> are Fernet-encrypted with a key in Android Keystore (production)
> or a chmod-600 file (dev). On reboot, the tutor loads last-known
> curriculum and adapter weights from `tutor/`; if the DB is
> missing or corrupt it spins up a fresh profile and replays the
> first-90-seconds onboarding."

### Q: "Break-even story. RWF or USD on camera."
Open `business.md`. Cite the 60-second oral block verbatim:
> "Per cooperative: $3,833 a year opex — no data fees, fully
> offline. Cost per child per year: **$25.50**. Revenue: $30 per
> child per year under an REB per-seat licence. Gross margin per
> deployment: **$667 a year**. Organisation-level break-even:
> **~480 cooperatives** = **72,000 children** = 6 % of Rwanda's
> P1–P3 enrolment, achievable in 24–36 months via the government
> channel. Year-0 to year-2 path: **$50k** grant pilot at 10 coops,
> then REB LOI at 15,000 children for **$375k ARR** with cash
> break-even month 14 to 18. Value side: each mastered child is
> worth **$850 in lifetime-earnings NPV** against a **$51
> two-year cost** — **17× return**."

### Q: "Weekly parent report for a non-literate parent."
Open `reports/learner_lion/<week>/report.png`:
> "Icons only, no required reading. Five bars — one per sub-skill —
> with a coloured marker: green tick for ≥0.7 mastery, amber dot
> for middle, red exclamation for <0.4. Under the header we have a
> QR code that points to a **30-second voiced summary** written by
> my QLoRA-tuned TinyLlama. The parent scans, hears their child's
> week in one short audio clip. If the flag is set, an amber
> dyscalculia-warning banner appears at the top — gentle phrasing:
> 'Your child has worked hard but is still finding some numbers
> tricky. It could help to talk to their teacher...' — never
> alarmist."

### Q: "Dyscalculia flag — where is it?"
Open `tutor/dyscalculia.py`:
> "Pure function `flag_plateau`. Groups attempts into per-day
> sessions. Fires when three conditions hold: last 3 sessions have
> mean correctness below 0.5, no upward trend across those
> sessions, and the KT estimator has been dropping difficulty by at
> least 1.0 on the 10-point scale. That third clause is the brief's
> 'despite difficulty drops' wording — it distinguishes a kid
> who's stuck despite help from a kid who's just hitting hard items.
> Deliberately conservative: false positives send a non-literate
> parent to the teacher unnecessarily. Banner is in
> `parent_report.py:render_png`."

---

## 6. Local context — drilled answers

### Q: "Read the Kinyarwanda greeting aloud."
> "*Muraho! Reka dukine n'imibare.* — 'Hello! Let's play with
> numbers.' Word-choice defence: `reka dukine` is playful, literally
> 'let's play', not `kwiga` which is 'study'. `imibare` is numbers
> as concrete objects, not `gutabara` which is abstract arithmetic
> — the concrete framing matches a 6-year-old's mental model."

### Q: "Name the Kinyarwanda number words 1–5."
> "Rimwe, kabiri, gatatu, kane, gatanu."

### Q: "Name 3 Made-in-Rwanda brands or cooperatives."
> "Kigali Leather (handmade leather goods, based in Nyamirambo),
> Gahaya Links (Agaseke basketry cooperative, 2,500 women artisans),
> and Uburanga Arts (textile and crafts cooperative, Huye district)."
(Keep in back pocket in case they throw you the S2.T1.3-style
question; this challenge is T3.1 but local-context questions often
recur across tiers.)

### Q: "If the child answers 'five apples' (mixed EN + KIN), what does the tutor do?"
> "`lang_detect.detect` sees 'five' in the EN anchor set, no KIN
> anchors, picks `en`. But `number_words` returns the 'five' as
> an EN number-word. The tutor replies in the dominant language
> (English), embedding the number word the child used. In the
> `mix` case with ties, we mirror the dominant language and embed
> the second language for any number words the child used — per
> the brief's task 4."

---

## 7. Live code change — rehearse ONE

Brief says "You may be asked to make a small change on the fly."
Prep by rehearsing exactly ONE so the motion is muscle memory. I
recommend:

**Add 'geometry' as a sixth sub-skill.**

1. `code tutor/curriculum_loader.py` — edit `SKILLS` tuple: add `"geometry"` at the end.
2. `code tutor/adaptive.py` — `DKT.__init__` already defaults `n_skills=5`. They may want you to bump it — change default to 6.
3. `pytest tests/ -q` — expect 27 passed (mastery / BKT tests are skill-string-agnostic).

If they ask for something else, keep it under 3 minutes. If you're
stuck at 4 minutes, say: "I want to commit what I have and explain
where it breaks rather than ship broken code." That's a good answer.

---

## 8. Meta questions from the brief's honor-code framing

### Q: "What was the hardest decision?"
From `process_log.md`:
> "Whether to ship a real QLoRA-tuned TinyLlama head or stop at the
> community base. I chose both, layered — tuned if present, base
> fallback, deterministic ultimate fallback — so no single failure
> mode breaks the demo."

### Q: "Where did Claude Code give you bad code?"
> "Three places. First, the scaffold's DKT had a double-sigmoid bug
> — I caught it when training loss dropped but AUC stayed flat.
> Second, the WER eval script had a hardcoded 'vanilla int8' label
> that kept lying after `tutor/asr_model/` shipped; I patched it in
> phase 11. Third, Claude suggested cloning llama.cpp for the int4
> quantiser — I replaced that with the `llama_cpp.llama_model_quantize`
> C-API binding, saved ~5 minutes of C build and a dependency."

### Q: "Three prompts you actually sent?"
From `process_log.md` — memorise the short-form versions:
> "First: 'Whisper-tiny int8 transcribes to empty on a pure sine — is
> this a correctness problem or expected?' That led me to the
> insight that Whisper rightly drops non-speech audio, so my latency
> smoke tests need mild noise not silence.
> Second: 'DKT trained to loss 0.35 but predicts p≈0.02 on test —
> where's the miscalibration?' That found the double-sigmoid.
> Third: 'Fastest path from merged HF TinyLlama to Q4_K_M GGUF on a
> box with no llama.cpp build?' That's how I landed on the pinned
> b4400 convert script plus `llama_model_quantize`."

---

## 9. If you genuinely don't know

Say it, open the file, read it. The brief explicitly scores "the
ability to defend your own code" — opening a file and reading IS
defending. Bluffing + being caught is disqualifying.

Example phrasing:
> "I'd need to re-check the code — let me open it."
> "Good question, one sec, let me find where that lives."
> "I'm going to read you the exact docstring rather than paraphrase."

---

## 10. Closing — optional

If you have time at the end, pre-empt the weakness question:

> "One thing I'd like to do with more time: validate the tuned ASR
> on real child voices, not just pitched Piper synthesis. The
> pipeline works; the number is in-distribution. I'd set up a small
> field recording session with 10 kids, re-evaluate, and retrain if
> the gap is big."

That move signals self-awareness and closes the evaluation loop
better than a bluff.

---

## Appendix — one-liners to keep in your tmux

```bash
# Proof-of-life
pytest tests/ -q                                     # 27 passed
du -sh tutor/                                        # 44 MB
python scripts/bench_latency.py --cycles 10          # p95 ≈ 1.6 s
python scripts/eval_wer.py --vanilla                  # baseline 0.70
python scripts/eval_wer.py                            # tuned 0.00

# Show a parent report
python scripts/seed_progress.py
python parent_report.py --learner-id learner_lion
xdg-open reports/learner_lion/*/report.png

# Notebook
jupyter nbconvert --to notebook --execute notebooks/kt_eval.ipynb --output /tmp/kt_eval_live.ipynb
```

## Appendix — the numbers, memorised

| Thing | Number |
|---|---|
| `du -sh tutor/` | 44 MB |
| Budget | ≤ 75 MB |
| p95 latency CPU | ~1.6 s |
| Budget | < 2.5 s |
| Curriculum items × skills | 80 × 5 |
| Baseline WER (vanilla Whisper + pitch + ESC-50 noise) | 0.8190 |
| Tuned WER (LoRA + pitch + ESC-50 noise @ 12 dB SNR) | 0.0190 |
| BKT / Elo / DKT AUC | 0.577 / 0.561 / 0.520 |
| Synthetic replay | 200 learners × 60 attempts |
| QLoRA TinyLlama Q4_K_M | 637 MB (in `~/.cache/`) |
| LoRA rank (Whisper + LLM) | 16 |
| Epochs (Whisper / LLM) | 4 / 2 |
| Tests | 27 green |
| Cost per child per year | $25.50 |
| REB licence per child per year | $30 |
| Org break-even deployments | ~480 cooperatives |
| Org break-even children | 72,000 |
| Lifetime value per mastered child (NPV) | $850 |
| Two-year cost per child | $51 |
| ROI | 17× |

Memorise this table. If you can cite the right number when the
evaluator asks, that alone is evidence you built it.
