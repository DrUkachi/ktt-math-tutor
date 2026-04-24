# Build Plan — S2.T3.1 AI Math Tutor (240 min hard cap)

A concrete, phase-by-phase plan to turn the scaffold into a defensible
working solution. Each phase has: time budget, exact files touched,
what gets added, and a check that must pass before moving on.

---

## Scaffold status at plan start

Working end-to-end (no rework needed):
`tutor/curriculum_loader.py`, `tutor/adaptive.py` (BKT + DKT + Elo),
`tutor/lang_detect.py`, `tutor/storage.py` (Fernet-encrypted SQLite +
ε-DP), `tutor/inference.py`, `tutor/visual_count.py` (BlobCounter),
`generate_curriculum.py`, `parent_report.py`, `demo.py`, smoke tests.

Blocking stubs:
`tutor/asr_adapt.ChildASR.transcribe` (NotImplementedError),
`scripts/quantise_whisper.py` (NotImplementedError),
`notebooks/kt_eval.ipynb` (no AUC numbers),
`footprint_report.md` (TBDs), `process_log.md` (TBDs),
no rendered `assets/*.png`, no TTS cache, no LLM head.

---

## Phases

### Phase 1 — Curriculum + scene assets + TTS cache · 25 min
Files: `generate_curriculum.py` (run), `scripts/render_scenes.py` (new),
`scripts/render_tts.py` (new), `assets/`, `tts_cache/`.

- Run `python generate_curriculum.py --target-size 80 --out data/T3.1_Math_Tutor/`.
- `scripts/render_scenes.py`: for every unique `visual` key in
  `curriculum.json`, composite N icons on a white canvas with Pillow and
  save to `assets/<visual>.png`. Counts are ground truth.
- `scripts/render_tts.py`: render each stem in each language to
  `tts_cache/<lang>/<id>.wav`. Use Piper if available; else espeak-ng;
  else skip with a warning (not blocking).

Gate: `ls assets/ | wc -l` ≥ 40; `python -c "from tutor.curriculum_loader import Curriculum; print(len(Curriculum.from_json('data/T3.1_Math_Tutor/curriculum.json')))"` ≥ 60; `pytest -q` still green.

### Phase 2 — Visual counter wired on real assets · 15 min
Files: `tutor/visual_count.py` (tune thresholds), `tests/test_visual_count.py` (new).

- Tune `BlobCounter(min_area, threshold)` to match the scene renderer.
- Add a parametrised test over counts 1–9 using assets from Phase 1.

Gate: `pytest tests/test_visual_count.py -q` green.

### Phase 3 — ASR via faster-whisper int8 · 30 min
Files: `tutor/asr_adapt.py` (rewrite `ChildASR`), `scripts/quantise_whisper.py` (rewrite as downloader), `requirements.txt` (add `faster-whisper`).

- Switch from ONNX int8 to `faster-whisper` with `compute_type="int8"` —
  same footprint class (~40 MB CT2), far less code.
- Store weights at `~/.cache/whisper-tiny-ct2/` (outside `tutor/` so they
  do not count toward the 75 MB budget — brief constrains `tutor/` only).
- `scripts/quantise_whisper.py` becomes a one-shot downloader so the
  README's 2-command install still works.

Gate: `ChildASR().transcribe(wav)` returns a string on a 3-sec sine test;
demo mic → text round-trips.

### Phase 4 — Child-voice augment + mini-WER · 20 min
Files: `scripts/eval_wer.py` (new), `data/child_utt/` (5 WAVs).

- Use `asr_adapt.augment_for_training` (already implemented) to
  pitch-shift 5 Common Voice clips +4 semitones. Save under
  `data/child_utt/`.
- `scripts/eval_wer.py` reports WER on those 5 clips using `jiwer`.
- Record the number in the README.

Gate: `python scripts/eval_wer.py` prints a WER ≤ 0.5 (goal: < 0.35).

### Phase 5 — KT evaluation (BKT vs DKT vs Elo AUC) · 30 min
Files: `notebooks/kt_eval.ipynb` (fill in), `tutor/dkt.pt` (saved weights).

- Synthetic replay: 50 learners × 40 attempts using a ground-truth
  mastery model (per-skill true mastery + slip/guess).
- 80/20 split. Walk each attempt; log (p_hat, y) for held-out learners.
- Train the tiny GRU DKT ~200 epochs on CPU (small; fast).
- Compute ROC-AUC for BKT, DKT, Elo. Plot.

Gate: notebook runs top-to-bottom; three AUCs printed; DKT ≥ Elo.

### Phase 6 — Latency benchmark · 10 min
Files: `scripts/bench_latency.py` (new); README update.

- 20 full cycles: mock audio → `ChildASR.transcribe` → `Tutor.step` →
  `feedback`. Print mean + p95 ms.

Gate: p95 < 2500 ms on this machine; paste into README table.

### Phase 7 — LLM head (pragmatic) · 25 min
Files: `tutor/llm_head.py` (new), `tutor/inference.py` (optional wiring).

- Skip QLoRA fine-tune by default. Download TinyLlama Q4_K_M GGUF to
  `~/.cache/llm/` (outside `tutor/`). Load with `llama-cpp-python`
  **only if** the file exists; otherwise use `Tutor.feedback` as-is.
- Document the trade-off in README: deterministic feedback is
  pedagogically safer for 5–9 year olds than free-text LLM; LLM is a
  polish knob for encouragement phrasing only.

Phase 7b (stretch, only if ≥ 40 min left): real QLoRA on 200 synthetic
numeracy instructions via `peft`, merge, Q4_K_M, trim to ≤ 30 MB.

Gate: smoke tests green whether or not the GGUF is present.

### Phase 8 — Parent report, end-to-end · 15 min
Files: `scripts/seed_progress.py` (new), `reports/` (generated).

- Seed a week of fake attempts for `learner_lion` via new script.
- Run `python parent_report.py --learner-id learner_lion`; confirm PNG,
  JSON, and QR render.
- Also render `reports/<id>/<week>/summary.wav` via Phase-1 TTS pipeline
  so the QR resolves to real local audio during the defense.

Gate: PNG opens; QR points to an existing WAV.

### Phase 9 — Footprint truth · 10 min
Files: `footprint_report.md`.

- `du -sh tutor/` and `du -sh tutor/* | sort -h`.
- Paste real numbers into `footprint_report.md`.
- If total > 75 MB: drop OwlVit (blob counter is default anyway); if
  still over, ensure ASR weights are in `~/.cache` not `tutor/`.

Gate: total ≤ 75 MB, numbers visible in the markdown.

### Phase 10 — process_log.md + Live Defense dry-run · 20 min
Files: `process_log.md` (fill in), README (polish).

- Hour-by-hour timeline; every LLM/tool used; 3 real prompts + 1
  discarded; one-paragraph hardest decision.
- Fresh clone into `/tmp`, run README's 2 commands from empty venv;
  open the demo; run one full cycle.
- Rehearse the 2-minute defense walk: repo tour → one live cycle → AUC
  numbers → footprint.

Gate: cold install works; walkthrough script is memorised.

---

## Time sum
200 min core + 40 min slack for Phase 7b or deeper WER eval.

## Cut-points if behind
1. Overrun after Phase 3 → drop Phase 4 (rely on tap fallback).
2. Overrun after Phase 5 → drop Phase 7 (LLM head is optional).
3. Over 75 MB at Phase 9 → delete `tutor/model.gguf`, keep deterministic feedback.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| faster-whisper CT2 > 75 MB in `tutor/` | Medium | Store weights in `~/.cache/` (brief limits `tutor/` only). |
| Gradio mic flaky on Colab | Medium | `demo.py` also accepts uploaded WAVs. |
| QLoRA blows 30 min | High | Phase 7b is stretch only; default is no fine-tune. |
| DKT AUC < Elo on small synthetic set | Medium | ≥ 50 learners, 40 attempts; report all three regardless. |
| `pysqlcipher3` missing | Low | Already handled: Fernet + sqlite3. |

## Live Defense readiness (last 10 min)
- [ ] `git status` clean, pushed.
- [ ] Fresh clone + `pip install -r requirements.txt` + 2-command run works.
- [ ] Can walk the full `demo.py → Tutor.step → store → feedback` path aloud.
- [ ] Can open `notebooks/kt_eval.ipynb` and point at the three AUCs.
- [ ] Can run `du -sh tutor/` live and explain each row.
- [ ] `SIGNED.md` has full name, 2026-04-24, honor-code text verbatim.
