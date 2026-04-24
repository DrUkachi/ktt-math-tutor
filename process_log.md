# Process Log — S2.T3.1 AI Math Tutor

Hour-by-hour timeline, LLM/tool declaration, and the three-prompt
record required by the brief.

## Tooling declaration

| Tool                          | Why used                                                                             |
|-------------------------------|--------------------------------------------------------------------------------------|
| Claude Code (Opus 4.7)        | Repo scaffold, phase plan, stub authoring, iterative debugging, doc writing         |
| Piper TTS (en_US-lessac-medium) | Synthesising adult English utterances to pitch-shift into child-voice training data  |
| faster-whisper / CTranslate2  | CPU int8 Whisper inference at runtime                                                |
| Hugging Face `transformers` + `peft` | LoRA fine-tune Whisper-tiny and QLoRA fine-tune TinyLlama on GPU              |
| bitsandbytes                  | NF4 4-bit quantisation for QLoRA base model loading                                  |
| `llama-cpp-python`            | CPU Q4_K_M GGUF inference + the `llama_model_quantize` C-API for int4 quantisation   |
| pinned `convert_hf_to_gguf.py` (llama.cpp b4400) | HF → GGUF conversion, keeps the build dependency-free                  |
| `jiwer`                       | WER metric                                                                           |
| `scikit-learn`                | ROC-AUC for KT evaluation                                                            |

## Hour-by-hour timeline

### Hour 0 — Scaffold (pre-timer)
Read the brief; wrote a 10-phase plan into `PLAN.md`; scaffolded the
`tutor/` package with working stubs (curriculum loader, BKT/DKT/Elo
skeletons, Fernet-encrypted SQLite store, ε-DP aggregation, Gradio
demo wiring, parent-report skeleton, smoke tests). Initial commit
pushed to GitHub.

### Hour 1 — Data, assets, ASR plumbing (Phases 1 → 3)
- `generate_curriculum.py` expanded the 12-item seed to 80 items.
- `scripts/render_scenes.py` composited 70 PNGs (non-touching coloured
  blobs for counting-skill items so the filename encodes ground truth).
- `scripts/render_tts.py` wrote 240 placeholder WAVs; graceful upgrade
  if Piper voices land later.
- `scripts/quantise_whisper.py` rewritten to fetch CT2 int8 weights
  into `~/.cache/` (outside `tutor/`).
- `tutor/asr_adapt.py` rewired onto `faster-whisper`; lazy model load
  keeps `import tutor` cheap.
- `tests/test_visual_count.py` parametrised BlobCounter over every
  counting asset: 17/17 exact matches.
- **Commit + push**: phases 1–3 in a single batch.

### Hour 2 — Child-voice ASR + GPU switch (Phase 4)
- Switched to an L4 GPU studio at user's offer.
- `scripts/build_child_corpus.py`: Piper synth of 60 phrasings × 3
  pitch shifts → 144 train + 36 eval WAVs; split by utterance so eval
  phrasings are unseen to the fine-tune.
- `scripts/eval_wer.py`: jiwer on the eval split.
- Baseline vanilla whisper-tiny int8 on pitched voices: WER **0.7238**.
- `scripts/train_whisper_lora.py`: LoRA r=16 on q_proj/v_proj, 4 epochs
  on L4. Merged, exported, CT2 int8 → `tutor/asr_model/`.
- Post-LoRA WER on the same eval split: **0.0000** (in-distribution).
  Caveat called out honestly in README and in this log: the tuned
  model validates the pipeline; unseen speakers would need separate
  validation.
- **Commit + push**: phase 4.

### Hour 3 — Knowledge tracing + latency + QLoRA head (Phases 5 → 7)
- `tutor/adaptive.py` DKT rewritten with a real backprop trainer
  (next-response BCE loss) and save/load. Fixed a double-sigmoid bug
  that was squashing predictions.
- `scripts/kt_simulate.py`: 200 learners × 60 attempts synthetic
  ground-truth (Beta(1,4) priors, slip=0.1, guess=0.2). Trains DKT
  on train split, rolls through held-out learners.
- **AUC**: BKT 0.5766 · Elo 0.5614 · DKT 0.5200. Written up in the
  notebook with honest interpretation — BKT is Bayes-optimal on the
  synthetic generator, DKT needs more data.
- `scripts/bench_latency.py` with `CUDA_VISIBLE_DEVICES=''` to force
  CPU kernels even on GPU boxes. **p95 1003 ms, max 1007 ms** vs 2500
  ms budget.
- Phase 7: downloaded community TinyLlama Q4_K_M base GGUF; wrote
  `tutor/llm_head.py` with graceful-absence behaviour. Then ran
  `scripts/train_llm_qlora.py`: synth 200-instruction numeracy
  dataset, QLoRA on TinyLlama-1.1B with NF4 base, 2 epochs. Merged
  to FP16, converted to GGUF via pinned b4400 convert script, and
  quantised to Q4_K_M via `llama_cpp.llama_model_quantize` in
  Python — no llama.cpp C++ build needed. Final tuned artefact is
  637 MB, lives in `~/.cache/llm/` (not `tutor/`).

### Hour 4 — Parent report, footprint, defense prep (Phases 8 → 10)
- `scripts/seed_progress.py`: a week of attempts across three
  avatar-tagged learners.
- `parent_report.py` now pulls a summary via `LLMHead.weekly_summary`,
  truncates to 2 sentences (TinyLlama Q4 drifts into meta-commentary
  after a coherent opener), and renders the WAV via Piper.
- `footprint_report.md` filled with live numbers:
  **`du -sh tutor/` = 44 MB** (41% under budget).
- This `process_log.md`; README metrics updated; cold-install dry-run.

## Three sample prompts I actually sent

1. **"Whisper-tiny int8 transcribes to empty on a pure sine — is this
   a correctness problem or expected?"** Produced the insight that
   Whisper's voice activity logic drops non-speech audio; use mild
   Gaussian noise for the latency smoke test, and rely on real WER
   on the augmented corpus for correctness.

2. **"DKT trained to loss 0.35 but predicts p≈0.02 on test — where's
   the miscalibration?"** Traced to two issues: (a) a double-sigmoid
   bug in `mastery()` that the scaffold shipped with; (b) the DKT
   needing more learners to match BKT on a BKT-shaped generator.
   Fix landed in the backprop trainer; honesty about the finding
   landed in the notebook.

3. **"Fastest path from merged HF TinyLlama to Q4_K_M GGUF on a box
   with no llama.cpp C build?"** Combined a pinned `convert_hf_to_gguf.py`
   for the HF → GGUF step with `llama_cpp.llama_model_quantize` for
   the int4 step. Saved ~5 min of C build and a fragile dependency.

## One prompt I discarded

"Clone and build llama.cpp from source to get `llama-quantize`." I
started this and killed it after realising the `llama-cpp-python` C
bindings already expose `llama_model_quantize` — a five-line Python
call replaces a 3–5 min build and an extra runtime dependency.

## Hardest decision

Whether to ship a real QLoRA-tuned TinyLlama head or stop at the
community-pretrained base. A real fine-tune risks eating into the
Phase 8/9/10 budget if GGUF conversion fights us; skipping it loses
the most visible brief requirement for Task 3. I chose to do both:
ship the tuned artefact, but make it strictly optional at runtime via
the resolution-order fallback in `LLMHead`. If the fine-tune had
failed, the community base was already downloaded as the primary
path, and the deterministic fallback inside `parent_report` would
still produce a coherent weekly narrative — no single failure mode
could bring down the demo. The decision cost 25 min of GPU time but
recovered the full Task-3 score.
