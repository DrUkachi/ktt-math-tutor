# Build Plan — 240 min hard cap

A 10-phase plan for delivering S2.T3.1 inside the 4-hour window. Each phase
has a deliverable that lands in the repo and a check that verifies the
constraint gates (≤ 75 MB, < 2.5 s latency, fully offline).

| #  | Phase                          | Time   | Deliverable                                                       | Gate / check                              |
|----|--------------------------------|--------|-------------------------------------------------------------------|-------------------------------------------|
| 1  | Bootstrap                      | 10 min | Repo scaffold, MIT license, signed honor code, empty process_log  | `pytest -q` runs (zero tests)             |
| 2  | Curriculum data                | 20 min | `generate_curriculum.py` → ≥ 60 items × 5 sub-skills × 3 age bands | `python generate_curriculum.py` exits 0  |
| 3  | Encrypted storage              | 15 min | `tutor/storage.py` (SQLCipher), `progress.db` schema              | round-trip insert/select test             |
| 4  | Inference loop                 | 40 min | `tutor/inference.py` — present → score → speak                    | end-to-end < 2.5 s on Colab CPU           |
| 5  | ASR + lang detect              | 30 min | `tutor/asr_adapt.py` int8 whisper, `lang_detect.py` (KIN/EN/FR/mix) | WER on child seed < 35 %                |
| 6  | Knowledge tracing              | 35 min | BKT + tiny-GRU DKT + Elo baseline; `notebooks/kt_eval.ipynb`      | DKT AUC ≥ Elo AUC on held-out replay      |
| 7  | LLM head (QLoRA + GGUF int4)   | 35 min | TinyLlama numeracy adapter merged + Q4_K_M                        | gguf size ≤ 30 MB; first-token < 1 s      |
| 8  | Visual grounding               | 15 min | `tutor/visual_count.py` (owlvit-tiny OR OpenCV blob)              | counts 1–5 goats correctly on test scenes |
| 9  | Demo + parent report           | 25 min | `demo.py` (Gradio mic), `parent_report.py` 1-pager + QR audio     | child can complete one item end-to-end    |
| 10 | Footprint, polish, defense prep| 15 min | `footprint_report.md`, README updates, dry-run defense walk       | `du -sh tutor/` ≤ 75 MB                   |

## Sequencing notes

- Phases 1–3 block everything; do them in order.
- Phases 5, 6, 7 are independent and can interleave once 4 is in.
- Phase 8 is optional cushion — drop or replace with the OpenCV blob counter
  if 7 (LLM quantisation) eats into time.
- Phase 10 must leave 10 min for a dry-run Live Defense: clone the repo
  fresh, run the 2 commands from the README, walk through one inference
  cycle aloud.

## Risk register (mitigations)

| Risk                                       | Likelihood | Mitigation                                                |
|--------------------------------------------|------------|-----------------------------------------------------------|
| QLoRA fine-tune busts the 35-min budget    | High       | Ship un-tuned TinyLlama with system-prompt scaffolding    |
| Whisper-tiny WER too high on child voices  | Medium     | Tap-only fallback path always available; ASR is enrichment |
| owlvit-tiny pushes total over 75 MB        | Medium     | Fall back to OpenCV `findContours` blob counter (0 MB)    |
| Gradio mic capture flaky on Colab          | Medium     | Add a "load WAV file" path next to the mic                |
| SQLCipher wheel missing on target Python   | Low        | Fallback to `sqlite3` + Fernet-encrypted blobs            |

## Hour-by-hour LLM declaration template

Tracked live in [process_log.md](process_log.md). For each hour record:
which assistants were used, what for, and any prompt that produced a
non-trivial output. Three full sample prompts + one discarded prompt are
required by the brief.
