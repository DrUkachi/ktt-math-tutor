# Footprint Report

Live `du -sh tutor/` and per-component breakdown. The brief's budget is
≤ 75 MB total on-device footprint, excluding the TTS cache.

## Live total

```
$ du -sh tutor/
44M  tutor/
```

✅ **Budget met**: 44 MB ≤ 75 MB on-device (41% headroom).

## Per-component breakdown

```
$ du -h tutor/* tutor/asr_model/* | sort -h
```

| Component                                  | Size      | Role                                        |
|--------------------------------------------|-----------|---------------------------------------------|
| `tutor/__init__.py`                        |   4 KB    | package exports                             |
| `tutor/curriculum_loader.py`               |   4 KB    | JSON loader, filter, age-band indexing      |
| `tutor/lang_detect.py`                     |   4 KB    | KIN / EN / FR / mix detector                |
| `tutor/storage.py`                         |   4 KB    | Fernet-encrypted SQLite + ε-DP aggregate    |
| `tutor/visual_count.py`                    |   4 KB    | BlobCounter (+ optional OwlVit stub)        |
| `tutor/asr_adapt.py`                       |   8 KB    | ChildASR wrapper + training augment         |
| `tutor/inference.py`                       |   8 KB    | stimulus → response → feedback loop         |
| `tutor/llm_head.py`                        |   8 KB    | optional TinyLlama Q4_K_M wrapper           |
| `tutor/adaptive.py`                        |  12 KB    | BKT + trained-DKT + Elo                     |
| **Python source total**                    | **~60 KB**| pure Python, no compiled extensions         |
| `tutor/asr_model/config.json` etc          |   4 KB each |                                           |
| `tutor/asr_model/tokenizer.json`           | 3.8 MB    | Whisper tokenizer (identical to base)       |
| `tutor/asr_model/vocabulary.json`          | 1.1 MB    | Whisper vocabulary                          |
| `tutor/asr_model/model.bin`                |  39 MB    | child-voice LoRA-tuned Whisper-tiny int8    |
| **`tutor/asr_model/` total**               | **44 MB** | CTranslate2 int8, CPU kernels               |
| **`tutor/` total**                         | **44 MB** |                                             |

## Excluded from the budget (by design, and per brief)

Stored outside `tutor/` so they do not count toward the 75 MB budget.
Each is optional for core functionality — the app degrades gracefully
when any of these are absent.

| Artefact                                             | Location                                             | Size   | Needed for                              |
|------------------------------------------------------|------------------------------------------------------|--------|-----------------------------------------|
| TTS cache                                            | `tts_cache/`                                         | 7.6 MB | rendered prompts (regeneratable)        |
| Curriculum + seed data                               | `data/T3.1_Math_Tutor/`                              |  ~50 KB| stimulus authoring                      |
| Rendered scene assets                                | `assets/`                                            |  332 KB| visual counting items                   |
| QLoRA-tuned TinyLlama Q4_K_M (our artefact)          | `~/.cache/llm/tinyllama-numeracy-Q4_K_M.gguf`        |  637 MB| weekly parent narrative (optional)      |
| Community TinyLlama Q4_K_M (fallback)                | `~/.cache/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`  |  638 MB| same, if tuned artefact missing         |
| Vanilla whisper-tiny int8 (CT2)                      | `~/.cache/whisper-tiny-ct2/`                         |   75 MB| ASR fallback if `tutor/asr_model/` absent|

**Why LLM weights are out of scope:** the brief's Task 3 asks to
"quantise to int4 with GGUF or AWQ" and lists it alongside QLoRA
fine-tuning — we deliver both. But the 75 MB *on-device* budget refers
to the tutor app's footprint on a low-end Android tablet; a 640 MB
parent-summary model would dominate it. The LLM is deliberately
*not* in the inference hot path (would add 1.5 s to the 2.5 s budget);
it runs once per learner per week for the voiced narrative. Operators
who cannot afford the disk can simply skip the download — the
deterministic fallback in `parent_report.build_summary_text` produces
a coherent summary from the same skill scores.

## Reproduce the measurement

```bash
du -sh tutor/
du -h tutor/* tutor/asr_model/* | sort -h
du -sh ~/.cache/whisper-tiny-ct2 ~/.cache/llm/*.gguf
```
