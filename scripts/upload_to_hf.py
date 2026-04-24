"""Upload all three artefacts to Hugging Face Hub for the submission form.

Creates / updates two repos under your HF account:

    <user>/ktt-math-tutor-models    (type=model)
      - whisper-tiny-child-lora-ct2int8/   (44 MB, tutor/asr_model/)
      - tinyllama-numeracy-qlora-adapter/  (21 MB, outputs/tinyllama_lora/)
      - tinyllama-numeracy-Q4_K_M.gguf     (637 MB, ~/.cache/llm/...)
      - README.md                           (auto-generated model card)

    <user>/ktt-math-tutor-data      (type=dataset)
      - T3.1_Math_Tutor/                    (curriculum + seeds + schema)
      - child_utt/                          (manifests; WAVs are regeneratable)
      - README.md                           (auto-generated dataset card)

Pre-requisite
-------------
1. ``pip install huggingface_hub``
2. One of:
    a. ``huggingface-cli login``   (interactive; recommended)
    b. ``export HF_TOKEN=hf_xxx``  (env var; non-interactive)

Run
---
    # Everything:
    python scripts/upload_to_hf.py

    # Just the model repo:
    python scripts/upload_to_hf.py --skip-dataset

    # Preview what would happen without uploading:
    python scripts/upload_to_hf.py --dry-run

The script is idempotent: re-running updates changed files and leaves
everything else alone.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ASR_MODEL_DIR = REPO_ROOT / "tutor" / "asr_model"
LORA_ADAPTER_DIR = REPO_ROOT / "outputs" / "tinyllama_lora"
LLM_GGUF = Path.home() / ".cache" / "llm" / "tinyllama-numeracy-Q4_K_M.gguf"

CURRICULUM_DIR = REPO_ROOT / "data" / "T3.1_Math_Tutor"
CHILD_MANIFEST_DIR = REPO_ROOT / "data" / "child_utt"

MODEL_REPO_NAME = "ktt-math-tutor-models"
DATASET_REPO_NAME = "ktt-math-tutor-data"

GITHUB_URL = "https://github.com/DrUkachi/ktt-math-tutor"


# --------------------------------------------------------------------------- #
# Model card + dataset card bodies (written to the respective repos)
# --------------------------------------------------------------------------- #

def _model_card(hf_user: str) -> str:
    return f"""---
license: mit
tags:
- whisper
- asr
- tinyllama
- qlora
- llm
- gguf
- edtech
- math-tutor
- kinyarwanda
language:
- en
- fr
- rw
---

# KTT Math Tutor — Models

Companion model artefacts for the AIMS KTT Hackathon Tier-3 submission
**S2.T3.1 AI Math Tutor for Early Learners**. Source code and training
scripts: {GITHUB_URL}.

## What's here

| Subfolder / file | Size | Role |
|---|---|---|
| `whisper-tiny-child-lora-ct2int8/` | 44 MB | child-voice LoRA-tuned Whisper-tiny, merged, CTranslate2 int8 for CPU |
| `tinyllama-numeracy-qlora-adapter/` | 21 MB | QLoRA adapter (r=16, NF4 base) trained on 200 synthetic numeracy instructions |
| `tinyllama-numeracy-Q4_K_M.gguf` | 637 MB | the adapter merged into TinyLlama-1.1B and quantised to Q4_K_M |

## How to use

### ASR (child-voice Whisper)

```python
from faster_whisper import WhisperModel
model = WhisperModel("{hf_user}/{MODEL_REPO_NAME}",
                     device="cpu", compute_type="int8",
                     local_files_only=False)
segments, _ = model.transcribe(wav, language="en", beam_size=1)
```

Or, via the tutor's wrapper (auto-picks `tutor/asr_model/` from the repo):

```bash
git clone {GITHUB_URL}
cd ktt-math-tutor && pip install -r requirements.txt
python demo.py
```

Eval on the in-distribution child-voice corpus (36 clips, pitched +3/+4.5/+6 semitones):

- Baseline vanilla Whisper-tiny int8: **WER 0.7048**
- This LoRA-tuned model:                **WER 0.0000**

See `scripts/eval_wer.py` and `metrics/wer_*.json` in the code repo.

### LLM head (weekly parent summary)

```python
from llama_cpp import Llama
llm = Llama(
    model_path="tinyllama-numeracy-Q4_K_M.gguf",
    n_ctx=512, n_threads=4, verbose=False,
)
r = llm.create_chat_completion(messages=[
    {{"role": "system", "content": "You are a warm math tutor. One short sentence."}},
    {{"role": "user", "content": "The child is strong at addition; needs practice on number sense."}},
])
```

Or via the tutor's wrapper (`tutor/llm_head.py`): the model is resolved
in order `$TUTOR_LLM_GGUF` → this tuned Q4_K_M → community TinyLlama
base → deterministic fallback. None of the LLM path is in the
inference hot path; it runs once per learner per week for the
voiced parent summary.

## Training recipes

- **ASR LoRA**: `scripts/train_whisper_lora.py` — 4 epochs on L4 GPU,
  LoRA r=16 on q_proj/v_proj, merge, export to CT2 int8.
- **LLM QLoRA**: `scripts/train_llm_qlora.py` — 2 epochs on L4 GPU,
  NF4 4-bit base, LoRA r=16 on q/k/v/o_proj, merge, convert to GGUF
  via pinned llama.cpp b4400 script, quantise to Q4_K_M via the
  `llama_cpp.llama_model_quantize` Python binding.

## License

MIT. Attribution welcomed; not required.
"""


def _dataset_card(hf_user: str) -> str:
    return f"""---
license: mit
language:
- en
- fr
- rw
task_categories:
- question-answering
- automatic-speech-recognition
tags:
- edtech
- numeracy
- math
- children
- low-resource
---

# KTT Math Tutor — Data

Data artefacts for the AIMS KTT Hackathon Tier-3 submission
**S2.T3.1 AI Math Tutor for Early Learners**. Source code:
{GITHUB_URL}.

## Contents

### `T3.1_Math_Tutor/`
Core curriculum + seeds.

- `curriculum.json` — **80 items** × 5 sub-skills (counting, number
  sense, addition, subtraction, word problem) with EN / FR / KIN
  stems, difficulty 1–10, age bands 5–6 / 6–7 / 7–8 / 8–9, visual
  asset keys, expected integer answer.
- `curriculum_seed.json` — the 12 hand-authored seed items the
  generator expands from.
- `diagnostic_probes_seed.csv` — 5 quick diagnostic probe items.
- `child_utt_sample_seed.csv` + `child_utt_index.md` — utterance
  manifest schema and sources.
- `parent_report_schema.json` — schema for the weekly parent report.

### `child_utt/`
Manifests for the synthetic child-voice ASR corpus:

- `manifest_train.csv` — 144 clips (60 utterances × 3 pitch shifts),
  used to LoRA-fine-tune Whisper-tiny.
- `manifest_eval.csv` — 36 clips (12 held-out utterances × 3 pitch
  shifts). Utterances are disjoint from train.

The audio WAVs themselves are **not** bundled — they are
deterministically reproducible by running
`scripts/build_child_corpus.py` from the code repo. This keeps the
dataset small and avoids redistributing third-party voice data.

## Reproduction

```bash
git clone {GITHUB_URL}
cd ktt-math-tutor
pip install -r requirements.txt
# Optional: a Piper voice for the source audio
mkdir -p ~/.local/share/piper-voices && curl -L -o \\
  ~/.local/share/piper-voices/en_US-lessac-medium.onnx \\
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

# Generate curriculum (idempotent; fixed seed)
python generate_curriculum.py

# Generate the child-voice corpus (deterministic)
python scripts/build_child_corpus.py
```

## License

MIT.
"""


# --------------------------------------------------------------------------- #
# Upload logic
# --------------------------------------------------------------------------- #

def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    try:
        me = api.whoami()
    except Exception as e:
        raise SystemExit(
            "Not logged in to Hugging Face. Run `huggingface-cli login` "
            f"or `export HF_TOKEN=...`. (whoami error: {e})"
        )
    return api, me["name"]


def _ensure_repo(api, repo_id: str, repo_type: str, private: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would create {repo_type} repo {repo_id}")
        return
    api.create_repo(
        repo_id=repo_id, repo_type=repo_type,
        private=private, exist_ok=True,
    )


def _upload_file(api, *, repo_id: str, repo_type: str, local: Path,
                 path_in_repo: str, dry_run: bool) -> None:
    if dry_run:
        sz = local.stat().st_size // (1024 * 1024) if local.exists() else "MISSING"
        print(f"  [dry-run] {local}  ->  {repo_id}:{path_in_repo}  ({sz} MB)")
        return
    api.upload_file(
        path_or_fileobj=str(local), path_in_repo=path_in_repo,
        repo_id=repo_id, repo_type=repo_type,
    )


def _upload_folder(api, *, repo_id: str, repo_type: str, local: Path,
                   path_in_repo: str, dry_run: bool) -> None:
    if dry_run:
        n = sum(1 for _ in local.rglob("*") if _.is_file()) if local.exists() else 0
        print(f"  [dry-run] {local}/  ->  {repo_id}:{path_in_repo}/  ({n} files)")
        return
    api.upload_folder(
        folder_path=str(local), path_in_repo=path_in_repo,
        repo_id=repo_id, repo_type=repo_type,
    )


def upload_models(api, hf_user: str, private: bool, dry_run: bool) -> str:
    repo_id = f"{hf_user}/{MODEL_REPO_NAME}"
    print(f"\n=== Model repo: https://huggingface.co/{repo_id} ===")
    _ensure_repo(api, repo_id, "model", private, dry_run)

    # Model card
    card = REPO_ROOT / ".cache_model_card.md"
    card.write_text(_model_card(hf_user))
    _upload_file(api, repo_id=repo_id, repo_type="model",
                 local=card, path_in_repo="README.md", dry_run=dry_run)

    # ASR (tutor/asr_model/) — 44 MB
    if ASR_MODEL_DIR.exists():
        _upload_folder(api, repo_id=repo_id, repo_type="model",
                       local=ASR_MODEL_DIR,
                       path_in_repo="whisper-tiny-child-lora-ct2int8",
                       dry_run=dry_run)
    else:
        print(f"  SKIP: {ASR_MODEL_DIR} missing")

    # LoRA adapter (outputs/tinyllama_lora/) — 21 MB
    if LORA_ADAPTER_DIR.exists():
        _upload_folder(api, repo_id=repo_id, repo_type="model",
                       local=LORA_ADAPTER_DIR,
                       path_in_repo="tinyllama-numeracy-qlora-adapter",
                       dry_run=dry_run)
    else:
        print(f"  SKIP: {LORA_ADAPTER_DIR} missing "
              "(run scripts/train_llm_qlora.py first)")

    # Q4_K_M GGUF — 637 MB
    if LLM_GGUF.exists():
        _upload_file(api, repo_id=repo_id, repo_type="model",
                     local=LLM_GGUF,
                     path_in_repo="tinyllama-numeracy-Q4_K_M.gguf",
                     dry_run=dry_run)
    else:
        print(f"  SKIP: {LLM_GGUF} missing")

    card.unlink(missing_ok=True)
    return f"https://huggingface.co/{repo_id}"


def upload_dataset(api, hf_user: str, private: bool, dry_run: bool) -> str:
    repo_id = f"{hf_user}/{DATASET_REPO_NAME}"
    print(f"\n=== Dataset repo: https://huggingface.co/datasets/{repo_id} ===")
    _ensure_repo(api, repo_id, "dataset", private, dry_run)

    card = REPO_ROOT / ".cache_dataset_card.md"
    card.write_text(_dataset_card(hf_user))
    _upload_file(api, repo_id=repo_id, repo_type="dataset",
                 local=card, path_in_repo="README.md", dry_run=dry_run)

    if CURRICULUM_DIR.exists():
        _upload_folder(api, repo_id=repo_id, repo_type="dataset",
                       local=CURRICULUM_DIR,
                       path_in_repo="T3.1_Math_Tutor",
                       dry_run=dry_run)

    for name in ("manifest_train.csv", "manifest_eval.csv"):
        p = CHILD_MANIFEST_DIR / name
        if p.exists():
            _upload_file(api, repo_id=repo_id, repo_type="dataset",
                         local=p, path_in_repo=f"child_utt/{name}",
                         dry_run=dry_run)

    card.unlink(missing_ok=True)
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--private", action="store_true",
                        help="Create as private (default is public — evaluators need read access)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without uploading.")
    args = parser.parse_args()

    api, hf_user = _api()
    print(f"Logged in as: {hf_user}")

    links: list[tuple[str, str]] = []
    if not args.skip_models:
        link = upload_models(api, hf_user, args.private, args.dry_run)
        links.append(("Model Link", link))
    if not args.skip_dataset:
        link = upload_dataset(api, hf_user, args.private, args.dry_run)
        links.append(("Dataset Link", link))

    print("\n" + "=" * 60)
    print("Paste these into the submission form:")
    print("=" * 60)
    print(f"  Source Code Link:  {GITHUB_URL}")
    for label, url in links:
        print(f"  {label:17s} {url}")
    if args.dry_run:
        print("\n(dry-run only; re-run without --dry-run to actually upload)")


if __name__ == "__main__":
    main()
