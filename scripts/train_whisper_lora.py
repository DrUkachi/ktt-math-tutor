"""LoRA fine-tune Whisper-tiny on the child-voice train manifest.

Pipeline:
  1. Load ``openai/whisper-tiny`` + tokenizer + feature extractor.
  2. Apply LoRA adapters to q_proj / v_proj of the decoder attention.
  3. Train a few epochs on the pitch-shifted train manifest (GPU).
  4. Merge adapters back into base weights.
  5. Convert to CTranslate2 int8 and save under ``tutor/asr_model/``
     so the CPU-side ``ChildASR(model_path=...)`` picks it up.

The final artefact is CPU-inferable — the GPU is only used for training.

Run:
    python scripts/train_whisper_lora.py  # uses manifest_train.csv by default
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

MODEL_ID = "openai/whisper-tiny"


def _load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_wav(p: str) -> np.ndarray:
    wav, sr = sf.read(p, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _make_dataset(
    manifest_rows: list[dict],
    feat_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> Dataset:
    def gen():
        for row in manifest_rows:
            wav = _load_wav(row["audio_path"])
            feats = feat_extractor(wav, sampling_rate=16000).input_features[0]
            labels = tokenizer(row["transcript_en"]).input_ids
            yield {"input_features": feats, "labels": labels}
    return Dataset.from_generator(gen)


class _Collator:
    def __init__(self, tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = torch.tensor([f["input_features"] for f in features])
        label_lists = [f["labels"] for f in features]
        max_len = max(len(x) for x in label_lists)
        pad_id = self.tokenizer.pad_token_id or -100
        padded = [x + [-100] * (max_len - len(x)) for x in label_lists]
        labels = torch.tensor(padded)
        return {"input_features": input_features, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/child_utt/manifest_train.csv")
    parser.add_argument("--adapter-out", default="outputs/whisper_lora/")
    parser.add_argument("--merged-out", default="outputs/whisper_merged/")
    parser.add_argument("--ct2-out", default="tutor/asr_model/")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU required for this script. Switch to a GPU studio.")
    device = "cuda"

    feat_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")

    rows = _load_manifest(Path(args.train_manifest))
    print(f"Train manifest: {len(rows)} clips")
    train_ds = _make_dataset(rows, feat_extractor, tokenizer)

    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    base = base.to(device)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.adapter_out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=_Collator(tokenizer),
    )
    trainer.train()

    Path(args.adapter_out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_out)
    processor.save_pretrained(args.adapter_out)
    print(f"LoRA adapter saved to {args.adapter_out}")

    # --- Merge adapter into base weights ---
    print("Merging adapter into base weights...")
    merged_base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    merged = PeftModel.from_pretrained(merged_base, args.adapter_out).merge_and_unload()
    merged_out = Path(args.merged_out)
    merged_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out)
    processor.save_pretrained(merged_out)
    print(f"Merged HF model saved to {merged_out}")

    # --- Convert to CTranslate2 int8 for CPU inference ---
    ct2_out = Path(args.ct2_out)
    if ct2_out.exists():
        shutil.rmtree(ct2_out)
    print(f"Converting to CTranslate2 int8 at {ct2_out}...")
    subprocess.run(
        ["ct2-transformers-converter",
         "--model", str(merged_out),
         "--output_dir", str(ct2_out),
         "--quantization", "int8",
         "--copy_files", "tokenizer.json", "tokenizer_config.json",
                         "preprocessor_config.json", "generation_config.json",
                         "special_tokens_map.json", "vocabulary.json"],
        check=True,
    )
    print(f"CT2 model ready at {ct2_out}")


if __name__ == "__main__":
    main()
