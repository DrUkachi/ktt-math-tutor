"""QLoRA fine-tune TinyLlama-1.1B on a synthetic numeracy instruction set.

Pipeline
--------
1. Synthesise ~200 short (instruction, response) pairs covering the
   five curriculum sub-skills: counting, number sense, addition,
   subtraction, word problems. Each response is one short, positive,
   child-safe sentence in English (the base model already handles
   instruction-following in English; multilingual output is left to
   the system prompt at inference).
2. 4-bit quantise the base model with bitsandbytes (nf4) and wrap with
   LoRA adapters on q_proj / k_proj / v_proj / o_proj.
3. Train 2 epochs on GPU (~5 min on an L4).
4. Save the adapter to ``outputs/tinyllama_lora/`` AND the merged
   (non-quantised) weights to ``outputs/tinyllama_merged/``.
5. GGUF conversion and Q4_K_M quantisation of the merged model is
   documented in footprint_report.md but NOT run here, because it
   requires the llama.cpp C++ build. Runtime LLM inference uses the
   community pre-quantised base GGUF; the adapter is trained for
   defensive evidence of the QLoRA pipeline.

Run:
    python scripts/train_llm_qlora.py --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ---------- synthetic instruction set ----------

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

ENCOURAGE_CORRECT = [
    "Great job! You counted {n} correctly.",
    "Yes, the answer is {n}. Wonderful!",
    "Exactly {n} — well done!",
    "Perfect, it's {n}. Keep going!",
    "You got it — {n} is right.",
]
ENCOURAGE_INCORRECT = [
    "Not quite — let's try that one more time.",
    "Good try! The answer is {n}. Let's keep going.",
    "Close! Remember, the answer is {n}.",
    "Let's look again — it's {n}.",
    "Take another look — the answer is {n}.",
]
OBJECTS = ["apples", "goats", "cows", "mangoes", "beads", "drums", "books"]
SKILLS = ("counting", "number_sense", "addition", "subtraction", "word_problem")


def _rand_example(rng: random.Random, correct: bool) -> dict:
    n = rng.randint(1, 10)
    skill = rng.choice(SKILLS)
    if correct:
        user = f"The child just answered correctly on a {skill.replace('_',' ')} question about the number {n}."
        response = rng.choice(ENCOURAGE_CORRECT).format(n=n)
    else:
        user = f"The child just answered incorrectly. The correct answer was {n}. Skill: {skill.replace('_',' ')}."
        response = rng.choice(ENCOURAGE_INCORRECT).format(n=n)
    return {"user": user, "assistant": response, "correct": correct}


def build_instructions(n: int, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(_rand_example(rng, correct=rng.random() < 0.6))
    return out


# ---------- training ----------

SYSTEM_PROMPT = (
    "You are a warm, encouraging math tutor for a 6-year-old. "
    "Reply in ONE short sentence, under 12 words. Be positive. "
    "Never criticise. No emoji."
)


def _format(examples: list[dict], tokenizer) -> Dataset:
    def fmt(ex):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    rows = [fmt(ex) for ex in examples]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adapter-out", default="outputs/tinyllama_lora/")
    parser.add_argument("--merged-out", default="outputs/tinyllama_merged/")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU required for QLoRA. Switch to a GPU studio.")

    examples = build_instructions(args.n_examples)
    Path(args.adapter_out).mkdir(parents=True, exist_ok=True)
    (Path(args.adapter_out) / "train_instructions.json").write_text(
        json.dumps(examples, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="cuda",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = _format(examples, tokenizer)

    def _tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=256,
                        padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.adapter_out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    print(f"Adapter saved to {args.adapter_out}")

    # Merge: reload in fp16 (not 4-bit) for merge_and_unload to work
    print("Reloading base in fp16 to merge adapter...")
    from peft import PeftModel
    base_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
    )
    merged = PeftModel.from_pretrained(base_fp16, args.adapter_out).merge_and_unload()
    Path(args.merged_out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged_out)
    tokenizer.save_pretrained(args.merged_out)
    print(f"Merged fp16 model saved to {args.merged_out}")

    # --- GGUF conversion + Q4_K_M quantisation (CPU-native) ---
    import ctypes
    import subprocess
    import urllib.request
    import llama_cpp

    fp16_gguf = Path.home() / ".cache" / "llm" / "tinyllama-numeracy-fp16.gguf"
    q4_gguf = Path.home() / ".cache" / "llm" / "tinyllama-numeracy-Q4_K_M.gguf"
    fp16_gguf.parent.mkdir(parents=True, exist_ok=True)

    # Fetch a pinned convert script — keeps us working even when
    # llama.cpp HEAD adds architectures our gguf-py doesn't know.
    convert_py = Path("/tmp/convert_hf_to_gguf.py")
    if not convert_py.exists():
        print("Downloading convert_hf_to_gguf.py (b4400 pinned)...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ggml-org/llama.cpp/b4400/convert_hf_to_gguf.py",
            convert_py,
        )

    print(f"Converting merged model -> {fp16_gguf}")
    subprocess.run([
        sys.executable, str(convert_py), args.merged_out,
        "--outfile", str(fp16_gguf), "--outtype", "f16",
    ], check=True)

    print(f"Quantising -> {q4_gguf} (Q4_K_M)")
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    params.nthread = 4
    rc = llama_cpp.llama_model_quantize(
        str(fp16_gguf).encode(), str(q4_gguf).encode(), ctypes.byref(params),
    )
    if rc != 0:
        raise SystemExit(f"llama_model_quantize failed with rc={rc}")
    size_mb = q4_gguf.stat().st_size // (1024 * 1024)
    print(f"Q4_K_M GGUF ready: {q4_gguf}  ({size_mb} MB)")


if __name__ == "__main__":
    main()
