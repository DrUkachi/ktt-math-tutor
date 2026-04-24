"""One-shot downloader for the optional LLM head GGUF.

The tutor works fully offline without this file (see
:mod:`tutor.llm_head` and the fallback path in ``parent_report.py``).
Download it only if you want natural-language encouragement phrases
and voiced weekly summaries.

Default target: ``~/.cache/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf``
(~640 MB).

Run:
    python scripts/download_llm.py
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
    "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default=os.environ.get(
                            "TUTOR_LLM_GGUF",
                            str(Path.home() / ".cache" / "llm" /
                                "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 100_000_000:
        print(f"Already present: {out}  ({out.stat().st_size // (1024*1024)} MB)")
        return

    print(f"Downloading {URL}\n      to {out}")
    with urllib.request.urlopen(URL) as r:
        with open(out, "wb") as fh:
            total = 0
            while chunk := r.read(1024 * 1024):
                fh.write(chunk)
                total += len(chunk)
                if total % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"  {total // (1024*1024)} MB")
    print(f"Done: {out}  ({out.stat().st_size // (1024*1024)} MB)")


if __name__ == "__main__":
    main()
