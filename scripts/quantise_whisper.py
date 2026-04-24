"""Download whisper-tiny, export to ONNX, and dynamic-quantise to int8.

Run once during setup. The output ``tutor/whisper_int8.onnx`` is what the
runtime ASR loads. This script is intentionally a placeholder until Phase 5.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tutor/whisper_int8.onnx")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Phase-5 deliverable. See PLAN.md for the export + quantise steps."
    )


if __name__ == "__main__":
    main()
