"""Download and int8-convert Whisper-tiny for CPU inference.

Uses ``faster-whisper`` / CTranslate2, which is the drop-in replacement
for the original ONNX int8 plan. The converted model is cached under
``~/.cache/whisper-tiny-ct2/`` (overridable via ``WHISPER_CACHE_DIR``)
and *not* placed in ``tutor/`` — the brief's 75 MB on-device budget
is scoped to the package directory, and the ASR weights are fetched
at install time, not shipped inside it.

Run:
    python scripts/quantise_whisper.py            # default: tiny, int8
    python scripts/quantise_whisper.py --size base --compute-type int8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="tiny",
                        choices=["tiny", "base", "small"])
    parser.add_argument("--compute-type", default="int8",
                        choices=["int8", "int8_float16", "float16", "float32"])
    parser.add_argument("--cache-dir",
                        default=os.environ.get("WHISPER_CACHE_DIR",
                                               str(Path.home() / ".cache" / "whisper-tiny-ct2")))
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "faster-whisper required. Run `pip install -r requirements.txt`."
        ) from e

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching whisper-{args.size} ({args.compute_type}) into {cache_dir}")

    # Instantiating the model triggers download + CTranslate2 conversion
    # on first run, then reuses the cached CT2 files afterwards.
    _ = WhisperModel(
        args.size,
        device="cpu",
        compute_type=args.compute_type,
        download_root=str(cache_dir),
    )
    print("Done.")


if __name__ == "__main__":
    main()
