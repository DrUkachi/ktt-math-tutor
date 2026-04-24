"""End-to-end latency benchmark for the stimulus -> response -> feedback cycle.

The brief's per-cycle budget is < 2.5 s on Colab CPU. This script runs
N cycles in tight loop and reports mean / median / p95 / p99 in ms.

ASR is forced to CPU kernels (ChildASR hardcodes device="cpu"), so the
numbers are honest even when run on a GPU-enabled box during development.
A final CPU-only run is still recommended for the Live Defense number.

Run:
    python scripts/bench_latency.py --cycles 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Force CPU-only torch/CT2 even on GPU boxes so the number represents
# the shipping-runtime environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")

from tutor.asr_adapt import ChildASR
from tutor.inference import Tutor


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * p))
    return xs[max(0, min(k, len(xs) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--curriculum",
                        default="data/T3.1_Math_Tutor/curriculum.json")
    parser.add_argument("--learner-id", default="bench")
    parser.add_argument("--out-json", default="metrics/latency.json")
    args = parser.parse_args()

    # 1 s of mild noise so faster-whisper does real work.
    rng = np.random.default_rng(1)
    wav = (0.02 * rng.standard_normal(16000)).astype(np.float32)

    asr = ChildASR()
    _ = asr.transcribe(wav)  # warm the model; exclude cold-start from numbers

    tutor = Tutor(learner_id=args.learner_id, curriculum_path=args.curriculum)

    asr_ms: list[float] = []
    tutor_ms: list[float] = []
    total_ms: list[float] = []

    for _ in range(args.cycles):
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        response = asr.transcribe(wav)
        t1 = time.perf_counter()
        _cycle = tutor.step(age_band="6-7", response_text=response or "5")
        t2 = time.perf_counter()
        asr_ms.append((t1 - t0) * 1000)
        tutor_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t_all) * 1000)

    def _stats(xs: list[float]) -> dict:
        return {
            "mean_ms": round(mean(xs), 1),
            "median_ms": round(median(xs), 1),
            "p95_ms": round(_percentile(xs, 0.95), 1),
            "p99_ms": round(_percentile(xs, 0.99), 1),
            "max_ms": round(max(xs), 1),
        }

    report = {
        "cycles": args.cycles,
        "budget_ms": 2500,
        "pass": max(total_ms) < 2500,
        "asr": _stats(asr_ms),
        "tutor_step": _stats(tutor_ms),
        "total": _stats(total_ms),
    }

    print(f"Cycles: {args.cycles}   Budget: 2500 ms   Max: {report['total']['max_ms']} ms "
          f"({'PASS' if report['pass'] else 'FAIL'})")
    for k in ("asr", "tutor_step", "total"):
        s = report[k]
        print(f"  {k:12s} mean={s['mean_ms']:>6.1f}  p95={s['p95_ms']:>6.1f}  "
              f"p99={s['p99_ms']:>6.1f}  max={s['max_ms']:>6.1f} ms")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
