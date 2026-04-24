"""Seed ``progress.db`` with a week of synthetic attempts for demo /
defense. Populates three learner profiles (lion / goat / bird) so the
parent report script has something to aggregate.

Run:
    python scripts/seed_progress.py
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.curriculum_loader import SKILLS
from tutor.storage import Attempt, ProgressStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", nargs="+",
                        default=["learner_lion", "learner_goat", "learner_bird"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--attempts-per-day", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    store = ProgressStore()

    now = time.time()
    for lid in args.learners:
        # Each learner has a distinctive strengths profile so the
        # parent report has interesting shape.
        strengths = {s: rng.uniform(0.3, 0.9) for s in SKILLS}
        for d in range(args.days):
            day_ts = now - (args.days - d) * 86400
            for _ in range(args.attempts_per_day):
                skill = rng.choice(SKILLS)
                correct = rng.random() < strengths[skill]
                store.record(Attempt(
                    learner_id=lid,
                    item_id=f"seed_{d}_{skill[:3]}",
                    skill_id=skill,
                    correct=correct,
                    response_ms=rng.randint(400, 2500),
                    ts=day_ts + rng.uniform(0, 3600),
                ))
        print(f"  {lid}: seeded {args.days * args.attempts_per_day} attempts  "
              f"(strengths: " + ", ".join(f"{k[:4]}={v:.2f}" for k, v in strengths.items()) + ")")

    store.close()
    print("Done.")


if __name__ == "__main__":
    main()
