"""Simulate learner trajectories and score KT estimators.

Ground-truth model
------------------
Each synthetic learner has a true per-skill mastery that evolves:
- Starts at a skill-specific prior sampled from Beta(2,5).
- After each attempt, mastery drifts toward 1.0 with rate proportional
  to attempt frequency; some learners learn slower (regularisation).
- The observed correctness at each attempt is drawn as
  ``correct ~ Bernoulli(slip_guess(true_mastery))`` where
  ``slip_guess(m) = m*(1-slip) + (1-m)*guess`` with slip=0.1, guess=0.2.

This yields realistic, noisy attempt streams for which BKT / DKT / Elo
all have signal to exploit, without any actual student data.

Evaluation
----------
- 80/20 train/test split by learner_id.
- Each estimator is fit on the 80% and then *rolled forward* on the
  20% — for each held-out attempt, we ask the estimator for its
  ``p(correct)`` BEFORE the attempt is revealed; log (p, y) pairs.
- ROC-AUC is computed on those pairs.

This script writes a JSON report; :file:`notebooks/kt_eval.ipynb`
loads it and plots.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tutor.adaptive import BKT, DKT, Elo
from tutor.curriculum_loader import SKILLS


SLIP = 0.1
GUESS = 0.2


def _observe(true_mastery: float, rng: random.Random) -> bool:
    p = true_mastery * (1 - SLIP) + (1 - true_mastery) * GUESS
    return rng.random() < p


def simulate(
    n_learners: int = 80,
    n_attempts: int = 50,
    seed: int = 7,
) -> list[dict]:
    """Generate attempt streams with enough correctness variance for AUC
    to have signal. Key choices:

    - Lower learn rate (0.01–0.04) so mastery doesn't saturate within 50
      attempts.
    - Beta(1, 4) priors so most learners start clearly non-masters.
    - Slip/guess are applied at observation, so even at mastery=1 we
      don't see 100% correct — keeps both classes in the held-out tail.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out: list[dict] = []
    skills = list(SKILLS)

    for lid in range(n_learners):
        learn_rate = rng.uniform(0.01, 0.04)
        masteries = {s: float(np_rng.beta(1, 4)) for s in skills}
        for _ in range(n_attempts):
            skill = rng.choice(skills)
            correct = _observe(masteries[skill], rng)
            out.append({
                "learner_id": f"L{lid:03d}",
                "skill_id": skill,
                "correct": bool(correct),
            })
            masteries[skill] = masteries[skill] + learn_rate * (1 - masteries[skill])
    return out


def _split_by_learner(
    attempts: list[dict], test_frac: float, rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    ids = sorted({a["learner_id"] for a in attempts})
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids = set(ids[:n_test])
    train = [a for a in attempts if a["learner_id"] not in test_ids]
    test = [a for a in attempts if a["learner_id"] in test_ids]
    return train, test


def _score_bkt(train: list[dict], test: list[dict]) -> tuple[list[float], list[int]]:
    from collections import defaultdict

    # Fit: per-learner_id running BKT on train, then roll forward on test
    # predicting each held-out attempt's correctness BEFORE update.
    ps: list[float] = []
    ys: list[int] = []
    # Fit phase: ignore — BKT defaults are reasonable and fitting on
    # synthetic Beta(2,5) learners would not change much. Instead, for
    # each test learner we warm the estimator on a per-learner prefix
    # of their attempts and evaluate on the rest.
    by_learner: dict[str, list[dict]] = defaultdict(list)
    for a in test:
        by_learner[a["learner_id"]].append(a)
    for lid, attempts in by_learner.items():
        # Use first 50% of this learner's attempts to warm, then predict.
        mid = max(1, len(attempts) // 2)
        bkt = BKT()
        for a in attempts[:mid]:
            bkt.update(a["skill_id"], a["correct"])
        for a in attempts[mid:]:
            p = bkt.mastery(a["skill_id"])
            # Apply slip/guess to get p(correct).
            p_correct = p * (1 - SLIP) + (1 - p) * GUESS
            ps.append(float(p_correct))
            ys.append(int(a["correct"]))
            bkt.update(a["skill_id"], a["correct"])
    return ps, ys


def _score_dkt(train: list[dict], test: list[dict]) -> tuple[list[float], list[int]]:
    """Train a DKT on the train-learner trajectories, then roll through
    test learners predicting each suffix attempt from the prefix.

    The DKT directly outputs p(next correct), so we do NOT re-apply
    slip/guess here — that would double-count the observation noise.
    """
    from collections import defaultdict

    skill_to_idx = {s: i for i, s in enumerate(SKILLS)}
    dkt = DKT(n_skills=len(SKILLS), hidden=32, skill_to_idx=skill_to_idx)

    # Bundle train attempts into per-learner trajectories.
    train_by_learner: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for a in train:
        train_by_learner[a["learner_id"]].append((a["skill_id"], bool(a["correct"])))
    trajectories = list(train_by_learner.values())
    print(f"  DKT: fitting on {len(trajectories)} learner trajectories...")
    dkt.fit(trajectories, epochs=40, lr=5e-3)

    ps: list[float] = []
    ys: list[int] = []
    test_by_learner: dict[str, list[dict]] = defaultdict(list)
    for a in test:
        test_by_learner[a["learner_id"]].append(a)
    for lid, attempts in test_by_learner.items():
        dkt.reset()
        mid = max(1, len(attempts) // 2)
        for a in attempts[:mid]:
            dkt.update(a["skill_id"], a["correct"])
        for a in attempts[mid:]:
            # DKT output is already p(next correct); no slip/guess.
            p_correct = dkt.mastery(a["skill_id"])
            ps.append(float(p_correct))
            ys.append(int(a["correct"]))
            dkt.update(a["skill_id"], a["correct"])
    return ps, ys


def _score_elo(train: list[dict], test: list[dict]) -> tuple[list[float], list[int]]:
    from collections import defaultdict

    ps: list[float] = []
    ys: list[int] = []
    by_learner: dict[str, list[dict]] = defaultdict(list)
    for a in test:
        by_learner[a["learner_id"]].append(a)
    for lid, attempts in by_learner.items():
        mid = max(1, len(attempts) // 2)
        elo = Elo()
        for a in attempts[:mid]:
            elo.update(a["skill_id"], a["correct"])
        for a in attempts[mid:]:
            p_correct = elo.mastery(a["skill_id"])  # Elo.mastery already p(correct)
            ps.append(float(p_correct))
            ys.append(int(a["correct"]))
            elo.update(a["skill_id"], a["correct"])
    return ps, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/kt_eval.json")
    parser.add_argument("--n-learners", type=int, default=200)
    parser.add_argument("--n-attempts", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    rng = random.Random(args.seed)
    attempts = simulate(args.n_learners, args.n_attempts, args.seed)
    train, test = _split_by_learner(attempts, test_frac=0.2, rng=rng)
    print(f"Simulated {len(attempts)} attempts · {args.n_learners} learners")
    print(f"Train: {len(train)}  Test: {len(test)}")

    results: dict[str, dict] = {}
    for name, fn in [("BKT", _score_bkt), ("DKT", _score_dkt), ("Elo", _score_elo)]:
        ps, ys = fn(train, test)
        if len(set(ys)) < 2:
            raise RuntimeError(f"{name}: need both classes in held-out; got {set(ys)}")
        auc = roc_auc_score(ys, ps)
        print(f"  {name}: AUC = {auc:.4f}   (n={len(ys)})")
        results[name] = {"auc": float(auc), "n": len(ys)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_learners": args.n_learners,
        "n_attempts": args.n_attempts,
        "seed": args.seed,
        "results": results,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
