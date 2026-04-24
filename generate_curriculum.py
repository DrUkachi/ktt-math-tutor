"""Expand the 12-item curriculum_seed.json into a ≥ 60-item full curriculum.

Strategy: programmatically generate variants for the four arithmetic-style
skills (counting, addition, subtraction, number_sense) so we keep the same
schema as the seed. Word problems are kept hand-authored — quality matters
more than quantity for those.

Usage:
    python generate_curriculum.py --out data/T3.1_Math_Tutor/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

KIN_NUM = {
    1: "rimwe", 2: "kabiri", 3: "gatatu", 4: "kane", 5: "gatanu",
    6: "gatandatu", 7: "karindwi", 8: "umunani", 9: "icyenda", 10: "icumi",
}
EN_OBJECTS = ["apples", "goats", "cows", "drums", "beads", "mangoes", "books", "pots"]
KIN_OBJECTS = {
    "apples": "Pome", "goats": "Ihene", "cows": "Inka", "drums": "Ingoma",
    "beads": "Imitako", "mangoes": "Imyembe", "books": "Ibitabo", "pots": "Inkono",
}
FR_OBJECTS = {
    "apples": "pommes", "goats": "chèvres", "cows": "vaches", "drums": "tambours",
    "beads": "perles", "mangoes": "mangues", "books": "livres", "pots": "pots",
}


def age_band_for(difficulty: int) -> str:
    if difficulty <= 2:
        return "5-6"
    if difficulty <= 4:
        return "6-7"
    if difficulty <= 6:
        return "7-8"
    return "8-9"


def gen_counting(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        count = rng.randint(2, 9)
        obj = rng.choice(EN_OBJECTS)
        difficulty = max(1, min(10, count // 2))
        out.append({
            "id": f"GC{i+1:03d}",
            "skill": "counting",
            "difficulty": difficulty,
            "age_band": age_band_for(difficulty),
            "stem_en": f"How many {obj}?",
            "stem_fr": f"Combien de {FR_OBJECTS[obj]}?",
            "stem_kin": f"{KIN_OBJECTS[obj]} zingahe?",
            "visual": f"{obj}_{count}",
            "answer_int": count,
        })
    return out


def gen_addition(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        difficulty = max(2, min(10, (a + b) // 2))
        out.append({
            "id": f"GA{i+1:03d}",
            "skill": "addition",
            "difficulty": difficulty,
            "age_band": age_band_for(difficulty),
            "stem_en": f"{a} plus {b} equals?",
            "stem_fr": f"{a} plus {b} égale?",
            "stem_kin": f"{a} + {b} ni angahe?",
            "visual": f"beads_{a}_plus_{b}",
            "answer_int": a + b,
        })
    return out


def gen_subtraction(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        a = rng.randint(3, 12)
        b = rng.randint(1, a - 1)
        difficulty = max(3, min(10, a // 2))
        out.append({
            "id": f"GS{i+1:03d}",
            "skill": "subtraction",
            "difficulty": difficulty,
            "age_band": age_band_for(difficulty),
            "stem_en": f"{a} minus {b} equals?",
            "stem_fr": f"{a} moins {b} égale?",
            "stem_kin": f"{a} - {b} ni angahe?",
            "visual": f"beads_{a}_minus_{b}",
            "answer_int": a - b,
        })
    return out


def gen_number_sense(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        a, b = sorted(rng.sample(range(1, 20), 2))
        difficulty = max(2, min(10, b // 2))
        out.append({
            "id": f"GN{i+1:03d}",
            "skill": "number_sense",
            "difficulty": difficulty,
            "age_band": age_band_for(difficulty),
            "stem_en": f"Which number is bigger: {a} or {b}?",
            "stem_fr": f"Quel nombre est plus grand: {a} ou {b}?",
            "stem_kin": f"Ni iyihe nimero nini: {a} cyangwa {b}?",
            "visual": f"compare_{a}_{b}",
            "answer_int": b,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/T3.1_Math_Tutor"),
                        help="Output directory; writes curriculum.json there.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=80)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    seed_path = args.out / "curriculum_seed.json"
    if seed_path.exists():
        with open(seed_path, "r", encoding="utf-8") as fh:
            items = json.load(fh)
    else:
        items = []

    per_skill = max(1, (args.target_size - len(items)) // 4)
    items += gen_counting(rng, per_skill)
    items += gen_addition(rng, per_skill)
    items += gen_subtraction(rng, per_skill)
    items += gen_number_sense(rng, per_skill)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "curriculum.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
