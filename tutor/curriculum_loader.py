"""Loads the curriculum JSON and exposes filtering / lookup.

Schema matches the seed file ``data/T3.1_Math_Tutor/curriculum_seed.json``:

    id          : str    — e.g. "C001"
    skill       : str    — one of {counting, number_sense, addition, subtraction, word_problem}
    difficulty  : int    — 1..10
    age_band    : str    — "5-6" | "6-7" | "7-8" | "8-9"
    stem_en     : str    — English prompt
    stem_fr     : str?   — French prompt (optional)
    stem_kin    : str?   — Kinyarwanda prompt (optional)
    visual      : str?   — asset key, e.g. "goats_5"
    answer_int  : int    — expected numeric answer
    tts_en/fr/kin : str? — relative path to cached TTS wav

Next-item selection is the knowledge-tracing model's job (see tutor.adaptive);
this module only loads, indexes, and filters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SKILLS: tuple[str, ...] = (
    "counting",
    "number_sense",
    "addition",
    "subtraction",
    "word_problem",
)


@dataclass(frozen=True)
class Item:
    id: str
    skill: str
    difficulty: int
    age_band: str
    stem_en: str
    answer_int: int
    stem_fr: str | None = None
    stem_kin: str | None = None
    visual: str | None = None
    tts_en: str | None = None
    tts_fr: str | None = None
    tts_kin: str | None = None

    def stem(self, lang: str) -> str:
        """Return the prompt for ``lang`` ('en' | 'fr' | 'kin'), falling back to EN."""
        return getattr(self, f"stem_{lang}", None) or self.stem_en

    def tts_path(self, lang: str) -> str | None:
        return getattr(self, f"tts_{lang}", None)


class Curriculum:
    def __init__(self, items: list[Item]):
        self._by_id: dict[str, Item] = {it.id: it for it in items}
        self._items: list[Item] = items

    @classmethod
    def from_json(cls, path: str | Path) -> "Curriculum":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        items: list[Item] = []
        for row in raw:
            # Drop unknown keys defensively so a curriculum extension does
            # not crash the loader.
            allowed = {f for f in Item.__dataclass_fields__}
            items.append(Item(**{k: v for k, v in row.items() if k in allowed}))
        return cls(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def get(self, item_id: str) -> Item:
        return self._by_id[item_id]

    def filter(
        self,
        *,
        age_band: str | None = None,
        skill: str | None = None,
        max_difficulty: int | None = None,
    ) -> list[Item]:
        out: Iterable[Item] = self._items
        if age_band is not None:
            out = (i for i in out if i.age_band == age_band)
        if skill is not None:
            out = (i for i in out if i.skill == skill)
        if max_difficulty is not None:
            out = (i for i in out if i.difficulty <= max_difficulty)
        return list(out)
