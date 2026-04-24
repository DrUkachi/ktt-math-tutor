"""End-to-end inference loop: stimulus → response → feedback in < 2.5 s.

The loop is deliberately small and synchronous so it can be reasoned about on
a Live Defense screen-share. Heavier components (LLM head, ASR) are loaded
lazily on first use so cold-start cost lands on the first item, not on
``import tutor``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .adaptive import BKT, KTEstimator
from .curriculum_loader import Curriculum, Item
from .lang_detect import Lang, detect
from .storage import Attempt, ProgressStore


@dataclass
class Cycle:
    """One stimulus-response-feedback cycle, retained for the kt_eval replay."""

    item: Item
    response: str
    correct: bool
    response_ms: int
    lang_detected: Lang


class Tutor:
    """Owns the curriculum, the KT estimator, and the storage handle."""

    def __init__(
        self,
        learner_id: str,
        curriculum_path: str | Path,
        store: ProgressStore | None = None,
        estimator: KTEstimator | None = None,
        default_lang: Lang = "kin",
    ):
        self.learner_id = learner_id
        self.curriculum = Curriculum.from_json(curriculum_path)
        self.store = store or ProgressStore()
        self.estimator: KTEstimator = estimator or BKT()
        self.default_lang: Lang = default_lang
        # Replay prior attempts so the estimator starts warm.
        for attempt in self.store.replay(learner_id):
            self.estimator.update(attempt.skill_id, attempt.correct)

    def next_item(self, age_band: str) -> Item:
        candidates = self.curriculum.filter(age_band=age_band)
        if not candidates:
            candidates = list(self.curriculum)
        return self.estimator.pick_next(candidates)

    def score(self, item: Item, response: str) -> bool:
        """Numeric tolerance: child must produce the digits or word for ``answer_int``."""
        from .lang_detect import number_words

        # Direct integer extraction first.
        digits = "".join(c for c in response if c.isdigit())
        if digits and int(digits) == item.answer_int:
            return True

        # Number-word extraction in any of the three languages.
        words = number_words(response)
        word_to_int = {
            **dict(zip(
                ["one", "two", "three", "four", "five",
                 "six", "seven", "eight", "nine", "ten"], range(1, 11))),
            **dict(zip(
                ["un", "deux", "trois", "quatre", "cinq",
                 "six", "sept", "huit", "neuf", "dix"], range(1, 11))),
            **dict(zip(
                ["rimwe", "kabiri", "gatatu", "kane", "gatanu",
                 "gatandatu", "karindwi", "umunani", "icyenda", "icumi"],
                range(1, 11))),
        }
        for lang_words in words.values():
            for w in lang_words:
                if word_to_int.get(w) == item.answer_int:
                    return True
        return False

    def feedback(self, item: Item, correct: bool, lang: Lang) -> str:
        """Deterministic short feedback. The LLM head is optional polish."""
        if correct:
            return {
                "kin": f"Yego! Igisubizo ni {item.answer_int}.",
                "en":  f"Yes! The answer is {item.answer_int}.",
                "fr":  f"Oui! La réponse est {item.answer_int}.",
                "mix": f"Yego! The answer is {item.answer_int}.",
            }[lang]
        return {
            "kin": "Ntibikiriho. Reka tugerageze nanone.",
            "en":  "Not quite. Let's try again.",
            "fr":  "Pas tout à fait. Essayons encore.",
            "mix": "Not quite. Reka tugerageze nanone.",
        }[lang]

    def ask(self, age_band: str) -> Item:
        """Pick and return the next item. Interactive UX uses this to
        display the prompt BEFORE the child answers (see demo.py).
        """
        return self.next_item(age_band)

    def answer(self, item: Item, response_text: str) -> Cycle:
        """Score ``response_text`` against a previously-asked ``item``.
        Records the attempt and updates the estimator.
        """
        t0 = time.perf_counter()
        lang = detect(response_text) if response_text else self.default_lang
        correct = self.score(item, response_text)
        response_ms = int((time.perf_counter() - t0) * 1000)
        self.store.record(Attempt(
            learner_id=self.learner_id,
            item_id=item.id, skill_id=item.skill,
            correct=correct, response_ms=response_ms, ts=time.time(),
        ))
        self.estimator.update(item.skill, correct)
        return Cycle(
            item=item, response=response_text, correct=correct,
            response_ms=response_ms, lang_detected=lang,
        )

    def step(self, age_band: str, response_text: str) -> Cycle:
        """Ask + answer in one call. Kept for batched replay / tests;
        interactive UIs should use :meth:`ask` and :meth:`answer`."""
        return self.answer(self.ask(age_band), response_text)
