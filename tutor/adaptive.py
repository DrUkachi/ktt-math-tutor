"""Knowledge-tracing models + Elo baseline for next-item selection.

Three estimators implemented to the same interface:

    estimator.update(skill_id, correct: bool)
    estimator.mastery(skill_id) -> float in [0, 1]
    estimator.pick_next(items) -> Item

They are kept side-by-side so the kt_eval notebook can compute next-response
AUC for each on a held-out replay split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .curriculum_loader import Item


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
class KTEstimator(Protocol):
    def update(self, skill_id: str, correct: bool) -> None: ...
    def mastery(self, skill_id: str) -> float: ...
    def pick_next(self, items: list[Item]) -> Item: ...


# ---------------------------------------------------------------------------
# Bayesian Knowledge Tracing (closed-form, 4 params per skill)
# ---------------------------------------------------------------------------
@dataclass
class BKTParams:
    p_init: float = 0.2
    p_transit: float = 0.1
    p_slip: float = 0.1
    p_guess: float = 0.2


class BKT:
    """Standard 4-parameter BKT, one set of params per sub-skill."""

    def __init__(self, params: dict[str, BKTParams] | None = None):
        self.params = params or {}
        self._p_known: dict[str, float] = {}

    def _p(self, skill_id: str) -> BKTParams:
        return self.params.setdefault(skill_id, BKTParams())

    def update(self, skill_id: str, correct: bool) -> None:
        p = self._p(skill_id)
        prior = self._p_known.get(skill_id, p.p_init)
        if correct:
            num = prior * (1 - p.p_slip)
            den = num + (1 - prior) * p.p_guess
        else:
            num = prior * p.p_slip
            den = num + (1 - prior) * (1 - p.p_guess)
        post = num / den if den > 0 else prior
        self._p_known[skill_id] = post + (1 - post) * p.p_transit

    def mastery(self, skill_id: str) -> float:
        return self._p_known.get(skill_id, self._p(skill_id).p_init)

    def pick_next(self, items: list[Item]) -> Item:
        # Item difficulty is 1..10; map to [0, 1] for comparison with mastery.
        def gap(it: Item) -> float:
            target = 1.0 - self.mastery(it.skill)  # higher gap → easier item
            return abs((it.difficulty / 10.0) - target)
        return min(items, key=gap)


# ---------------------------------------------------------------------------
# Deep Knowledge Tracing (tiny GRU)
# ---------------------------------------------------------------------------
class DKT:
    """Tiny GRU DKT. Lazy-imports torch so the package imports cheaply."""

    def __init__(self, n_skills: int = 5, hidden: int = 32):
        self.n_skills = n_skills
        self.hidden = hidden
        self._model = None  # torch.nn.GRU built on first call
        self._h = None
        self._skill_to_idx: dict[str, int] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_skills: int, hidden: int):
                super().__init__()
                self.gru = nn.GRU(n_skills * 2, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_skills)

            def forward(self, x, h=None):
                y, h2 = self.gru(x, h)
                return torch.sigmoid(self.out(y)), h2

        self._model = _Net(self.n_skills, self.hidden)

    def _idx(self, skill_id: str) -> int:
        return self._skill_to_idx.setdefault(skill_id, len(self._skill_to_idx))

    def update(self, skill_id: str, correct: bool) -> None:
        self._ensure_model()
        import torch
        idx = self._idx(skill_id)
        x = torch.zeros(1, 1, self.n_skills * 2)
        x[0, 0, idx + (self.n_skills if correct else 0)] = 1.0
        with torch.no_grad():
            _, self._h = self._model(x, self._h)

    def mastery(self, skill_id: str) -> float:
        self._ensure_model()
        import torch
        idx = self._idx(skill_id)
        if self._h is None:
            return 0.5
        with torch.no_grad():
            probs = torch.sigmoid(self._model.out(self._h[-1]))
        return float(probs[0, idx].item())

    def pick_next(self, items: list[Item]) -> Item:
        def gap(it: Item) -> float:
            target = 1.0 - self.mastery(it.skill)
            return abs((it.difficulty / 10.0) - target)
        return min(items, key=gap)


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------
@dataclass
class Elo:
    k: float = 16.0
    learner_rating: float = 1000.0
    item_ratings: dict[str, float] = field(default_factory=dict)
    skill_ratings: dict[str, float] = field(default_factory=dict)

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, skill_id: str, correct: bool) -> None:
        rb = self.skill_ratings.setdefault(skill_id, 1000.0)
        e = self._expected(self.learner_rating, rb)
        s = 1.0 if correct else 0.0
        self.learner_rating += self.k * (s - e)
        self.skill_ratings[skill_id] = rb + self.k * ((1 - s) - (1 - e))

    def mastery(self, skill_id: str) -> float:
        rb = self.skill_ratings.get(skill_id, 1000.0)
        return self._expected(self.learner_rating, rb)

    def pick_next(self, items: list[Item]) -> Item:
        # Pick the item whose expected-correct probability is closest to 0.7
        # (the "zone of proximal development" sweet spot).
        def gap(it: Item) -> float:
            rb = self.skill_ratings.get(it.skill, 1000.0)
            return abs(self._expected(self.learner_rating, rb) - 0.7)
        return min(items, key=gap)
