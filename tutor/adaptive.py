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
    """Tiny GRU DKT with a real backprop trainer.

    Input  : one-hot ``[skill_id, correct_flag]`` ∈ ℝ^{2·n_skills}.
             First ``n_skills`` cells are active when correct=True;
             next ``n_skills`` when correct=False.
    Output : per-skill next-correct logits. Sigmoid is applied only at
             readout so training uses ``binary_cross_entropy_with_logits``.

    Usage::

        dkt = DKT(skill_to_idx={"counting": 0, ...})
        dkt.fit(train_trajectories, epochs=40)          # GPU-free, 32-dim hidden
        dkt.reset(); dkt.update("counting", True)
        dkt.mastery("counting")                          # p(next attempt correct)
    """

    def __init__(
        self,
        n_skills: int = 5,
        hidden: int = 32,
        skill_to_idx: dict[str, int] | None = None,
    ):
        self.n_skills = n_skills
        self.hidden = hidden
        self._model = None
        self._h = None
        self._skill_to_idx: dict[str, int] = dict(skill_to_idx or {})

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_skills: int, hidden: int):
                super().__init__()
                self.gru = nn.GRU(n_skills * 2, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_skills)

            def forward(self, x, h=None):
                y, h2 = self.gru(x, h)
                return self.out(y), h2  # logits; sigmoid at eval only

        self._model = _Net(self.n_skills, self.hidden)

    def _idx(self, skill_id: str) -> int:
        if skill_id not in self._skill_to_idx:
            # Cap at n_skills; extras map to the last bucket (shouldn't
            # happen if the caller passes the canonical skill list).
            self._skill_to_idx[skill_id] = min(
                len(self._skill_to_idx), self.n_skills - 1
            )
        return self._skill_to_idx[skill_id]

    def _encode(self, skill_id: str, correct: bool):
        import torch
        idx = self._idx(skill_id)
        x = torch.zeros(1, 1, self.n_skills * 2)
        # Correct=first block so the signal is unambiguous at readout.
        x[0, 0, idx + (0 if correct else self.n_skills)] = 1.0
        return x

    def reset(self) -> None:
        self._h = None

    def update(self, skill_id: str, correct: bool) -> None:
        self._ensure_model()
        import torch
        x = self._encode(skill_id, correct)
        with torch.no_grad():
            _, self._h = self._model(x, self._h)

    def mastery(self, skill_id: str) -> float:
        self._ensure_model()
        import torch
        idx = self._idx(skill_id)
        if self._h is None:
            return 0.5
        with torch.no_grad():
            logits = self._model.out(self._h[-1])
            probs = torch.sigmoid(logits)
        return float(probs[0, idx].item())

    def fit(
        self,
        trajectories: list[list[tuple[str, bool]]],
        epochs: int = 40,
        lr: float = 5e-3,
    ) -> "DKT":
        """Train on a list of per-learner (skill, correct) sequences.

        Objective: given the GRU output after step t, predict whether
        the attempt at step t+1 on its skill will be correct. Standard
        DKT next-response loss.
        """
        self._ensure_model()
        import torch
        import torch.nn.functional as F

        # Pre-register the full skill vocabulary so indexing is stable.
        for traj in trajectories:
            for skill, _ in traj:
                self._idx(skill)

        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        device = next(self._model.parameters()).device
        for epoch in range(epochs):
            total = 0.0
            n = 0
            for traj in trajectories:
                if len(traj) < 2:
                    continue
                # Build per-step input and next-step supervision.
                xs = [self._encode(s, c).squeeze(0) for s, c in traj]
                skill_idx = [self._idx(s) for s, _ in traj]
                ys = [1.0 if c else 0.0 for _, c in traj]
                x = torch.cat(xs, dim=0).unsqueeze(0).to(device)  # [1, T, 2n]
                logits, _ = self._model(x)  # [1, T, n_skills]
                # Predict attempt t+1 from hidden at t.
                pred = logits[0, :-1, :]  # [T-1, n_skills]
                target_idx = torch.tensor(skill_idx[1:], device=device)
                target_y = torch.tensor(ys[1:], device=device)
                picked = pred.gather(1, target_idx.unsqueeze(-1)).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(picked, target_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
        return self

    def save(self, path: str | "Path") -> None:
        from pathlib import Path as _Path
        import torch
        self._ensure_model()
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "skill_to_idx": self._skill_to_idx,
            "n_skills": self.n_skills,
            "hidden": self.hidden,
        }, str(path))

    @classmethod
    def load(cls, path: str | "Path") -> "DKT":
        import torch
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        dkt = cls(n_skills=blob["n_skills"], hidden=blob["hidden"],
                  skill_to_idx=blob["skill_to_idx"])
        dkt._ensure_model()
        dkt._model.load_state_dict(blob["state_dict"])
        return dkt

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
