"""Dyscalculia early-warning flag (brief stretch goal).

From the candidate brief:

> Dyscalculia early-warning: if the learner plateaus for 3+ sessions
> despite difficulty drops, surface a gentle 'talk to a teacher'
> message for the parent.

This module is deliberately conservative. A false positive here sends a
non-literate parent to a teacher unnecessarily; a false negative misses
a child who needs help. The thresholds below are defaults that can be
tuned per cohort without changing the API.

Plateau detection
-----------------
1. Group the learner's attempts into sessions by calendar day (UTC).
2. Keep only learners with ``>= min_sessions`` (default 3) recent
   sessions.
3. A plateau is declared when, across the last ``min_sessions``:
   - mean per-session correctness rate is ``<= correctness_ceiling``
     (default 0.5 — below the "I'm getting it" threshold), and
   - the trend is flat: ``(latest - earliest) <= slope_epsilon``
     (default 0.05).
4. ``despite difficulty drops`` is checked only when a ``Curriculum``
   is supplied: if the estimator has been lowering item difficulty
   (mean difficulty in the latest session is at least ``difficulty_drop``
   below the earliest session, default 1.0 on the 1-10 scale) AND
   correctness has not risen, the flag fires. Without a curriculum the
   check degenerates to correctness-only plateau, which still catches
   the most important case.

The function returns a structured verdict so the caller (parent report,
teacher dashboard, clinician escalation) can decide what to do with it.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .curriculum_loader import Curriculum
    from .storage import Attempt


@dataclass
class DyscalculiaFlag:
    flagged: bool
    reason: str
    sessions_considered: int
    correctness_trend: list[float]
    difficulty_trend: list[float] | None
    message_for_parent: str

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "reason": self.reason,
            "sessions_considered": self.sessions_considered,
            "correctness_trend": [round(x, 3) for x in self.correctness_trend],
            "difficulty_trend": (
                [round(x, 2) for x in self.difficulty_trend]
                if self.difficulty_trend is not None else None
            ),
            "message_for_parent": self.message_for_parent,
        }


_MESSAGE = (
    "Your child has worked hard but is still finding some numbers "
    "tricky. It could help to talk to their teacher about a little "
    "extra practice — nothing to worry about."
)


def flag_plateau(
    attempts: "list[Attempt]",
    *,
    curriculum: "Curriculum | None" = None,
    min_sessions: int = 3,
    correctness_ceiling: float = 0.5,
    slope_epsilon: float = 0.05,
    difficulty_drop: float = 1.0,
) -> DyscalculiaFlag:
    """Inspect a learner's attempt stream and decide whether to escalate.

    Parameters
    ----------
    attempts:
        As returned by ``ProgressStore.replay(learner_id)``. Can be
        empty or short — the function handles that gracefully.
    curriculum:
        If provided, enables the "despite difficulty drops" clause by
        joining ``attempt.item_id`` to ``Item.difficulty``. Items missing
        from the curriculum contribute no difficulty signal.
    """
    # 1. Group by calendar day (UTC).
    by_day: dict[str, list] = defaultdict(list)
    for a in attempts:
        day = dt.datetime.fromtimestamp(a.ts, dt.timezone.utc).date().isoformat()
        by_day[day].append(a)

    days = sorted(by_day.keys())
    if len(days) < min_sessions:
        return DyscalculiaFlag(
            flagged=False,
            reason=f"not enough sessions ({len(days)} < {min_sessions})",
            sessions_considered=len(days),
            correctness_trend=[],
            difficulty_trend=None,
            message_for_parent="",
        )

    recent_days = days[-min_sessions:]
    correctness_trend: list[float] = []
    difficulty_trend: list[float] | None = [] if curriculum is not None else None

    for day in recent_days:
        day_attempts = by_day[day]
        if not day_attempts:
            continue
        correct = sum(1 for x in day_attempts if x.correct) / len(day_attempts)
        correctness_trend.append(correct)
        if difficulty_trend is not None:
            diffs = []
            for x in day_attempts:
                try:
                    diffs.append(curriculum.get(x.item_id).difficulty)  # type: ignore[union-attr]
                except KeyError:
                    continue
            if diffs:
                difficulty_trend.append(sum(diffs) / len(diffs))

    mean_correct = sum(correctness_trend) / len(correctness_trend)
    slope = correctness_trend[-1] - correctness_trend[0]
    plateau_flat = mean_correct <= correctness_ceiling and abs(slope) <= slope_epsilon

    if not plateau_flat:
        return DyscalculiaFlag(
            flagged=False,
            reason=(
                f"correctness is moving ({correctness_trend[0]:.2f} -> "
                f"{correctness_trend[-1]:.2f}) or mean is above ceiling "
                f"({mean_correct:.2f} > {correctness_ceiling})"
            ),
            sessions_considered=len(recent_days),
            correctness_trend=correctness_trend,
            difficulty_trend=difficulty_trend,
            message_for_parent="",
        )

    # With a curriculum, require evidence that the estimator has been
    # trying to help by lowering difficulty. Without one, correctness
    # plateau alone is sufficient.
    if difficulty_trend is not None and len(difficulty_trend) >= 2:
        dropped_by = difficulty_trend[0] - difficulty_trend[-1]
        if dropped_by < difficulty_drop:
            return DyscalculiaFlag(
                flagged=False,
                reason=(
                    f"correctness plateaued but difficulty has not dropped enough "
                    f"({dropped_by:+.2f} < {difficulty_drop}) — may be genuinely "
                    f"stuck on hard items rather than needing escalation"
                ),
                sessions_considered=len(recent_days),
                correctness_trend=correctness_trend,
                difficulty_trend=difficulty_trend,
                message_for_parent="",
            )

    return DyscalculiaFlag(
        flagged=True,
        reason=(
            f"correctness flat at {mean_correct:.2f} across "
            f"{len(recent_days)} sessions"
            + (
                f" despite difficulty drop "
                f"{difficulty_trend[0]:.1f} -> {difficulty_trend[-1]:.1f}"
                if difficulty_trend and len(difficulty_trend) >= 2 else ""
            )
        ),
        sessions_considered=len(recent_days),
        correctness_trend=correctness_trend,
        difficulty_trend=difficulty_trend,
        message_for_parent=_MESSAGE,
    )
