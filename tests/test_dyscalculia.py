"""Unit tests for the dyscalculia early-warning flag.

Covers the three cases that matter for Live Defense:
- Flat-low correctness across >=3 days → flag fires.
- Improving correctness → never fires (even if low).
- Too few sessions → returns unflagged with a clear reason.
"""
from __future__ import annotations

import datetime as dt
import time

from tutor.dyscalculia import flag_plateau
from tutor.storage import Attempt


def _attempt(day_offset: int, correct: bool, skill: str = "counting",
             item_id: str = "X") -> Attempt:
    base = dt.datetime(2026, 4, 20, 10, 0, 0).timestamp()
    return Attempt(
        learner_id="L1",
        item_id=item_id,
        skill_id=skill,
        correct=correct,
        response_ms=500,
        ts=base + day_offset * 86400,
    )


def test_flat_low_correctness_flags() -> None:
    # Three consecutive days, ~40% correct each, no upward trend.
    attempts = []
    for day in range(3):
        for i in range(10):
            attempts.append(_attempt(day, correct=(i < 4)))
    flag = flag_plateau(attempts)
    assert flag.flagged is True, flag.reason
    assert "flat" in flag.reason
    assert flag.message_for_parent  # non-empty parent guidance
    assert flag.sessions_considered == 3


def test_improving_correctness_does_not_flag() -> None:
    # Day 0: 20% → Day 1: 50% → Day 2: 80%. Clear upward trend.
    attempts = []
    rates = [2, 5, 8]  # out of 10
    for day, hits in enumerate(rates):
        for i in range(10):
            attempts.append(_attempt(day, correct=(i < hits)))
    flag = flag_plateau(attempts)
    assert flag.flagged is False
    assert "moving" in flag.reason


def test_too_few_sessions_returns_unflagged() -> None:
    attempts = [_attempt(0, correct=False) for _ in range(10)]
    flag = flag_plateau(attempts)
    assert flag.flagged is False
    assert "not enough sessions" in flag.reason


def test_flag_dict_shape_matches_report_schema() -> None:
    attempts = [_attempt(d, correct=(i < 4)) for d in range(3) for i in range(10)]
    d = flag_plateau(attempts).to_dict()
    assert set(d) == {"flagged", "reason", "sessions_considered",
                      "correctness_trend", "difficulty_trend",
                      "message_for_parent"}
    assert d["flagged"] is True
    assert len(d["correctness_trend"]) == 3
