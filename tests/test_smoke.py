"""Minimal smoke tests so CI is green from the scaffold commit onward."""
from __future__ import annotations

from pathlib import Path

from tutor.adaptive import BKT, Elo
from tutor.curriculum_loader import SKILLS, Curriculum
from tutor.lang_detect import detect, number_words

SEED = Path(__file__).parent.parent / "data" / "T3.1_Math_Tutor" / "curriculum_seed.json"


def test_curriculum_loads() -> None:
    c = Curriculum.from_json(SEED)
    assert len(c) >= 12
    assert all(it.skill in SKILLS for it in c)


def test_bkt_updates_increase_mastery_after_correct() -> None:
    bkt = BKT()
    before = bkt.mastery("counting")
    for _ in range(5):
        bkt.update("counting", correct=True)
    after = bkt.mastery("counting")
    assert after > before


def test_elo_pickup() -> None:
    c = Curriculum.from_json(SEED)
    elo = Elo()
    chosen = elo.pick_next(list(c))
    assert chosen.id


def test_lang_detect_basic() -> None:
    assert detect("yego rimwe kabiri") == "kin"
    assert detect("five and six") == "en"
    assert detect("trois quatre") == "fr"


def test_number_word_extraction() -> None:
    nw = number_words("I have five mangues")
    assert "five" in nw["en"]
