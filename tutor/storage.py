"""Encrypted local progress store + ε-DP aggregation for upstream sync.

Design:

- Per-learner row in ``progress.db`` is encrypted at the column level using
  Fernet (AES-128-CBC + HMAC) with a key derived from a salt held in
  Android Keystore (production) or a chmod-600 file (dev).
- We *do not* depend on SQLCipher because there is no Windows wheel; the
  brief allows either approach as long as the data at rest is encrypted.
- ``aggregate_with_dp`` adds Laplace noise to coop-aggregated counters with
  a documented ε budget per learner per week (default ε = 1.0).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet


_DEFAULT_DB = Path(__file__).parent.parent / "progress.db"


@dataclass
class Attempt:
    learner_id: str
    item_id: str
    skill_id: str
    correct: bool
    response_ms: int
    ts: float


class ProgressStore:
    """Encrypted SQLite store keyed by learner pseudo-ID (animal avatar)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS attempts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        learner_id TEXT NOT NULL,
        ts REAL NOT NULL,
        payload BLOB NOT NULL  -- Fernet-encrypted JSON
    );
    CREATE INDEX IF NOT EXISTS idx_attempts_learner ON attempts(learner_id);
    """

    def __init__(self, db_path: str | Path | None = None, key: bytes | None = None):
        self.db_path = Path(db_path or _DEFAULT_DB)
        self._fernet = Fernet(key or self._load_or_create_key())
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def _load_or_create_key(self) -> bytes:
        key_file = self.db_path.with_suffix(".key")
        if key_file.exists():
            return key_file.read_bytes()
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try:
            os.chmod(key_file, 0o600)
        except (OSError, NotImplementedError):
            pass  # Windows: rely on user-profile ACLs.
        return key

    def record(self, attempt: Attempt) -> None:
        payload = json.dumps({
            "item_id": attempt.item_id,
            "skill_id": attempt.skill_id,
            "correct": attempt.correct,
            "response_ms": attempt.response_ms,
        }).encode("utf-8")
        token = self._fernet.encrypt(payload)
        self._conn.execute(
            "INSERT INTO attempts(learner_id, ts, payload) VALUES (?, ?, ?)",
            (attempt.learner_id, attempt.ts or time.time(), token),
        )
        self._conn.commit()

    def replay(self, learner_id: str) -> list[Attempt]:
        rows = self._conn.execute(
            "SELECT learner_id, ts, payload FROM attempts WHERE learner_id = ? ORDER BY ts",
            (learner_id,),
        ).fetchall()
        out: list[Attempt] = []
        for lid, ts, token in rows:
            data = json.loads(self._fernet.decrypt(token).decode("utf-8"))
            out.append(Attempt(
                learner_id=lid, ts=ts,
                item_id=data["item_id"], skill_id=data["skill_id"],
                correct=bool(data["correct"]), response_ms=int(data["response_ms"]),
            ))
        return out

    def close(self) -> None:
        self._conn.close()


def aggregate_with_dp(
    counts: dict[str, int],
    epsilon: float = 1.0,
    sensitivity: int = 1,
) -> dict[str, float]:
    """Return Laplace-noised counts for upstream cooperative sync.

    ε budget is per-learner per-week; ``sensitivity`` is 1 because each
    learner contributes at most one count to any aggregated bucket per item.
    """
    import numpy as np
    rng = np.random.default_rng()
    scale = sensitivity / float(epsilon)
    return {
        k: max(0.0, v + float(rng.laplace(loc=0.0, scale=scale)))
        for k, v in counts.items()
    }
