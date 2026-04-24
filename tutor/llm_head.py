"""Optional LLM head (TinyLlama-1.1B Q4_K_M GGUF) for child-safe
natural-language encouragement and weekly parent-report summaries.

Design choices
--------------

- **Offline, CPU-only.** Loads via ``llama-cpp-python`` on CPU
  (llama.cpp kernels). Works in the child's tablet and in Colab.

- **Not in the inference hot path.** TinyLlama Q4_K_M on CPU is
  ~1.5 s per short reply, which would blow the 2.5 s per-cycle budget.
  The tutor's real-time feedback stays deterministic; the LLM is used
  for the weekly parent-report narrative and for pre-generated phrase
  pools — neither is latency-critical.

- **Optional.** If the GGUF file is missing, ``LLMHead`` returns
  ``None`` from every method; the app falls back to
  ``Tutor.feedback``'s deterministic phrasing without crashing.

- **Child-safe system prompt.** The system prompt constrains output
  to one short, positive sentence, with explicit no-criticism rules.

Model file location
-------------------
Default: ``~/.cache/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`` (set by
``scripts/download_llm.py``). Not bundled in ``tutor/`` so the 75 MB
on-device budget stays honest; the file is fetched at install time.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

Lang = Literal["kin", "en", "fr", "mix"]


# Resolution order, first match wins:
#   1. $TUTOR_LLM_GGUF environment variable.
#   2. The numeracy-tuned QLoRA product from scripts/train_llm_qlora.py.
#   3. The vanilla community TinyLlama Q4_K_M shipped by scripts/download_llm.py.
_CANDIDATES = [
    Path.home() / ".cache" / "llm" / "tinyllama-numeracy-Q4_K_M.gguf",
    Path.home() / ".cache" / "llm" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
]

def _resolve_default() -> Path:
    env = os.environ.get("TUTOR_LLM_GGUF")
    if env:
        return Path(env)
    for c in _CANDIDATES:
        if c.exists():
            return c
    return _CANDIDATES[-1]  # default fallback even if missing

_DEFAULT_GGUF = _resolve_default()


_SYSTEM_BY_LANG: dict[Lang, str] = {
    "en": (
        "You are a warm, encouraging math tutor for a 6-year-old. "
        "Reply in ONE short sentence, under 12 words. Be positive. "
        "Never criticise. No emoji. Use the child's language: English."
    ),
    "fr": (
        "Tu es un tuteur de maths bienveillant pour un enfant de 6 ans. "
        "Réponds en UNE courte phrase, moins de 12 mots. Reste positif. "
        "Aucune critique. Pas d'émoji. Langue: français."
    ),
    "kin": (
        "Uri umwarimu w'imibare ushimisha umwana w'imyaka 6. "
        "Subiza mu ijambo rimwe gito, ritarenze amagambo 12. Ba ushimisha. "
        "Ntugaragaze uburakari. Ururimi: Ikinyarwanda."
    ),
    "mix": (
        "You are a warm math tutor. Reply in ONE short sentence, under 12 "
        "words. Mix English and Kinyarwanda if the child used both."
    ),
}


class LLMHead:
    """Lazy-loaded TinyLlama Q4_K_M head. Safe to instantiate when the
    GGUF is missing — methods then return ``None``.
    """

    def __init__(
        self,
        gguf_path: Path | None = None,
        n_ctx: int = 512,
        n_threads: int = 4,
    ):
        self.gguf_path = Path(gguf_path or _DEFAULT_GGUF)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def available(self) -> bool:
        return self.gguf_path.exists()

    def _load(self) -> None:
        if self._llm is not None or not self.available():
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            return
        self._llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

    def _chat(self, system: str, user: str, max_tokens: int = 32) -> str | None:
        self._load()
        if self._llm is None:
            return None
        r = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
        )
        return r["choices"][0]["message"]["content"].strip()

    def encourage(self, correct: bool, skill: str, lang: Lang = "en") -> str | None:
        """Short encouragement phrase. Returns ``None`` if model missing.

        Not called in the inference hot path (see module docstring).
        Useful for pre-generating a pool of phrases at app boot.
        """
        action = "answered correctly" if correct else "gave a wrong answer but is trying hard"
        user = f"The child just {action} on a {skill.replace('_', ' ')} question."
        return self._chat(_SYSTEM_BY_LANG.get(lang, _SYSTEM_BY_LANG["en"]), user, max_tokens=24)

    def weekly_summary(
        self,
        skills_scores: dict[str, float],
        lang: Lang = "en",
    ) -> str | None:
        """Generate a 2-sentence weekly narrative for the parent-report
        QR-to-audio path. Called once per learner per week — latency is
        not a concern.
        """
        best = max(skills_scores, key=lambda s: skills_scores[s])
        worst = min(skills_scores, key=lambda s: skills_scores[s])
        user = (
            f"Tell the parent: the child is strongest at {best.replace('_', ' ')} "
            f"and could use more practice on {worst.replace('_', ' ')}. "
            "Two short positive sentences. Suggest one simple at-home activity."
        )
        return self._chat(_SYSTEM_BY_LANG.get(lang, _SYSTEM_BY_LANG["en"]), user, max_tokens=80)
