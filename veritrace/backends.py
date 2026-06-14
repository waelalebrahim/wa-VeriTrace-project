"""Verification backends: how a claim is matched against source passages.

The default backend (LexicalBackend) is pure standard library -- no model
downloads, no API keys, runs anywhere. It is intentionally simple and honest:
it catches claims that share no real content with any source (the fabrication
case). For semantic paraphrase matching, plug in the optional EmbeddingBackend
or LlmJudgeBackend.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple

from .sources import Passage

# A small, deliberately tiny stopword set. Keep dependencies at zero.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as", "by",
    "that", "this", "these", "those", "it", "its", "from", "into", "about",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "not", "no", "than", "then", "so", "if", "which", "who", "what", "there",
}
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if len(t) < 2 or t in _STOPWORDS:
            continue
        out.append(t)
    return out


class VerifierBackend(ABC):
    """Strategy for scoring how well a claim is supported by passages."""

    @abstractmethod
    def index(self, passages: List[Passage]) -> None:
        """Prepare to match claims against this set of passages."""

    @abstractmethod
    def rank(self, claim: str) -> List[Tuple[int, float]]:
        """Return (passage_index, score) pairs, best first. Scores in [0, 1]."""


class LexicalBackend(VerifierBackend):
    """TF-IDF cosine similarity over the source passages. Zero dependencies."""

    # How much each signal counts. Coverage dominates on purpose: a claim is
    # only "supported" if most of *it* appears in the source, not merely if it
    # shares a subject. Cosine alone lets a fabrication ride a real entity.
    _W_COSINE = 0.3
    _W_COVERAGE = 0.7

    def __init__(self):
        self._passages: List[Passage] = []
        self._idf: dict = {}
        self._vectors: List[dict] = []  # normalized tf-idf per passage
        self._token_sets: List[set] = []  # token set per passage, for coverage

    def index(self, passages: List[Passage]) -> None:
        self._passages = passages
        docs = [_tokens(p.text) for p in passages]
        n = len(docs)
        df: Counter = Counter()
        for toks in docs:
            for t in set(toks):
                df[t] += 1
        # smoothed idf
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorize(toks) for toks in docs]
        self._token_sets = [set(toks) for toks in docs]

    def _coverage(self, claim_toks: List[str], passage_tokens: set) -> float:
        """IDF-weighted fraction of the claim's content found in the passage.

        This is recall of the claim, not symmetric similarity. A claim that
        invents a predicate ("...cures every disease") loses most of its weight
        because those tokens are absent from the source.
        """
        if not claim_toks:
            return 0.0
        total = sum(self._idf.get(t, 1.0) for t in set(claim_toks))
        hit = sum(
            self._idf.get(t, 1.0) for t in set(claim_toks) if t in passage_tokens
        )
        return hit / total if total else 0.0

    def _vectorize(self, toks: List[str]) -> dict:
        if not toks:
            return {}
        tf = Counter(toks)
        vec = {t: (c / len(toks)) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {t: v / norm for t, v in vec.items()}

    def rank(self, claim: str) -> List[Tuple[int, float]]:
        claim_toks = _tokens(claim)
        q = self._vectorize(claim_toks)
        if not q:
            return [(i, 0.0) for i in range(len(self._passages))]
        scores = []
        for i, vec in enumerate(self._vectors):
            # cosine of two normalized vectors == dot product
            cosine = sum(q[t] * vec.get(t, 0.0) for t in q)
            coverage = self._coverage(claim_toks, self._token_sets[i])
            blended = self._W_COSINE * cosine + self._W_COVERAGE * coverage
            scores.append((i, blended))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class EmbeddingBackend(VerifierBackend):
    """Semantic matching via sentence-transformers (optional extra).

    Install with:  pip install "veritrace[embeddings]"
    Catches paraphrases the lexical backend misses, at the cost of a model
    download. Falls back loudly if the dependency is missing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "EmbeddingBackend needs sentence-transformers. "
                'Install with: pip install "veritrace[embeddings]"'
            ) from e
        self._model = SentenceTransformer(model_name)
        self._passages: List[Passage] = []
        self._emb = None

    def index(self, passages: List[Passage]) -> None:  # pragma: no cover
        self._passages = passages
        texts = [p.text for p in passages]
        self._emb = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def rank(self, claim: str) -> List[Tuple[int, float]]:  # pragma: no cover
        import numpy as np  # local import; numpy ships with sentence-transformers

        q = self._model.encode([claim], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._emb @ q
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order]
