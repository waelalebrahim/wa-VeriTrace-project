"""Break an answer into individual factual claims to check one at a time."""

from __future__ import annotations

import re
from typing import List

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_HAS_ALPHA = re.compile(r"[A-Za-z]")


def split_into_claims(text: str, min_words: int = 3) -> List[str]:
    """Split an answer into candidate factual claims.

    V1 uses sentence segmentation. This is a deliberate, honest simplification:
    one sentence can hold several claims, and segmentation is imperfect. It is
    good enough to catch wholesale fabrication, which is the main job. Smarter
    claim extraction is on the roadmap (see README).
    """
    text = (text or "").strip()
    if not text:
        return []

    claims: List[str] = []
    for raw in _SENTENCE_SPLIT.split(text):
        s = raw.strip()
        if not s or not _HAS_ALPHA.search(s):
            continue
        if len(s.split()) < min_words:
            continue
        # skip pure questions -- they assert nothing to verify
        if s.endswith("?"):
            continue
        claims.append(s)
    return claims
