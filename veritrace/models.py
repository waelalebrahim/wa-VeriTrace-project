"""Data models for VeriTrace verification results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ConfidenceTier(str, Enum):
    """How well a claim is supported by the loaded sources.

    HIGH   -> strongly supported; answer it and cite the source.
    MEDIUM -> partially supported; surface it but ask the reader to verify.
    LOW    -> not supported by any source; the honest answer is "I don't know".
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Citation:
    """A pointer from a claim back to the source passage that supports it."""

    doc_id: str
    source_name: str
    passage: str
    score: float
    doc_hash: Optional[str] = None
    doc_date: Optional[str] = None  # ISO date string, if the source was dated

    def __str__(self) -> str:
        loc = self.source_name or self.doc_id
        return f"[{loc}] {self.passage!r} (match={self.score:.3f})"


@dataclass
class ClaimResult:
    """The verdict for a single factual claim extracted from the answer."""

    claim: str
    tier: ConfidenceTier
    score: float
    citation: Optional[Citation] = None

    @property
    def grounded(self) -> bool:
        return self.tier in (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM)


@dataclass
class VerificationResult:
    """The full report for one verified answer."""

    claims: List[ClaimResult] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True only if every claim is at least MEDIUM-supported."""
        return all(c.grounded for c in self.claims)

    def by_tier(self, tier: ConfidenceTier) -> List[ClaimResult]:
        return [c for c in self.claims if c.tier == tier]

    @property
    def ungrounded(self) -> List[ClaimResult]:
        return self.by_tier(ConfidenceTier.LOW)

    def summary(self) -> str:
        h = len(self.by_tier(ConfidenceTier.HIGH))
        m = len(self.by_tier(ConfidenceTier.MEDIUM))
        low = len(self.by_tier(ConfidenceTier.LOW))
        return f"{len(self.claims)} claims -> {h} high, {m} medium, {low} low"

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "summary": self.summary(),
            "claims": [
                {
                    "claim": c.claim,
                    "tier": c.tier.value,
                    "score": round(c.score, 4),
                    "citation": (
                        {
                            "doc_id": c.citation.doc_id,
                            "source_name": c.citation.source_name,
                            "passage": c.citation.passage,
                            "score": round(c.citation.score, 4),
                            "doc_hash": c.citation.doc_hash,
                            "doc_date": c.citation.doc_date,
                        }
                        if c.citation
                        else None
                    ),
                }
                for c in self.claims
            ],
        }
