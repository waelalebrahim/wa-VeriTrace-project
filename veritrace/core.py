"""VeriTrace: source-grounded verification for LLM answers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from .backends import LexicalBackend, VerifierBackend
from .claims import split_into_claims
from .exceptions import VeriTraceSourceFault
from .models import Citation, ClaimResult, ConfidenceTier, VerificationResult
from .sources import SourceDocument, SourceStore


class VeriTrace:
    """Verify that an answer is grounded in a fixed set of trusted sources.

    Example
    -------
    >>> vt = VeriTrace()
    >>> vt.add_source("Water boils at 100 degrees Celsius at sea level.", id="phys")
    >>> result = vt.verify("Water boils at 100 C at sea level. It also cures cancer.")
    >>> result.summary()
    '2 claims -> 1 high, 0 medium, 1 low'
    """

    def __init__(
        self,
        backend: Optional[VerifierBackend] = None,
        high_threshold: float = 0.55,
        medium_threshold: float = 0.18,
        conflict_margin: float = 0.05,
    ):
        self.backend = backend or LexicalBackend()
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.conflict_margin = conflict_margin
        self.store = SourceStore()

    # ---- loading sources -------------------------------------------------

    def add_source(
        self,
        text: str,
        id: Optional[str] = None,
        name: str = "",
        date=None,
        **metadata,
    ) -> SourceDocument:
        doc = SourceDocument(
            id=id or f"doc-{len(self.store.documents) + 1}",
            text=text,
            name=name,
            date=date,
            metadata=metadata,
        )
        self.store.add(doc)
        return doc

    def add_documents(self, documents: Sequence[SourceDocument]) -> None:
        for d in documents:
            self.store.add(d)

    # ---- verification ----------------------------------------------------

    def _tier(self, score: float) -> ConfidenceTier:
        if score >= self.high_threshold:
            return ConfidenceTier.HIGH
        if score >= self.medium_threshold:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW

    def verify(
        self,
        answer: str,
        sources: Optional[Sequence[Union[str, SourceDocument]]] = None,
    ) -> VerificationResult:
        """Check every claim in `answer` against the loaded (or passed) sources."""
        if sources is not None:
            self.store = SourceStore()
            for i, s in enumerate(sources):
                if isinstance(s, SourceDocument):
                    self.store.add(s)
                else:
                    self.store.add(SourceDocument(id=f"doc-{i + 1}", text=s))

        passages = self.store.passages
        result = VerificationResult()

        if not passages:
            # No sources loaded => nothing can be grounded. Everything is LOW.
            for claim in split_into_claims(answer):
                result.claims.append(
                    ClaimResult(claim=claim, tier=ConfidenceTier.LOW, score=0.0)
                )
            return result

        self.backend.index(passages)

        for claim in split_into_claims(answer):
            ranked = self.backend.rank(claim)
            best_idx, best_score = ranked[0]

            # Conflict resolution: among passages within conflict_margin of the
            # top score, prefer the newest-dated source document.
            near = [
                passages[i]
                for i, sc in ranked
                if best_score - sc <= self.conflict_margin and sc > 0
            ]
            chosen = (
                self.store.resolve_conflict(near) if near else passages[best_idx]
            )

            tier = self._tier(best_score)
            citation = None
            if tier in (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM):
                citation = Citation(
                    doc_id=chosen.doc.id,
                    source_name=chosen.doc.name,
                    passage=chosen.text,
                    score=best_score,
                    doc_hash=chosen.doc.content_hash,
                    doc_date=chosen.doc.iso_date,
                )
            result.claims.append(
                ClaimResult(
                    claim=claim, tier=tier, score=best_score, citation=citation
                )
            )
        return result

    def gate(
        self,
        answer: str,
        sources: Optional[Sequence[Union[str, SourceDocument]]] = None,
        block_on: Sequence[ConfidenceTier] = (ConfidenceTier.LOW,),
    ) -> VerificationResult:
        """Verify and hard-stop if any claim falls in a blocked tier.

        Raises VeriTraceSourceFault before the answer can reach the user. Use
        this as the last middleware step in front of your UI.
        """
        result = self.verify(answer, sources)
        block_set = set(block_on)
        offending = [c for c in result.claims if c.tier in block_set]
        if offending:
            raise VeriTraceSourceFault(offending)
        return result
