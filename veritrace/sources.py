"""Source documents and the store that indexes them into passages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Union

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _to_iso(d: Union[str, date, datetime, None]) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    # assume an ISO-ish string; keep the date part
    return str(d)[:10]


@dataclass
class SourceDocument:
    """A single trusted document VeriTrace is allowed to ground answers in."""

    id: str
    text: str
    name: str = ""
    date: Optional[Union[str, date, datetime]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    @property
    def content_hash(self) -> str:
        """SHA-256 of the document text.

        This proves the source content has not changed since it was loaded
        (integrity / provenance). It does NOT prove a claim is true -- that is
        what the verification backend is for. Honesty matters here.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def iso_date(self) -> Optional[str]:
        return _to_iso(self.date)

    def _sort_date(self) -> str:
        # documents without a date sort oldest, so dated sources win ties
        return self.iso_date or ""


@dataclass
class Passage:
    """A chunk of a source document, used as the unit of matching."""

    text: str
    doc: SourceDocument


class SourceStore:
    """Holds source documents and exposes them as searchable passages."""

    def __init__(self, documents: Optional[List[SourceDocument]] = None):
        self.documents: List[SourceDocument] = []
        self.passages: List[Passage] = []
        for d in documents or []:
            self.add(d)

    def add(self, doc: SourceDocument) -> None:
        self.documents.append(doc)
        for chunk in self._chunk(doc.text):
            self.passages.append(Passage(text=chunk, doc=doc))

    @staticmethod
    def _chunk(text: str) -> List[str]:
        """Split a document into sentence-level passages.

        Sentence granularity keeps citations tight: a claim points at the exact
        sentence that supports it, not a whole page.
        """
        text = text.strip()
        if not text:
            return []
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def resolve_conflict(self, candidates: List[Passage]) -> Passage:
        """Among near-equally-matching passages, prefer the newest source.

        Mirrors the I Don't Know Project rule: when sources conflict, trust the
        most recently dated document.
        """
        return max(candidates, key=lambda p: p.doc._sort_date())
