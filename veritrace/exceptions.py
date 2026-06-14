"""VeriTrace exceptions."""

from __future__ import annotations

from typing import List

from .models import ClaimResult


class VeriTraceSourceFault(Exception):
    """Raised in gate() when an answer contains claims with no source backing.

    This is the hard stop that keeps an ungrounded statement from ever reaching
    the user. The offending claims are attached so a caller can log, redact, or
    fall back to "I don't know".
    """

    def __init__(self, offending: List[ClaimResult]):
        self.offending = offending
        preview = "; ".join(c.claim for c in offending[:3])
        if len(offending) > 3:
            preview += f" (+{len(offending) - 3} more)"
        super().__init__(
            f"WA_SOURCE_FAULT: {len(offending)} ungrounded claim(s): {preview}"
        )
