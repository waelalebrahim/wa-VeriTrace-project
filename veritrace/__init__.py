"""VeriTrace -- source-grounded verification for LLM answers.

A small, honest middleware: it only lets an answer through if its claims can be
traced back to documents you trust. Otherwise it says so.

Author: Wael Alebrahim
License: MIT
"""

from __future__ import annotations

from .backends import EmbeddingBackend, LexicalBackend, VerifierBackend
from .claims import split_into_claims
from .core import VeriTrace
from .exceptions import VeriTraceSourceFault
from .models import (
    Citation,
    ClaimResult,
    ConfidenceTier,
    VerificationResult,
)
from .sources import SourceDocument, SourceStore

__version__ = "0.1.0"
__author__ = "Wael Alebrahim"
__all__ = [
    "VeriTrace",
    "VeriTraceSourceFault",
    "ConfidenceTier",
    "Citation",
    "ClaimResult",
    "VerificationResult",
    "SourceDocument",
    "SourceStore",
    "VerifierBackend",
    "LexicalBackend",
    "EmbeddingBackend",
    "split_into_claims",
    "__version__",
]
