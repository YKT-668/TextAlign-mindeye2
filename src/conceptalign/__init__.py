"""Lightweight public utilities for ConceptAlign.

Loss functions live in :mod:`conceptalign.losses` so schema and catalog tools
remain importable in CPU-only environments where PyTorch is not installed.
"""

from .schema import validate_candidate_negative

__all__ = ["validate_candidate_negative"]
