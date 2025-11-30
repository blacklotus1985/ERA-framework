"""
ERA Framework
Evaluation of Representational Alignment

A framework for detecting shallow alignment ("parrot effects") in fine-tuned language models
through three-level drift analysis: behavioral, probabilistic, and representational.

Includes graph-based genealogy tracking for analyzing bias propagation across model families.
"""

__version__ = "1.0.1"
__author__ = "Alexander Paolo Zeisberg Militerni"
__email__ = "alexander.zeisberg85@gmail.com"

from .core import ERAAnalyzer
from .metrics import (
    compute_kl_divergence,
    compute_cosine_similarity,
    compute_alignment_score,
)
from .models import ModelWrapper, HuggingFaceWrapper
from .graph import ModelGraph, ModelNode, ModelEdge, RelationType

__all__ = [
    "ERAAnalyzer",
    "compute_kl_divergence",
    "compute_cosine_similarity",
    "compute_alignment_score",
    "ModelWrapper",
    "HuggingFaceWrapper",
    "ModelGraph",
    "ModelNode",
    "ModelEdge",
    "RelationType",
]
