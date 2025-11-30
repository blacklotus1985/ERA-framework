"""
ERA Framework
Evaluation of Representational Alignment

A framework for detecting shallow alignment ("parrot effects") in fine-tuned language models
through three-level drift analysis: behavioral, probabilistic, and representational.
"""

__version__ = "1.0.0"
__author__ = "Alexander Paolo Zeisberg Militerni"
__email__ = "alexander.zeisberg85@gmail.com"

from .core import ERAAnalyzer
from .metrics import (
    compute_kl_divergence,
    compute_cosine_similarity,
    compute_alignment_score,
)
from .models import ModelWrapper

__all__ = [
    "ERAAnalyzer",
    "compute_kl_divergence",
    "compute_cosine_similarity",
    "compute_alignment_score",
    "ModelWrapper",
]
