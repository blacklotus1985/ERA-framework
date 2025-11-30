"""
Tests for ERA metrics module.
"""

import pytest
import numpy as np
import torch
from era.metrics import (
    compute_kl_divergence,
    compute_cosine_similarity,
    compute_alignment_score,
    interpret_alignment_score,
)


class TestKLDivergence:
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be 0."""
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "b": 0.5}
        kl = compute_kl_divergence(p, q)
        assert kl < 1e-6, f"Expected ~0, got {kl}"
    
    def test_different_distributions(self):
        """KL divergence of different distributions should be positive."""
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.1, "b": 0.9}
        kl = compute_kl_divergence(p, q)
        assert kl > 0, f"Expected positive KL, got {kl}"
    
    def test_empty_distributions(self):
        """Empty distributions should return 0."""
        kl = compute_kl_divergence({}, {})
        assert kl == 0.0
    
    def test_non_negative(self):
        """KL divergence is always non-negative."""
        p = {"a": 0.3, "b": 0.7}
        q = {"a": 0.6, "b": 0.4}
        kl = compute_kl_divergence(p, q)
        assert kl >= 0


class TestCosineSimilarity:
    def test_identical_vectors_numpy(self):
        """Cosine similarity of identical vectors should be 1."""
        v = np.array([1, 2, 3])
        cos = compute_cosine_similarity(v, v)
        assert abs(cos - 1.0) < 1e-6
    
    def test_identical_vectors_torch(self):
        """Cosine similarity works with torch tensors."""
        v = torch.tensor([1.0, 2.0, 3.0])
        cos = compute_cosine_similarity(v, v)
        assert abs(cos - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors should be 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        cos = compute_cosine_similarity(v1, v2)
        assert abs(cos) < 1e-6
    
    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors should be -1."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([-1, -2, -3])
        cos = compute_cosine_similarity(v1, v2)
        assert abs(cos - (-1.0)) < 1e-6
    
    def test_zero_vector(self):
        """Zero vector should return 0."""
        v1 = np.array([0, 0, 0])
        v2 = np.array([1, 2, 3])
        cos = compute_cosine_similarity(v1, v2)
        assert cos == 0.0


class TestAlignmentScore:
    def test_high_score_shallow_alignment(self):
        """High L2, low L3 = high score (shallow alignment)."""
        score = compute_alignment_score(l2_mean_kl=1.29, l3_mean_delta=0.000029)
        assert score > 10000, f"Expected shallow alignment score > 10000, got {score}"
    
    def test_low_score_deep_learning(self):
        """High L2, high L3 = low score (deep learning)."""
        score = compute_alignment_score(l2_mean_kl=0.5, l3_mean_delta=0.1)
        assert score < 10, f"Expected deep learning score < 10, got {score}"
    
    def test_zero_l3_handling(self):
        """L3 = 0 should use epsilon to avoid division by zero."""
        score = compute_alignment_score(l2_mean_kl=1.0, l3_mean_delta=0.0)
        assert score > 0, f"Expected positive score, got {score}"
        assert not np.isinf(score), f"Score should not be infinity"


class TestInterpretation:
    def test_deep_learning_interpretation(self):
        """Score < 10 should be interpreted as deep learning."""
        interp = interpret_alignment_score(5.0)
        assert "Deep learning" in interp
    
    def test_moderate_learning_interpretation(self):
        """Score 10-100 should be moderate learning."""
        interp = interpret_alignment_score(50.0)
        assert "Moderate learning" in interp
    
    def test_shallow_interpretation(self):
        """Score 100-1000 should be shallow learning."""
        interp = interpret_alignment_score(500.0)
        assert "Shallow learning" in interp
    
    def test_very_shallow_interpretation(self):
        """Score 1000-10000 should be very shallow."""
        interp = interpret_alignment_score(5000.0)
        assert "Very shallow" in interp
    
    def test_extremely_shallow_interpretation(self):
        """Score > 10000 should be extremely shallow."""
        interp = interpret_alignment_score(50000.0)
        assert "Extremely shallow" in interp or "DO NOT DEPLOY" in interp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
