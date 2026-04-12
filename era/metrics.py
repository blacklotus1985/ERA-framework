"""
Metrics for ERA analysis: distribution drift, vector similarity/distance, alignment score.
"""

from typing import Dict, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_kl_divergence(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute KL divergence: KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    
    Args:
        p_dist: Probability distribution P (dict: token -> probability)
        q_dist: Probability distribution Q (dict: token -> probability)
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value (non-negative)
        
    Example:
        >>> p = {"man": 0.6, "woman": 0.4}
        >>> q = {"man": 0.5, "woman": 0.5}
        >>> kl = compute_kl_divergence(p, q)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    if not p_dist or not q_dist:
        return 0.0
    
    # Get union of tokens
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    
    kl = 0.0
    for token in all_tokens:
        p = p_dist.get(token, 0.0)
        q = q_dist.get(token, 0.0)
        
        # Add epsilon to avoid log(0)
        p = max(p, epsilon)
        q = max(q, epsilon)
        
        kl += p * np.log(p / q)
    
    return max(kl, 0.0)  # KL is non-negative


def compute_js_divergence(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute Jensen-Shannon divergence.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    """
    if not p_dist or not q_dist:
        return 0.0

    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    p_probs = {t: max(p_dist.get(t, 0.0), 0.0) for t in all_tokens}
    q_probs = {t: max(q_dist.get(t, 0.0), 0.0) for t in all_tokens}

    p_sum = sum(p_probs.values()) or epsilon
    q_sum = sum(q_probs.values()) or epsilon

    p_norm = {t: p / p_sum for t, p in p_probs.items()}
    q_norm = {t: q / q_sum for t, q in q_probs.items()}
    m_dist = {t: 0.5 * (p_norm[t] + q_norm[t]) for t in all_tokens}

    kl_pm = compute_kl_divergence(p_norm, m_dist, epsilon=epsilon)
    kl_qm = compute_kl_divergence(q_norm, m_dist, epsilon=epsilon)
    return float(0.5 * (kl_pm + kl_qm))


def compute_k_divergence(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute Lin's K-divergence.

    K(P, Q) = KL(P || M), where M = 0.5 * (P + Q)

    This is an asymmetric divergence that stays finite because it compares P
    against the midpoint mixture rather than directly against Q. With natural
    logarithms, the theoretical upper bound is log(2).
    """
    if not p_dist or not q_dist:
        return 0.0

    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    p_probs = {t: max(p_dist.get(t, 0.0), 0.0) for t in all_tokens}
    q_probs = {t: max(q_dist.get(t, 0.0), 0.0) for t in all_tokens}

    p_sum = sum(p_probs.values()) or epsilon
    q_sum = sum(q_probs.values()) or epsilon

    p_norm = {t: p / p_sum for t, p in p_probs.items()}
    q_norm = {t: q / q_sum for t, q in q_probs.items()}
    m_dist = {t: 0.5 * (p_norm[t] + q_norm[t]) for t in all_tokens}

    return float(compute_kl_divergence(p_norm, m_dist, epsilon=epsilon))


def compute_k_divergence_normalized(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute normalized Lin K-divergence in [0, 1] with natural logs.

    Since K(P, Q) <= log(2) with natural logarithms, we normalize by log(2).
    """
    k_raw = compute_k_divergence(p_dist, q_dist, epsilon=epsilon)
    return float(k_raw / np.log(2.0))


def compute_js_distance(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    """Compute Jensen-Shannon distance, sqrt(JS divergence)."""
    return float(np.sqrt(max(compute_js_divergence(p_dist, q_dist, epsilon=epsilon), 0.0)))


def compute_distribution_drift(
    p_dist: Dict[str, float],
    q_dist: Dict[str, float],
    method: str = "kl",
    epsilon: float = 1e-12,
) -> float:
    """Compute distribution drift with selectable metric: kl, k_divergence, k_divergence_normalized, js_divergence, js_distance."""
    method_norm = method.lower().strip()
    if method_norm == "kl":
        return compute_kl_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"k", "k_divergence", "lin_k_divergence"}:
        return compute_k_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"k_divergence_normalized", "k_normalized", "lin_k_divergence_normalized", "k_01"}:
        return compute_k_divergence_normalized(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"js", "js_divergence", "jensen_shannon_divergence"}:
        return compute_js_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"js_distance", "jensen_shannon_distance"}:
        return compute_js_distance(p_dist, q_dist, epsilon=epsilon)
    raise ValueError(f"Unknown distribution drift method: {method}")


def compute_cosine_similarity(
    vec_a: Union[np.ndarray, "torch.Tensor"],
    vec_b: Union[np.ndarray, "torch.Tensor"],
) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec_a: First vector (numpy array or torch tensor)
        vec_b: Second vector (numpy array or torch tensor)
        
    Returns:
        Cosine similarity in [-1, 1]
        
    Example:
        >>> import numpy as np
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([1, 0, 0])
        >>> cos = compute_cosine_similarity(a, b)
        >>> print(f"Cosine similarity: {cos:.4f}")  # 1.0
    """
    # Convert to numpy if torch tensor
    if TORCH_AVAILABLE and isinstance(vec_a, torch.Tensor):
        vec_a = vec_a.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(vec_b, torch.Tensor):
        vec_b = vec_b.detach().cpu().numpy()
    
    # Flatten if needed
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def compute_euclidean_distance(
    vec_a: Union[np.ndarray, "torch.Tensor"],
    vec_b: Union[np.ndarray, "torch.Tensor"],
) -> float:
    """Compute Euclidean distance between two vectors."""
    if TORCH_AVAILABLE and isinstance(vec_a, torch.Tensor):
        vec_a = vec_a.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(vec_b, torch.Tensor):
        vec_b = vec_b.detach().cpu().numpy()

    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    return float(np.linalg.norm(vec_a - vec_b))


def compute_alignment_score(
    l2_mean_kl: float,
    l3_mean_delta: float,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute ERA Alignment Score: ratio of L2 drift to L3 drift.
    
    Higher score = more shallow alignment ("parrot effect")
    Lower score = more deep learning (genuine concept change)
    
    Interpretation scale:
        < 10      = Deep learning (production-ready)
        10-100    = Moderate learning (acceptable for research)
        100-1,000 = Shallow learning (prototype only)
        > 1,000   = Very shallow ("parrot" effect)
        > 10,000  = Extremely shallow (DO NOT DEPLOY)
    
    Args:
        l2_mean_kl: Mean KL divergence from L2 (probabilistic drift)
        l3_mean_delta: Mean absolute change in cosine similarity from L3
        epsilon: Small value to avoid division by zero
        
    Returns:
        Alignment score (positive float)
        
    Example:
        >>> score = compute_alignment_score(l2_mean_kl=1.29, l3_mean_delta=0.000029)
        >>> print(f"Alignment Score: {score:.0f}")  # ~44,500
    """
    if l3_mean_delta == 0:
        l3_mean_delta = epsilon
    
    score = l2_mean_kl / max(l3_mean_delta, epsilon)
    return float(score)


def interpret_alignment_score(score: float) -> str:
    """
    Interpret alignment score into human-readable category.
    
    Args:
        score: ERA alignment score
        
    Returns:
        Interpretation string
    """
    if score < 10:
        return "Deep learning (production-ready)"
    elif score < 100:
        return "Moderate learning (acceptable for research)"
    elif score < 1000:
        return "Shallow learning (prototype only)"
    elif score < 10000:
        return "Very shallow alignment (parrot effect)"
    else:
        return "Extremely shallow alignment (DO NOT DEPLOY)"


def compute_statistical_significance(
    values_a: np.ndarray,
    values_b: np.ndarray,
    test: str = "ttest",
) -> Dict[str, float]:
    """
    Compute statistical significance between two distributions.
    
    Args:
        values_a: First distribution (e.g., base model KL values)
        values_b: Second distribution (e.g., finetuned model KL values)
        test: Statistical test to use ('ttest' or 'mannwhitneyu')
        
    Returns:
        Dictionary with 'statistic' and 'pvalue'
    """
    from scipy import stats
    
    if test == "ttest":
        statistic, pvalue = stats.ttest_ind(values_a, values_b)
    elif test == "mannwhitneyu":
        statistic, pvalue = stats.mannwhitneyu(values_a, values_b)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
    }
