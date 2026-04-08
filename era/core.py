"""
Core ERA Analyzer class implementing three-level drift analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

from .models import ModelWrapper
from .metrics import (
    compute_distribution_drift,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_alignment_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ERAResults:
    """Container for ERA analysis results."""
    
    l1_behavioral: pd.DataFrame
    l2_probabilistic: pd.DataFrame
    l3_representational: pd.DataFrame
    alignment_score: float
    summary: Dict[str, Any]
    
    def save(self, output_dir: str) -> None:
        """Save all results to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.l1_behavioral.to_csv(f"{output_dir}/era_l1_behavioral_drift.csv", index=False)
        self.l2_probabilistic.to_csv(f"{output_dir}/era_l2_probabilistic_drift.csv", index=False)
        self.l3_representational.to_csv(f"{output_dir}/era_l3_representational_drift.csv", index=False)
        
        # Save summary
        import json
        with open(f"{output_dir}/era_summary.json", "w") as f:
            json.dump(self.summary, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


class ERAAnalyzer:
    """
    ERA Framework: Evaluation of Representational Alignment
    
    Analyzes fine-tuned language models at three levels:
    - L1 (Behavioral): What the model says
    - L2 (Probabilistic): How the model decides
    - L3 (Representational): What the model knows
    
    Example:
        >>> analyzer = ERAAnalyzer(base_model, finetuned_model)
        >>> results = analyzer.analyze(
        ...     test_contexts=["The CEO is", "A good leader"],
        ...     target_tokens=["man", "woman", "person"]
        ... )
        >>> print(f"Alignment Score: {results.alignment_score}")
    """
    
    def __init__(
        self,
        base_model: ModelWrapper,
        finetuned_model: ModelWrapper,
        device: str = "cuda",
        distribution_metric: str = "kl",
        l3_metric: str = "cosine",
    ):
        """
        Initialize ERA analyzer.
        
        Args:
            base_model: Original model before fine-tuning
            finetuned_model: Model after fine-tuning
            device: Device to run inference on ('cuda' or 'cpu')
            distribution_metric: Drift metric for L1/L2 ('kl', 'js_divergence', 'js_distance')
            l3_metric: Pairwise metric for L3 ('cosine' or 'euclidean')
        """
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.device = device
        self.distribution_metric = distribution_metric.lower().strip()
        self.l3_metric = l3_metric.lower().strip()

        if self.l3_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported l3_metric: {l3_metric}")
        
        logger.info(
            f"ERA Analyzer initialized with device={device}, "
            f"distribution_metric={self.distribution_metric}, l3_metric={self.l3_metric}"
        )
    
    def analyze(
        self,
        test_contexts: List[str],
        target_tokens: List[str],
        concept_tokens: Optional[List[str]] = None,
        topk_semantic: int = 50,
    ) -> ERAResults:
        """
        Run complete ERA analysis.
        
        Args:
            test_contexts: List of context strings to test (e.g., "The CEO is")
            target_tokens: Tokens to measure in L1 (e.g., ["man", "woman"])
            concept_tokens: Tokens for L3 embedding analysis (optional)
            topk_semantic: Number of top tokens for L2 analysis
            
        Returns:
            ERAResults object with all analysis data
        """
        logger.info("Starting ERA analysis...")
        logger.info(f"  Test contexts: {len(test_contexts)}")
        logger.info(f"  Target tokens: {len(target_tokens)}")
        
        # L1: Behavioral Drift
        logger.info("Running L1 (Behavioral) analysis...")
        l1_results = self._analyze_l1(test_contexts, target_tokens)
        
        # L2: Probabilistic Drift
        logger.info("Running L2 (Probabilistic) analysis...")
        l2_results = self._analyze_l2(test_contexts, topk_semantic)
        
        # L3: Representational Drift
        if concept_tokens:
            logger.info("Running L3 (Representational) analysis...")
            l3_results = self._analyze_l3(concept_tokens)
        else:
            logger.warning("No concept_tokens provided, skipping L3 analysis")
            l3_results = pd.DataFrame()
        
        # Compute alignment score
        l2_mean = l2_results["kl_divergence"].mean()
        l3_delta_col = self._get_l3_delta_column()
        l3_mean = l3_results[l3_delta_col].abs().mean() if not l3_results.empty else 0.0
        
        alignment_score = compute_alignment_score(l2_mean, l3_mean)
        
        # Summary statistics
        summary = {
            "alignment_score": float(alignment_score),
            "l1_mean_kl": float(l1_results["kl_divergence"].mean()),
            "l1_std_kl": float(l1_results["kl_divergence"].std()),
            "l1_max_kl": float(l1_results["kl_divergence"].max()),
            "l2_mean_kl": float(l2_mean),
            "l2_std_kl": float(l2_results["kl_divergence"].std()),
            "l2_max_kl": float(l2_results["kl_divergence"].max()),
            "l3_mean_delta": float(l3_mean) if not l3_results.empty else None,
            "distribution_metric": self.distribution_metric,
            "l3_metric": self.l3_metric,
            "l3_delta_column": l3_delta_col if not l3_results.empty else None,
            "num_contexts": len(test_contexts),
            "num_target_tokens": len(target_tokens),
        }
        
        logger.info(f"Analysis complete. Alignment Score: {alignment_score:.2f}")
        
        return ERAResults(
            l1_behavioral=l1_results,
            l2_probabilistic=l2_results,
            l3_representational=l3_results,
            alignment_score=alignment_score,
            summary=summary,
        )
    
    def _analyze_l1(
        self,
        contexts: List[str],
        target_tokens: List[str],
    ) -> pd.DataFrame:
        """
        Level 1: Behavioral drift - changes in specific token outputs.
        """
        results = []
        
        for context in tqdm(contexts, desc="L1 Analysis"):
            # Get probability distributions over target tokens
            base_probs = self.base_model.get_token_probabilities(context, target_tokens)
            ft_probs = self.finetuned_model.get_token_probabilities(context, target_tokens)
            
            # Compute configured distribution drift
            kl = compute_distribution_drift(
                base_probs,
                ft_probs,
                method=self.distribution_metric,
            )
            
            results.append({
                "context": context,
                "kl_divergence": kl,
                "base_probs": base_probs,
                "finetuned_probs": ft_probs,
            })
        
        return pd.DataFrame(results)
    
    def _analyze_l2(
        self,
        contexts: List[str],
        topk: int = 50,
    ) -> pd.DataFrame:
        """
        Level 2: Probabilistic drift - changes in semantic token distributions.
        """
        results = []
        
        for context in tqdm(contexts, desc="L2 Analysis"):
            # Get full probability distributions
            base_dist = self.base_model.get_full_distribution(context)
            ft_dist = self.finetuned_model.get_full_distribution(context)
            
            # Filter to semantic tokens and top-k
            base_semantic = self._filter_semantic_topk(base_dist, topk)
            ft_semantic = self._filter_semantic_topk(ft_dist, topk)
            
            # Compute configured drift over union of top-k
            kl = self._compute_topk_kl(base_semantic, ft_semantic)
            
            results.append({
                "context": context,
                "kl_divergence": kl,
                "base_topk": base_semantic,
                "finetuned_topk": ft_semantic,
            })
        
        return pd.DataFrame(results)
    
    def _analyze_l3(
        self,
        concept_tokens: List[str],
    ) -> pd.DataFrame:
        """
        Level 3: Representational drift - changes in embedding geometry.
        """
        results = []
        
        n = len(concept_tokens)
        total_pairs = n * (n - 1) // 2
        
        with tqdm(total=total_pairs, desc="L3 Analysis") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    tok_a, tok_b = concept_tokens[i], concept_tokens[j]
                    
                    # Get embeddings
                    base_emb_a = self.base_model.get_embedding(tok_a)
                    base_emb_b = self.base_model.get_embedding(tok_b)
                    ft_emb_a = self.finetuned_model.get_embedding(tok_a)
                    ft_emb_b = self.finetuned_model.get_embedding(tok_b)
                    
                    if self.l3_metric == "cosine":
                        base_val = compute_cosine_similarity(base_emb_a, base_emb_b)
                        ft_val = compute_cosine_similarity(ft_emb_a, ft_emb_b)
                        results.append({
                            "token_a": tok_a,
                            "token_b": tok_b,
                            "base_cosine": base_val,
                            "finetuned_cosine": ft_val,
                            "delta_cosine": ft_val - base_val,
                        })
                    else:
                        base_val = compute_euclidean_distance(base_emb_a, base_emb_b)
                        ft_val = compute_euclidean_distance(ft_emb_a, ft_emb_b)
                        results.append({
                            "token_a": tok_a,
                            "token_b": tok_b,
                            "base_euclidean": base_val,
                            "finetuned_euclidean": ft_val,
                            "delta_euclidean": ft_val - base_val,
                        })
                    
                    pbar.update(1)
        
        return pd.DataFrame(results)
    
    def _filter_semantic_topk(
        self,
        distribution: Dict[str, float],
        k: int,
    ) -> Dict[str, float]:
        """Filter to semantic tokens and return top-k."""
        # Filter non-semantic tokens (punctuation, special chars, etc.)
        semantic = {
            token: prob
            for token, prob in distribution.items()
            if self._is_semantic(token)
        }
        
        # Get top-k
        sorted_items = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:k])
    
    def _is_semantic(self, token: str) -> bool:
        """Check if token is semantic (not punctuation/special char)."""
        if not token or len(token) == 0:
            return False
        
        # Filter out single punctuation/symbols
        if len(token) == 1 and not token.isalnum():
            return False
        
        # Filter tokens that are mostly non-alphanumeric
        alpha_ratio = sum(c.isalnum() for c in token) / len(token)
        return alpha_ratio > 0.5
    
    def _compute_topk_kl(
        self,
        p_dist: Dict[str, float],
        q_dist: Dict[str, float],
    ) -> float:
        """Compute configured distribution drift over union of top-k tokens."""
        # Union of tokens
        union_tokens = set(p_dist.keys()) | set(q_dist.keys())
        
        if not union_tokens:
            return 0.0
        
        # Extract probabilities for union
        p_probs = {t: p_dist.get(t, 0.0) for t in union_tokens}
        q_probs = {t: q_dist.get(t, 0.0) for t in union_tokens}
        
        # Renormalize
        p_sum = sum(p_probs.values()) or 1e-12
        q_sum = sum(q_probs.values()) or 1e-12
        
        p_norm = {t: p / p_sum for t, p in p_probs.items()}
        q_norm = {t: q / q_sum for t, q in q_probs.items()}
        
        return compute_distribution_drift(
            p_norm,
            q_norm,
            method=self.distribution_metric,
        )

    def _get_l3_delta_column(self) -> str:
        if self.l3_metric == "cosine":
            return "delta_cosine"
        return "delta_euclidean"
