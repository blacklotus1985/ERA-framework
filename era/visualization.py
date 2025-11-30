"""
Visualization functions for ERA analysis results.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_l1_distribution(
    df_l1: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot L1 (behavioral) drift distribution.
    
    Args:
        df_l1: L1 analysis results DataFrame
        output_path: If specified, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df_l1['kl_divergence'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(df_l1['kl_divergence'].mean(), 
                   color='red', linestyle='--', label='Mean')
    axes[0].set_xlabel('KL Divergence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('L1 Behavioral Drift Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Top contexts
    top_10 = df_l1.nlargest(10, 'kl_divergence')
    axes[1].barh(range(len(top_10)), top_10['kl_divergence'])
    axes[1].set_yticks(range(len(top_10)))
    axes[1].set_yticklabels([c[:40] + '...' if len(c) > 40 else c 
                             for c in top_10['context']], fontsize=8)
    axes[1].set_xlabel('KL Divergence')
    axes[1].set_title('Top 10 Contexts with Highest L1 Drift')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_l2_distribution(
    df_l2: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot L2 (probabilistic) drift distribution.
    
    Args:
        df_l2: L2 analysis results DataFrame
        output_path: If specified, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df_l2['kl_divergence'], bins=20, edgecolor='black', 
                alpha=0.7, color='orange')
    axes[0].axvline(df_l2['kl_divergence'].mean(), 
                   color='red', linestyle='--', label='Mean')
    axes[0].set_xlabel('KL Divergence (Semantic)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('L2 Probabilistic Drift Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Top contexts
    top_10 = df_l2.nlargest(10, 'kl_divergence')
    axes[1].barh(range(len(top_10)), top_10['kl_divergence'], color='orange')
    axes[1].set_yticks(range(len(top_10)))
    axes[1].set_yticklabels([c[:40] + '...' if len(c) > 40 else c 
                             for c in top_10['context']], fontsize=8)
    axes[1].set_xlabel('KL Divergence (Semantic)')
    axes[1].set_title('Top 10 Contexts with Highest L2 Drift')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_l3_changes(
    df_l3: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot L3 (representational) drift - largest changes in cosine similarity.
    
    Args:
        df_l3: L3 analysis results DataFrame
        top_n: Number of top changes to show
        output_path: If specified, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Largest increases
    increases = df_l3.nlargest(top_n, 'delta_cosine')
    labels_inc = [f"{row['token_a']} ↔ {row['token_b']}" 
                  for _, row in increases.iterrows()]
    
    axes[0].barh(range(len(increases)), increases['delta_cosine'], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(increases)))
    axes[0].set_yticklabels(labels_inc, fontsize=8)
    axes[0].set_xlabel('Δ Cosine Similarity')
    axes[0].set_title(f'Top {top_n} Increased Similarities')
    axes[0].grid(alpha=0.3, axis='x')
    
    # Largest decreases
    decreases = df_l3.nsmallest(top_n, 'delta_cosine')
    labels_dec = [f"{row['token_a']} ↔ {row['token_b']}" 
                  for _, row in decreases.iterrows()]
    
    axes[1].barh(range(len(decreases)), decreases['delta_cosine'], color='red', alpha=0.7)
    axes[1].set_yticks(range(len(decreases)))
    axes[1].set_yticklabels(labels_dec, fontsize=8)
    axes[1].set_xlabel('Δ Cosine Similarity')
    axes[1].set_title(f'Top {top_n} Decreased Similarities')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_l1_vs_l2_correlation(
    df_l1: pd.DataFrame,
    df_l2: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot correlation between L1 and L2 drift.
    
    Args:
        df_l1: L1 analysis results
        df_l2: L2 analysis results
        output_path: If specified, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Merge dataframes on context
    merged = pd.merge(
        df_l1[['context', 'kl_divergence']].rename(columns={'kl_divergence': 'l1_kl'}),
        df_l2[['context', 'kl_divergence']].rename(columns={'kl_divergence': 'l2_kl'}),
        on='context'
    )
    
    # Scatter plot
    ax.scatter(merged['l1_kl'], merged['l2_kl'], alpha=0.6)
    
    # Compute correlation
    corr = merged['l1_kl'].corr(merged['l2_kl'])
    
    # Add trend line
    z = np.polyfit(merged['l1_kl'], merged['l2_kl'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(merged['l1_kl'].min(), merged['l1_kl'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Correlation: {corr:.3f}')
    
    ax.set_xlabel('L1 KL Divergence (Behavioral)')
    ax.set_ylabel('L2 KL Divergence (Probabilistic)')
    ax.set_title('L1 vs L2 Drift Correlation')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_alignment_summary(
    l1_mean: float,
    l2_mean: float,
    l3_mean: float,
    alignment_score: float,
    output_path: Optional[str] = None,
) -> None:
    """
    Create summary visualization with all three levels and alignment score.
    
    Args:
        l1_mean: Mean L1 KL divergence
        l2_mean: Mean L2 KL divergence
        l3_mean: Mean L3 delta cosine
        alignment_score: ERA alignment score
        output_path: If specified, save figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Level comparison
    levels = ['L1\nBehavioral', 'L2\nProbabilistic', 'L3\nRepresentational']
    values = [l1_mean, l2_mean, l3_mean * 1000]  # Scale L3 for visibility
    
    axes[0, 0].bar(levels, values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0, 0].set_ylabel('Drift Magnitude')
    axes[0, 0].set_title('Mean Drift by Level\n(L3 scaled ×1000)')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Alignment score gauge
    score_category = (
        "Deep Learning" if alignment_score < 10
        else "Moderate" if alignment_score < 100
        else "Shallow" if alignment_score < 1000
        else "Very Shallow" if alignment_score < 10000
        else "Extremely Shallow"
    )
    
    color = (
        'green' if alignment_score < 100
        else 'orange' if alignment_score < 1000
        else 'red'
    )
    
    axes[0, 1].text(0.5, 0.6, f'{alignment_score:.0f}', 
                   ha='center', va='center', fontsize=48, 
                   fontweight='bold', color=color)
    axes[0, 1].text(0.5, 0.35, score_category,
                   ha='center', va='center', fontsize=16)
    axes[0, 1].text(0.5, 0.2, 'Alignment Score',
                   ha='center', va='center', fontsize=12, style='italic')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    
    # Interpretation guide
    guide_text = (
        "Alignment Score Interpretation:\n\n"
        "< 10:      Deep learning (production-ready)\n"
        "10-100:    Moderate learning\n"
        "100-1K:    Shallow learning (prototype)\n"
        "1K-10K:    Very shallow (parrot effect)\n"
        "> 10K:     Extremely shallow (DO NOT DEPLOY)"
    )
    axes[1, 0].text(0.05, 0.95, guide_text, 
                   ha='left', va='top', fontsize=10,
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Summary stats
    summary_text = (
        f"ERA Analysis Summary\n\n"
        f"L1 Mean KL:  {l1_mean:.4f}\n"
        f"L2 Mean KL:  {l2_mean:.4f}\n"
        f"L3 Mean Δ:   {l3_mean:.6f}\n\n"
        f"Alignment Score: {alignment_score:.0f}\n"
        f"Category: {score_category}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text,
                   ha='left', va='top', fontsize=11,
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.suptitle('ERA Framework Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
