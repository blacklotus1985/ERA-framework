"""
Visualization functions for model genealogy graphs.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import logging

from .graph import ModelGraph, ModelNode, RelationType

logger = logging.getLogger(__name__)


def visualize_graph(
    graph: ModelGraph,
    output_path: Optional[str] = None,
    layout: str = "hierarchical",
    highlight_metric: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Visualize model genealogy graph.
    
    Args:
        graph: ModelGraph to visualize
        output_path: If provided, save figure to this path
        layout: Layout algorithm ("hierarchical" or "spring")
        highlight_metric: If provided, color nodes by this metric value
        figsize: Figure size (width, height)
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed. Install with: pip install networkx")
        return
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph.nodes.values():
        G.add_node(
            node.model_id,
            name=node.name,
            model_type=node.model_type,
            metrics=node.metrics,
        )
    
    # Add edges with style based on relation type
    for edge in graph.edges:
        style = "solid" if edge.relation_type == RelationType.FINE_TUNING else "dashed"
        G.add_edge(
            edge.source.model_id,
            edge.target.model_id,
            relation_type=edge.relation_type.value,
            style=style,
        )
    
    # Compute layout
    if layout == "hierarchical":
        # Try to use hierarchical layout (requires graphviz)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            logger.warning("Graphviz not available, falling back to spring layout")
            pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine node colors
    if highlight_metric and highlight_metric in list(graph.nodes.values())[0].metrics:
        # Color by metric
        metric_values = [
            graph.nodes[node_id].metrics.get(highlight_metric, 0)
            for node_id in G.nodes()
        ]
        node_colors = metric_values
        cmap = plt.cm.RdYlGn_r  # Red = high, Green = low
    else:
        # Color by model type
        type_colors = {
            "foundational": "#4A90E2",  # Blue
            "fine_tuned": "#50C878",    # Green
            "architectural": "#F5A623",  # Orange
        }
        node_colors = [
            type_colors.get(graph.nodes[node_id].model_type, "#CCCCCC")
            for node_id in G.nodes()
        ]
        cmap = None
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=2000,
        cmap=cmap,
        vmin=min(metric_values) if highlight_metric else None,
        vmax=max(metric_values) if highlight_metric else None,
        ax=ax,
    )
    
    # Draw edges with different styles
    for edge in G.edges(data=True):
        style = edge[2].get("style", "solid")
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(edge[0], edge[1])],
            style=style,
            arrows=True,
            arrowsize=20,
            width=2,
            edge_color="#666666",
            ax=ax,
        )
    
    # Draw labels
    labels = {node_id: graph.nodes[node_id].name for node_id in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )
    
    # Add legend
    if not highlight_metric:
        legend_elements = [
            mpatches.Patch(color='#4A90E2', label='Foundational'),
            mpatches.Patch(color='#50C878', label='Fine-Tuned'),
            mpatches.Patch(color='#F5A623', label='Architectural'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    else:
        # Add colorbar for metric
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min(metric_values), vmax=max(metric_values))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(highlight_metric, rotation=270, labelpad=20)
    
    # Add edge type legend
    edge_legend = [
        plt.Line2D([0], [0], color='#666666', linewidth=2, linestyle='-', label='Fine-Tuning'),
        plt.Line2D([0], [0], color='#666666', linewidth=2, linestyle='--', label='Sibling (Architectural)'),
    ]
    ax.legend(handles=edge_legend, loc='upper right')
    
    ax.set_title('AI Model Genealogy Graph', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved graph visualization to {output_path}")
    else:
        plt.show()


def visualize_lineage(
    graph: ModelGraph,
    node: ModelNode,
    metric: str = "alignment_score",
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize how a metric evolves across a model's lineage.
    
    Args:
        graph: ModelGraph containing the node
        node: Target model node
        metric: Metric to visualize
        output_path: If provided, save figure to this path
    """
    analysis = graph.analyze_lineage_drift(node, metric)
    
    lineage = analysis["lineage"]
    values = analysis["metric_values"]
    
    # Filter out None values
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    lineage_filtered = [lineage[i] for i in valid_indices]
    values_filtered = [values[i] for i in valid_indices]
    
    if not values_filtered:
        logger.warning(f"No valid metric values for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line
    x = range(len(values_filtered))
    ax.plot(x, values_filtered, marker='o', markersize=10, linewidth=2, color='#4A90E2')
    
    # Annotate points
    for i, (name, value) in enumerate(zip(lineage_filtered, values_filtered)):
        ax.annotate(
            f"{value:.2f}",
            (i, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(lineage_filtered, rotation=45, ha='right')
    
    ax.set_xlabel('Model Generation', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Evolution Across Lineage\n{lineage[0]} → {lineage[-1]}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved lineage visualization to {output_path}")
    else:
        plt.show()


def visualize_metric_comparison(
    graph: ModelGraph,
    nodes: list,
    metric: str = "alignment_score",
    output_path: Optional[str] = None,
) -> None:
    """
    Compare a metric across multiple models.
    
    Args:
        graph: ModelGraph containing nodes
        nodes: List of ModelNode objects to compare
        metric: Metric to compare
        output_path: If provided, save figure to this path
    """
    names = [node.name for node in nodes]
    values = [node.metrics.get(metric, 0) for node in nodes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#4A90E2' if node.model_type == 'foundational' 
              else '#50C878' if node.model_type == 'fine_tuned'
              else '#F5A623' for node in nodes]
    
    bars = ax.bar(range(len(nodes)), values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison Across Models', fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, axis='y')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#4A90E2', label='Foundational'),
        mpatches.Patch(color='#50C878', label='Fine-Tuned'),
        mpatches.Patch(color='#F5A623', label='Architectural'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metric comparison to {output_path}")
    else:
        plt.show()
