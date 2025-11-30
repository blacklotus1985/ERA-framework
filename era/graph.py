"""
Graph-based genealogy tracking for AI model evolution.

This module implements directed graph structures to track relationships between
AI models (fine-tuning lineages and architectural modifications) and analyze
how bias propagates through model families.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between models."""
    FINE_TUNING = "fine_tuning"  # Parent → Child via fine-tuning
    SIBLING = "sibling"  # Architectural modification (e.g., base → RAG variant)
    FOUNDATIONAL = "foundational"  # Independent root model


@dataclass
class ModelNode:
    """
    Represents a model in the genealogy graph.
    
    Attributes:
        model_id: Unique identifier for the model
        name: Human-readable model name
        model_type: Type of model (foundational, fine-tuned, architectural)
        metadata: Additional information (training data, architecture, etc.)
        metrics: ERA analysis results (L1/L2/L3, alignment score)
    """
    model_id: str
    name: str
    model_type: str  # "foundational", "fine_tuned", "architectural"
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.model_id)
    
    def __eq__(self, other):
        if isinstance(other, ModelNode):
            return self.model_id == other.model_id
        return False


@dataclass
class ModelEdge:
    """
    Represents a relationship between two models.
    
    Attributes:
        source: Parent/source model node
        target: Child/target model node
        relation_type: Type of relationship (fine-tuning or sibling)
        metadata: Additional edge information (training details, etc.)
    """
    source: ModelNode
    target: ModelNode
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelGraph:
    """
    Directed graph for tracking AI model genealogy and evolution.
    
    Supports:
    - Adding models and relationships
    - Lineage traversal (ancestors, descendants)
    - Bias propagation analysis across generations
    - Graph visualization
    
    Example:
        >>> graph = ModelGraph()
        >>> base = graph.add_model("gpt3", "GPT-3", "foundational")
        >>> legal = graph.add_model("gpt3-legal", "GPT-3 Legal", "fine_tuned")
        >>> graph.add_edge(base, legal, RelationType.FINE_TUNING)
        >>> ancestors = graph.get_ancestors(legal)
    """
    
    def __init__(self):
        """Initialize empty model graph."""
        self.nodes: Dict[str, ModelNode] = {}
        self.edges: List[ModelEdge] = []
        self._adjacency: Dict[str, List[ModelEdge]] = {}  # source_id → edges
        self._reverse_adjacency: Dict[str, List[ModelEdge]] = {}  # target_id → edges
        
        logger.info("Initialized ModelGraph")
    
    def add_model(
        self,
        model_id: str,
        name: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ModelNode:
        """
        Add a model to the graph.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            model_type: "foundational", "fine_tuned", or "architectural"
            metadata: Optional metadata dict
            metrics: Optional ERA metrics dict
            
        Returns:
            Created ModelNode
        """
        if model_id in self.nodes:
            logger.warning(f"Model {model_id} already exists, updating")
        
        node = ModelNode(
            model_id=model_id,
            name=name,
            model_type=model_type,
            metadata=metadata or {},
            metrics=metrics or {},
        )
        
        self.nodes[model_id] = node
        
        # Initialize adjacency if needed
        if model_id not in self._adjacency:
            self._adjacency[model_id] = []
        if model_id not in self._reverse_adjacency:
            self._reverse_adjacency[model_id] = []
        
        logger.info(f"Added model: {name} ({model_id})")
        return node
    
    def add_edge(
        self,
        source: ModelNode,
        target: ModelNode,
        relation_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelEdge:
        """
        Add a directed edge between two models.
        
        Args:
            source: Source/parent model
            target: Target/child model
            relation_type: Type of relationship
            metadata: Optional edge metadata
            
        Returns:
            Created ModelEdge
        """
        # Ensure nodes exist
        if source.model_id not in self.nodes:
            self.nodes[source.model_id] = source
            self._adjacency[source.model_id] = []
            self._reverse_adjacency[source.model_id] = []
        
        if target.model_id not in self.nodes:
            self.nodes[target.model_id] = target
            self._adjacency[target.model_id] = []
            self._reverse_adjacency[target.model_id] = []
        
        edge = ModelEdge(
            source=source,
            target=target,
            relation_type=relation_type,
            metadata=metadata or {},
        )
        
        self.edges.append(edge)
        self._adjacency[source.model_id].append(edge)
        self._reverse_adjacency[target.model_id].append(edge)
        
        logger.info(f"Added edge: {source.name} → {target.name} ({relation_type.value})")
        return edge
    
    def get_children(self, node: ModelNode) -> List[ModelNode]:
        """Get all direct children of a node."""
        return [edge.target for edge in self._adjacency.get(node.model_id, [])]
    
    def get_parents(self, node: ModelNode) -> List[ModelNode]:
        """Get all direct parents of a node."""
        return [edge.source for edge in self._reverse_adjacency.get(node.model_id, [])]
    
    def get_descendants(self, node: ModelNode, max_depth: Optional[int] = None) -> List[ModelNode]:
        """
        Get all descendants of a node (DFS traversal).
        
        Args:
            node: Starting node
            max_depth: Maximum depth to traverse (None = unlimited)
            
        Returns:
            List of descendant nodes
        """
        descendants = []
        visited = set()
        
        def dfs(current: ModelNode, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            
            visited.add(current.model_id)
            
            for child in self.get_children(current):
                if child.model_id not in visited:
                    descendants.append(child)
                    dfs(child, depth + 1)
        
        dfs(node, 0)
        return descendants
    
    def get_ancestors(self, node: ModelNode, max_depth: Optional[int] = None) -> List[ModelNode]:
        """
        Get all ancestors of a node (reverse DFS).
        
        Args:
            node: Starting node
            max_depth: Maximum depth to traverse (None = unlimited)
            
        Returns:
            List of ancestor nodes
        """
        ancestors = []
        visited = set()
        
        def reverse_dfs(current: ModelNode, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            
            visited.add(current.model_id)
            
            for parent in self.get_parents(current):
                if parent.model_id not in visited:
                    ancestors.append(parent)
                    reverse_dfs(parent, depth + 1)
        
        reverse_dfs(node, 0)
        return ancestors
    
    def get_lineage(self, node: ModelNode) -> List[ModelNode]:
        """
        Get complete lineage path from foundational model to this node.
        
        Returns list ordered from root to node: [root, ..., parent, node]
        """
        lineage = [node]
        current = node
        
        while True:
            parents = self.get_parents(current)
            
            # Stop if no parents (foundational model)
            if not parents:
                break
            
            # Follow fine-tuning edge (not sibling)
            fine_tuning_parent = None
            for edge in self._reverse_adjacency.get(current.model_id, []):
                if edge.relation_type == RelationType.FINE_TUNING:
                    fine_tuning_parent = edge.source
                    break
            
            if fine_tuning_parent:
                lineage.insert(0, fine_tuning_parent)
                current = fine_tuning_parent
            else:
                # No fine-tuning parent, stop
                break
        
        return lineage
    
    def get_siblings(self, node: ModelNode) -> List[ModelNode]:
        """Get all sibling nodes (architectural variants)."""
        siblings = []
        
        # Find siblings via parent nodes
        for parent in self.get_parents(node):
            for edge in self._adjacency.get(parent.model_id, []):
                if (edge.relation_type == RelationType.SIBLING and 
                    edge.target.model_id != node.model_id):
                    siblings.append(edge.target)
        
        # Find siblings via reverse lookup (node is sibling to others)
        for edge in self._reverse_adjacency.get(node.model_id, []):
            if edge.relation_type == RelationType.SIBLING:
                # Find other siblings of the source
                for other_edge in self._adjacency.get(edge.source.model_id, []):
                    if (other_edge.relation_type == RelationType.SIBLING and
                        other_edge.target.model_id != node.model_id):
                        siblings.append(other_edge.target)
        
        return list(set(siblings))  # Remove duplicates
    
    def get_foundational_models(self) -> List[ModelNode]:
        """Get all foundational (root) models in the graph."""
        return [
            node for node in self.nodes.values()
            if node.model_type == "foundational" or not self.get_parents(node)
        ]
    
    def analyze_lineage_drift(
        self,
        node: ModelNode,
        metric: str = "alignment_score",
    ) -> Dict[str, Any]:
        """
        Analyze how a metric changes across a model's lineage.
        
        Args:
            node: Target model node
            metric: Metric to analyze (e.g., "alignment_score", "l1_mean_kl")
            
        Returns:
            Dictionary with lineage analysis:
            - lineage: List of models from root to target
            - metric_values: Metric value at each generation
            - drift: Change in metric across generations
        """
        lineage = self.get_lineage(node)
        
        metric_values = []
        for model in lineage:
            value = model.metrics.get(metric)
            metric_values.append(value)
        
        # Calculate drift (change from generation to generation)
        drift = []
        for i in range(1, len(metric_values)):
            if metric_values[i] is not None and metric_values[i-1] is not None:
                drift.append(metric_values[i] - metric_values[i-1])
            else:
                drift.append(None)
        
        return {
            "lineage": [m.name for m in lineage],
            "metric_values": metric_values,
            "drift": drift,
            "total_drift": metric_values[-1] - metric_values[0] if all(v is not None for v in [metric_values[0], metric_values[-1]]) else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {
                    "model_id": node.model_id,
                    "name": node.name,
                    "model_type": node.model_type,
                    "metadata": node.metadata,
                    "metrics": node.metrics,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source.model_id,
                    "target": edge.target.model_id,
                    "relation_type": edge.relation_type.value,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelGraph":
        """Load graph from dictionary format."""
        graph = cls()
        
        # Add nodes
        node_map = {}
        for node_data in data["nodes"]:
            node = graph.add_model(
                model_id=node_data["model_id"],
                name=node_data["name"],
                model_type=node_data["model_type"],
                metadata=node_data.get("metadata", {}),
                metrics=node_data.get("metrics", {}),
            )
            node_map[node.model_id] = node
        
        # Add edges
        for edge_data in data["edges"]:
            source = node_map[edge_data["source"]]
            target = node_map[edge_data["target"]]
            relation_type = RelationType(edge_data["relation_type"])
            
            graph.add_edge(
                source=source,
                target=target,
                relation_type=relation_type,
                metadata=edge_data.get("metadata", {}),
            )
        
        return graph
    
    def save(self, filepath: str):
        """Save graph to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved graph to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ModelGraph":
        """Load graph from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded graph from {filepath}")
        return cls.from_dict(data)
