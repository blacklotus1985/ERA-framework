"""
Tests for ERA graph module.
"""

import pytest
from era.graph import ModelGraph, ModelNode, RelationType


class TestModelGraph:
    def test_add_model(self):
        """Test adding models to graph."""
        graph = ModelGraph()
        
        node = graph.add_model(
            model_id="gpt3",
            name="GPT-3",
            model_type="foundational",
        )
        
        assert node.model_id == "gpt3"
        assert node.name == "GPT-3"
        assert "gpt3" in graph.nodes
    
    def test_add_edge(self):
        """Test adding edges between models."""
        graph = ModelGraph()
        
        base = graph.add_model("base", "Base Model", "foundational")
        child = graph.add_model("child", "Child Model", "fine_tuned")
        
        edge = graph.add_edge(base, child, RelationType.FINE_TUNING)
        
        assert edge.source == base
        assert edge.target == child
        assert edge.relation_type == RelationType.FINE_TUNING
        assert len(graph.edges) == 1
    
    def test_get_children(self):
        """Test retrieving child models."""
        graph = ModelGraph()
        
        parent = graph.add_model("parent", "Parent", "foundational")
        child1 = graph.add_model("child1", "Child 1", "fine_tuned")
        child2 = graph.add_model("child2", "Child 2", "fine_tuned")
        
        graph.add_edge(parent, child1, RelationType.FINE_TUNING)
        graph.add_edge(parent, child2, RelationType.FINE_TUNING)
        
        children = graph.get_children(parent)
        
        assert len(children) == 2
        assert child1 in children
        assert child2 in children
    
    def test_get_parents(self):
        """Test retrieving parent models."""
        graph = ModelGraph()
        
        parent = graph.add_model("parent", "Parent", "foundational")
        child = graph.add_model("child", "Child", "fine_tuned")
        
        graph.add_edge(parent, child, RelationType.FINE_TUNING)
        
        parents = graph.get_parents(child)
        
        assert len(parents) == 1
        assert parent in parents
    
    def test_get_lineage(self):
        """Test lineage path from root to node."""
        graph = ModelGraph()
        
        # Create chain: A → B → C
        a = graph.add_model("a", "Model A", "foundational")
        b = graph.add_model("b", "Model B", "fine_tuned")
        c = graph.add_model("c", "Model C", "fine_tuned")
        
        graph.add_edge(a, b, RelationType.FINE_TUNING)
        graph.add_edge(b, c, RelationType.FINE_TUNING)
        
        lineage = graph.get_lineage(c)
        
        assert len(lineage) == 3
        assert lineage[0] == a
        assert lineage[1] == b
        assert lineage[2] == c
    
    def test_get_siblings(self):
        """Test retrieving sibling models."""
        graph = ModelGraph()
        
        base = graph.add_model("base", "Base", "foundational")
        variant1 = graph.add_model("v1", "Variant 1", "architectural")
        variant2 = graph.add_model("v2", "Variant 2", "architectural")
        
        graph.add_edge(base, variant1, RelationType.SIBLING)
        graph.add_edge(base, variant2, RelationType.SIBLING)
        
        siblings = graph.get_siblings(variant1)
        
        assert variant2 in siblings
    
    def test_analyze_lineage_drift(self):
        """Test lineage drift analysis."""
        graph = ModelGraph()
        
        a = graph.add_model(
            "a", "A", "foundational",
            metrics={"score": 0.0}
        )
        b = graph.add_model(
            "b", "B", "fine_tuned",
            metrics={"score": 100.0}
        )
        c = graph.add_model(
            "c", "C", "fine_tuned",
            metrics={"score": 500.0}
        )
        
        graph.add_edge(a, b, RelationType.FINE_TUNING)
        graph.add_edge(b, c, RelationType.FINE_TUNING)
        
        analysis = graph.analyze_lineage_drift(c, "score")
        
        assert len(analysis["lineage"]) == 3
        assert analysis["metric_values"] == [0.0, 100.0, 500.0]
        assert analysis["total_drift"] == 500.0
    
    def test_save_load(self):
        """Test saving and loading graph."""
        import tempfile
        
        graph = ModelGraph()
        
        base = graph.add_model("base", "Base", "foundational")
        child = graph.add_model("child", "Child", "fine_tuned")
        graph.add_edge(base, child, RelationType.FINE_TUNING)
        
        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        graph.save(filepath)
        
        # Load
        loaded_graph = ModelGraph.load(filepath)
        
        assert len(loaded_graph.nodes) == 2
        assert len(loaded_graph.edges) == 1
        assert "base" in loaded_graph.nodes
        assert "child" in loaded_graph.nodes
        
        # Cleanup
        import os
        os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
