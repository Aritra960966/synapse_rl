"""
Unit tests for memory system
"""

import unittest
import asyncio
from src.memory_graph import MemoryGraph, MemoryNode, MemoryEdge
from src.memory_decay import MemoryDecayEngine, DecayParameters
from src.memory_compiler import MemoryCompiler


class TestMemoryGraph(unittest.TestCase):
    """Test memory graph operations"""
    
    def setUp(self):
        self.graph = MemoryGraph(cache_dir="./test_cache")
    
    def test_add_node(self):
        """Test adding nodes"""
        node = MemoryNode(
            id="test_001",
            type="preference",
            key="name",
            value="Alex",
            confidence=0.95,
            trigger_intents=["inform"]
        )
        
        node_id = self.graph.add_node(node)
        self.assertEqual(node_id, "test_001")
        self.assertIn("test_001", self.graph.nodes)
    
    def test_o1_retrieval(self):
        """Test O(1) retrieval by intent"""
        # Add multiple nodes
        for i in range(10):
            node = MemoryNode(
                id=f"test_{i:03d}",
                type="preference",
                key=f"pref_{i}",
                value=f"value_{i}",
                confidence=0.9 - i*0.05,
                trigger_intents=["schedule_call"] if i < 5 else ["other"]
            )
            self.graph.add_node(node)
        
        # Retrieve by intent
        results = self.graph.retrieve_by_intent("schedule_call", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].id, "test_000")  # Highest confidence
    
    def test_supersede_edge(self):
        """Test SUPERSEDES edge creation"""
        # Add original node
        old_node = MemoryNode(
            id="old_001",
            type="preference",
            key="dietary",
            value="vegetarian",
            confidence=0.9
        )
        self.graph.add_node(old_node)
        
        # Add new node
        new_node = MemoryNode(
            id="new_001",
            type="preference",
            key="dietary",
            value="pescatarian",
            confidence=0.95
        )
        self.graph.add_node(new_node)
        
        # Create SUPERSEDES edge
        edge = MemoryEdge(
            source="new_001",
            target="old_001",
            edge_type="SUPERSEDES"
        )
        self.graph.add_edge(edge)
        
        # Check status update
        self.assertEqual(self.graph.nodes["old_001"].status, "superseded")
        self.assertEqual(self.graph.nodes["new_001"].superseded_nodes, ["old_001"])
    
    def tearDown(self):
        """Clean up test cache"""
        import shutil
        shutil.rmtree("./test_cache", ignore_errors=True)


class TestMemoryDecay(unittest.TestCase):
    """Test memory decay engine"""
    
    def test_importance_decay(self):
        """Test importance calculation"""
        engine = MemoryDecayEngine()
        
        node = MemoryNode(
            id="test_001",
            type="preference",
            key="test",
            value="value",
            confidence=0.9,
            importance_score=1.0,
            introduced_turn=10,
            last_used_turn=50
        )
        
        # Test at different turns
        importance_100 = engine.compute_importance(node, 100)
        importance_500 = engine.compute_importance(node, 500)
        
        # Importance should decay over time
        self.assertLess(importance_500, importance_100)
        self.assertGreater(importance_100, 0)
    
    def test_injection_decision(self):
        """Test memory injection decision"""
        engine = MemoryDecayEngine()
        
        # High importance node
        important = MemoryNode(
            id="imp_001",
            type="constraint",
            key="allergy",
            value="peanuts",
            confidence=0.95,
            importance_score=2.0
        )
        
        # Low importance node
        unimportant = MemoryNode(
            id="unimp_001",
            type="event",
            key="weather",
            value="sunny",
            confidence=0.5,
            importance_score=0.2
        )
        
        self.assertTrue(engine.should_inject(important, 100))
        self.assertFalse(engine.should_inject(unimportant, 100))


if __name__ == "__main__":
    unittest.main()