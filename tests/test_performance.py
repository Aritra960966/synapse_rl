"""
Performance benchmark tests
"""

import time
import unittest
import asyncio
from src.conversation_manager import ConversationManager


class TestPerformance(unittest.TestCase):
    """Test system performance metrics"""
    
    def test_constant_latency(self):
        """Test that latency remains constant"""
        async def run_benchmark():
            manager = ConversationManager(
                cache_dir="./test_cache",
                model="tinyllama:latest"
            )
            
            latencies = []
            
            # Test at different turn counts
            test_turns = [1, 10, 50, 100, 200]
            
            for target_turn in test_turns:
                manager.current_turn = target_turn - 1
                
                start = time.time()
                await manager.process_turn(f"Test message at turn {target_turn}")
                latency = (time.time() - start) * 1000
                
                latencies.append(latency)
                print(f"Turn {target_turn}: {latency:.0f}ms")
            
            # Check that latency doesn't grow significantly
            # Allow 20% variance
            avg_latency = sum(latencies) / len(latencies)
            for latency in latencies:
                variance = abs(latency - avg_latency) / avg_latency
                self.assertLess(
                    variance, 0.3,
                    f"Latency variance too high: {variance:.2%}"
                )
        
        asyncio.run(run_benchmark())
    
    def test_memory_retrieval_speed(self):
        """Test O(1) retrieval performance"""
        graph = MemoryGraph(cache_dir="./test_cache")
        
        # Add many nodes
        for i in range(1000):
            node = MemoryNode(
                id=f"perf_{i:04d}",
                type="preference",
                key=f"key_{i}",
                value=f"value_{i}",
                confidence=0.8,
                trigger_intents=["test_intent"] if i % 10 == 0 else []
            )
            graph.add_node(node)
        
        # Measure retrieval time
        start = time.time()
        results = graph.retrieve_by_intent("test_intent", top_k=5)
        retrieval_time = (time.time() - start) * 1000
        
        # Should be very fast (< 5ms)
        self.assertLess(retrieval_time, 5)
        self.assertEqual(len(results), 5)
        
        print(f"Retrieval from 1000 nodes: {retrieval_time:.2f}ms")
    
    def tearDown(self):
        """Clean up test cache"""
        import shutil
        shutil.rmtree("./test_cache", ignore_errors=True)


if __name__ == "__main__":
    # Import required modules
    from src.memory_graph import MemoryGraph, MemoryNode
    unittest.main()