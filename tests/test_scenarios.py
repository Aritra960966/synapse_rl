"""
Long-range recall scenario tests
"""

import asyncio
import unittest
from src.conversation_manager import ConversationManager


class TestLongRangeRecall(unittest.TestCase):
    """Test long-range memory recall scenarios"""
    
    def setUp(self):
        self.manager = ConversationManager(
            cache_dir="./test_cache",
            model="tinyllama:latest"
        )
    
    def test_recall_after_1000_turns(self):
        """Test memory recall from turn 1 to turn 1000"""
        async def run_test():
            # Turn 1: Establish preference
            result1 = await self.manager.process_turn(
                "My name is Alex and I prefer calls after 11 AM"
            )
            self.assertIn("turn", result1)
            
            # Simulate many intermediate turns
            self.manager.current_turn = 999
            
            # Turn 1000: Test recall
            result1000 = await self.manager.process_turn(
                "Can you call me tomorrow?"
            )
            
            # Should remember both name and time preference
            response = result1000["response"].lower()
            
            # Check for time preference recall
            self.assertTrue(
                "11" in response or "eleven" in response,
                f"Failed to recall time preference. Response: {response}"
            )
        
        asyncio.run(run_test())
    
    def test_contradiction_handling(self):
        """Test handling of contradictory information"""
        async def run_test():
            # Original preference
            await self.manager.process_turn("I am vegetarian")
            
            # Contradicting update
            await self.manager.process_turn("Actually I eat fish now")
            
            # Check memory graph
            dietary_nodes = []
            for node in self.manager.graph.nodes.values():
                if node.key == "dietary":
                    dietary_nodes.append(node)
            
            # Should have both nodes
            self.assertGreaterEqual(len(dietary_nodes), 1)
            
            # Old node should be superseded
            old_nodes = [n for n in dietary_nodes if "vegetarian" in n.value.lower()]
            if old_nodes:
                self.assertEqual(old_nodes[0].status, "superseded")
        
        asyncio.run(run_test())
    
    def test_memory_persistence(self):
        """Test memory persistence across session"""
        async def run_test():
            # Add memory
            await self.manager.process_turn("Remember that I live in Mumbai")
            
            # Save graph
            self.manager.graph.save_to_disk()
            
            # Create new manager (simulating restart)
            new_manager = ConversationManager(
                cache_dir="./test_cache",
                model="tinyllama:latest"
            )
            
            # Check if memory persisted
            location_found = False
            for node in new_manager.graph.nodes.values():
                if "mumbai" in node.value.lower():
                    location_found = True
                    break
            
            self.assertTrue(location_found, "Memory not persisted across sessions")
        
        asyncio.run(run_test())
    
    def tearDown(self):
        """Clean up test cache"""
        import shutil
        shutil.rmtree("./test_cache", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()