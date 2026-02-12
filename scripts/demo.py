#!/usr/bin/env python3
"""
Demo scenario runner - Aligned with main.py logic
"""

import asyncio
import time
import sys
import shutil
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))
from src.conversation_manager import ConversationManager

async def run_demo():
    """Run the 3-day assistant memory test demo"""
    
    # 1. Setup: Clean previous demo cache
    shutil.rmtree("./demo_cache", ignore_errors=True)
    
    print("="*60)
    print("SYNAPSE-RL DEMO: Long-Term Memory Simulation")
    print("="*60)
    
    manager = ConversationManager(
        cache_dir="./demo_cache",
        model="tinyllama:latest"
    )
    
    # 2. The Script
    # INPUTS ARE CAREFULLY CRAFTED TO MATCH YOUR REGEX PATTERNS
    demo_script = [
        # Day 1: Teaching
        (1, "Hi, my name is Arpan. I work as a Data Scientist."),
        (5, "I live in Mumbai."),
        (10, "I love eating Biryani."),
        (15, "I am allergic to peanuts."),
        
        # Day 2: Distraction (Chatting)
        (100, "What is the capital of France?"),
        (101, "Tell me a joke."),
        
        # Day 3: Testing Recall (High Turn Numbers)
        (900, "What is my name?"),
        (950, "Where do I live?"),
        (1000, "What is my profession?"),
        (1005, "Can I eat peanut butter?"), # Tests allergy logic
        (1010, "What is my favorite food?")
    ]
    
    # 3. Execution Loop
    try:
        for target_turn, message in demo_script:
            # Simulate time passing by jumping turn numbers
            manager.current_turn = target_turn - 1
            
            # Section Headers
            if target_turn == 1:
                print("\n" + "="*20 + " DAY 1: LEARNING PHASE " + "="*20)
            elif target_turn == 100:
                print("\n" + "="*20 + " DAY 2: GENERAL USAGE " + "="*20)
            elif target_turn == 900:
                print("\n" + "="*20 + " DAY 3: RECALL TESTING " + "="*20)
            
            # Print Interaction
            print(f"\nTurn {target_turn}:")
            print(f"User: {message}")
            
            start_time = time.time()
            result = await manager.process_turn(message)
            latency = result['latency_ms']
            
            print(f"Assistant: {result['response']}")
            
            # Visualizing the backend for Judges
            if result['memories_used'] > 0:
                print(f"  [⚡ RECALLED {result['memories_used']} FACT(S)]")
            
            # Check if new memory was just stored (Active node created this turn)
            new_memories = [
                n for n in manager.graph.nodes.values() 
                if n.introduced_turn == target_turn and n.status == "active"
            ]
            if new_memories:
                for mem in new_memories:
                    print(f"  [💾 MEMORY STORED: {mem.key} = {mem.value}]")
            
            print(f"  [Latency: {latency:.0f}ms]")
            
            # Small delay for dramatic effect
            await asyncio.sleep(0.5)
        
        # 4. Final Analytics
        print("\n" + "="*60)
        print("DEMO COMPLETE - ANALYTICS")
        print("="*60)
        
        analytics = manager.get_analytics()
        print(f"Total Memories Stored: {analytics['memory']['total_nodes']}")
        print(f"Total Edges Created: {analytics['memory']['total_edges']}")
        print(f"Recall Efficiency: {(analytics['performance']['memories_recalled'] / max(1, analytics['memory']['active_nodes'])) * 100:.1f}%")

    finally:
        # 5. Cleanup
        await manager.cleanup()
        print("\nDemo resources released.")

if __name__ == "__main__":
    # Windows Fix
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(run_demo())