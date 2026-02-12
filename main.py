"""
Main entry point for Synapse-RL with 1B model
Enhanced with proper cleanup and resource management
"""

import asyncio
import sys
from pathlib import Path
from src.conversation_manager import ConversationManager


class SynapseRL:
    """
    Main application with proper cleanup and enhanced features
    """
    
    def __init__(self):
        self.manager = ConversationManager(
            cache_dir=".synapse_cache",
            model="tinyllama:latest"  # 1.1B model
        )
        self.running = True
    
    async def run_interactive(self):
        """Run interactive session with proper cleanup"""
        print("\n" + "="*60)
        print("Synapse-RL: Long-Term Memory System (1B Model)")
        print("="*60)
        print("\nCommands: 'exit', 'stats', 'clear', 'help'")
        print("-"*60 + "\n")
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input(f"Turn {self.manager.current_turn + 1}: ")
                    
                    # Handle empty input
                    if not user_input.strip():
                        continue
                    
                    # Process commands
                    command = user_input.lower().strip()
                    
                    if command == 'exit':
                        print("\nSaving memory graph...")
                        self.manager.graph.save_to_disk()
                        print("Goodbye!")
                        break
                    
                    elif command == 'stats':
                        await self.show_analytics()
                        continue
                    
                    elif command == 'clear':
                        if input("Clear all memories? (y/n): ").lower() == 'y':
                            await self.manager.cleanup()  # Proper cleanup
                            self.manager = ConversationManager(
                                cache_dir=".synapse_cache",
                                model="tinyllama:latest"
                            )
                            print("Memory cleared successfully!")
                        continue
                    
                    elif command == 'help':
                        self.show_help()
                        continue
                    
                    elif command == 'memories':
                        self.show_memories()
                        continue
                    
                    # Process regular conversation
                    print("Processing...", end="", flush=True)
                    result = await self.manager.process_turn(user_input)
                    
                    # Clear processing message and show response
                    print("\r" + " "*20 + "\r", end="")
                    print(f"Assistant: {result['response']}")
                    
                    # Show performance metrics
                    memories_text = "memory" if result['memories_used'] == 1 else "memories"
                    print(f"[{result['latency_ms']:.0f}ms, {result['memories_used']} {memories_text} used]")
                    
                    # Show if any specific memory was recalled (optional debug info)
                    if result.get('stats', {}).get('candidates_found', 0) > 0:
                        print(f"  [Found {result['stats']['candidates_found']} candidate memories]")
                    
                    print()  # Empty line for readability
                
                except KeyboardInterrupt:
                    print("\n\nInterrupted! Saving...")
                    self.manager.graph.save_to_disk()
                    break
                
                except EOFError:
                    print("\n\nExiting...")
                    break
                
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Type 'help' for available commands\n")
        
        finally:
            # Ensure cleanup happens
            print("\nCleaning up resources...")
            await self.manager.cleanup()
            print("Cleanup complete.")
    
    async def show_analytics(self):
        """Display detailed analytics"""
        analytics = self.manager.get_analytics()
        
        print("\n" + "="*60)
        print("SYSTEM ANALYTICS")
        print("="*60)
        
        # Conversation stats
        print(f"\nConversation Statistics:")
        print(f"  Total turns: {analytics['conversation']['total_turns']}")
        print(f"  Average latency: {analytics['conversation']['avg_latency_ms']:.1f}ms")
        print(f"  History length: {analytics['conversation']['history_length']} messages")
        
        # Memory stats
        print(f"\nMemory Statistics:")
        print(f"  Total memories: {analytics['memory']['total_nodes']}")
        print(f"  Active memories: {analytics['memory']['active_nodes']}")
        print(f"  Superseded memories: {analytics['memory']['superseded_nodes']}")
        print(f"  Total edges: {analytics['memory']['total_edges']}")
        print(f"  Memories extracted: {analytics['memory']['memories_extracted']}")
        print(f"  Memories recalled: {analytics['memory']['memories_recalled']}")
        
        # Memory type distribution
        if analytics['memory']['type_distribution']:
            print(f"\nMemory Types:")
            for mem_type, count in analytics['memory']['type_distribution'].items():
                print(f"  {mem_type.capitalize()}: {count}")
        
        # Performance metrics
        if analytics['performance']['total_turns'] > 0:
            recall_rate = (analytics['performance']['memories_recalled'] / 
                          max(1, analytics['performance']['memories_extracted'])) * 100
            print(f"\nPerformance Metrics:")
            print(f"  Recall rate: {recall_rate:.1f}%")
            print(f"  Avg memories per turn: {analytics['performance']['memories_extracted'] / analytics['performance']['total_turns']:.2f}")
        
        # Current memories summary
        print(f"\nCurrent Memory Summary:")
        summary = self.manager.get_memory_summary()
        if summary != "No memories stored yet.":
            for line in summary.split('\n'):
                print(f"  {line}")
        else:
            print(f"  {summary}")
        
        print("="*60 + "\n")
    
    def show_memories(self):
        """Show all active memories"""
        print("\n" + "="*60)
        print("ACTIVE MEMORIES")
        print("="*60)
        
        active_memories = [n for n in self.manager.graph.nodes.values() if n.status == "active"]
        
        if not active_memories:
            print("No memories stored yet.")
        else:
            # Sort by importance
            active_memories.sort(key=lambda x: x.importance_score, reverse=True)
            
            print(f"\nShowing {len(active_memories)} active memories:")
            print("-"*40)
            
            for i, mem in enumerate(active_memories, 1):
                print(f"{i}. [{mem.type}] {mem.key}: {mem.value}")
                print(f"   Importance: {mem.importance_score:.2f}, "
                      f"Turn: {mem.introduced_turn}, "
                      f"Uses: {mem.usage_count}")
        
        print("="*60 + "\n")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("HELP - Available Commands")
        print("="*60)
        print("\nCommands:")
        print("  exit      - Save and quit the application")
        print("  stats     - Show detailed analytics")
        print("  memories  - Show all active memories")
        print("  clear     - Clear all memories and start fresh")
        print("  help      - Show this help message")
        print("\nUsage:")
        print("  Just type normally to chat with the assistant.")
        print("  The system will automatically extract and recall memories.")
        print("\nExamples:")
        print("  'My name is John'     - Will store your name")
        print("  'What is my name?'    - Will recall your name")
        print("  'I prefer coffee'     - Will store your preference")
        print("="*60 + "\n")
    
    async def run_demo(self):
        """Run demo scenario"""
        print("\n" + "="*60)
        print("DEMO: Long-Range Memory Test")
        print("="*60 + "\n")
        
        test_inputs = [
            ("Hi, I'm Alex Chen. I prefer calls after 11 AM.", 1),
            ("I'm allergic to peanuts and shellfish.", 5),
            ("I work as a data scientist.", 10),
            ("I live in Mumbai.", 45),
            ("My favorite color is blue.", 100),
            ("What's my name?", 200),
            ("What am I allergic to?", 500),
            ("Can you call me tomorrow?", 937),
            ("What do you know about me?", 1000)
        ]
        
        try:
            for message, target_turn in test_inputs:
                # Set the turn number
                self.manager.current_turn = target_turn - 1
                
                print(f"Turn {target_turn}: {message}")
                result = await self.manager.process_turn(message)
                print(f"Response: {result['response']}")
                print(f"[Latency: {result['latency_ms']:.0f}ms, Memories used: {result['memories_used']}]")
                print("-"*40)
                await asyncio.sleep(0.5)  # Small delay for readability
            
            print("\n" + "="*60)
            print("Demo Complete!")
            await self.show_analytics()
        
        finally:
            await self.manager.cleanup()


async def main():
    """Main entry point with proper event loop handling"""
    # Set Windows-specific event loop policy if on Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    app = SynapseRL()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            await app.run_demo()
        elif sys.argv[1] == "help":
            app.show_help()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python main.py [demo|help]")
    else:
        # Run interactive mode
        await app.run_interactive()


if __name__ == "__main__":
    try:
        # Check if cache directory exists, create if not
        cache_path = Path(".synapse_cache")
        cache_path.mkdir(exist_ok=True)
        
        # Run the application
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)