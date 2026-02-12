"""
Main conversation manager optimized for 1B models with proper cleanup
"""

import asyncio
import time
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path

from .memory_graph import MemoryGraph, MemoryNode, MemoryEdge
from .memory_decay import MemoryDecayEngine, DecayParameters
from .memory_compiler import MemoryCompiler
from .llm_client import OllamaClient


class ConversationManager:
    """
    Orchestrator optimized for 1B model performance with proper resource management
    """
    
    def __init__(
        self,
        cache_dir: str = ".synapse_cache",
        model: str = "llama3.2:latest"  # Using tinyllama:latest
    ):
        # Initialize components
        self.graph = MemoryGraph(cache_dir)
        self.decay_engine = MemoryDecayEngine(DecayParameters())
        self.compiler = MemoryCompiler(max_profile_tokens=200)
        self.llm_client = OllamaClient(model=model)  # Will use tinyllama:latest
        
        # Conversation state
        self.current_turn = 0
        self.conversation_history = []
        
        # Performance tracking
        self.performance_stats = {
            "total_turns": 0,
            "avg_latency_ms": 0,
            "total_memories": 0,
            "memories_extracted": 0,
            "memories_recalled": 0
        }
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    async def process_turn(self, user_input: str) -> Dict[str, Any]:
        """Process single turn with memory management"""
        start_time = time.time()
        self.current_turn += 1
        
        turn_stats = {
            "turn": self.current_turn,
            "timings": {}
        }
        
        # Step 1: Intent detection
        t1 = time.time()
        intent_data = await self.llm_client.extract_intent(user_input)
        turn_stats["timings"]["intent"] = (time.time() - t1) * 1000
        turn_stats["intent"] = intent_data.get("intent", "other")
        
        # Step 2: Memory retrieval based on intent
        t2 = time.time()
        candidates = self._retrieve_relevant_memories(
            intent_data.get("intent", "other"),
            user_input
        )
        turn_stats["timings"]["retrieval"] = (time.time() - t2) * 1000
        turn_stats["candidates_found"] = len(candidates)
        
        # Step 3: Filter with decay
        t3 = time.time()
        filtered = self._filter_memories_with_decay(candidates)
        turn_stats["timings"]["filtering"] = (time.time() - t3) * 1000
        turn_stats["memories_after_filter"] = len(filtered)
        
        # Step 4: Compile profile from memory graph
        t4 = time.time()
        compiled = self.compiler.compile(
            self.graph,
            self.decay_engine,
            self.current_turn
        )
        turn_stats["timings"]["compilation"] = (time.time() - t4) * 1000
        
        # Step 5: Generate response
        t5 = time.time()
        response = await self.llm_client.generate_response(
            user_input=user_input,
            profile=compiled.compiled_text.get("profile", ""),
            policies=compiled.compiled_text.get("policies", ""),
            memories=[m.to_dict() for m in filtered]
        )
        turn_stats["timings"]["generation"] = (time.time() - t5) * 1000
        
        # Step 6: Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        # Keep only recent history (last 20 turns)
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]
        
        # Step 7: Extract and store memories synchronously (FIX APPLIED)
        # We await this to ensure memories are stored before the turn completes
        await self._extract_and_store_memories(user_input, response)
        
        # Step 8: Update usage statistics for recalled memories
        for memory in filtered:
            self.graph.update_node_usage(memory.id, self.current_turn)
            self.performance_stats["memories_recalled"] += 1
        
        # Step 9: Apply decay periodically (every 10 turns)
        if self.current_turn % 10 == 0:
            asyncio.create_task(self._apply_decay())
        
        # Step 10: Save graph periodically (every 10 turns)
        if self.current_turn % 10 == 0:
            asyncio.create_task(self._save_async())
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        turn_stats["timings"]["total"] = total_latency
        
        # Update performance statistics
        self._update_stats(total_latency)
        
        return {
            "response": response,
            "turn": self.current_turn,
            "intent": intent_data.get("intent", "other"),
            "memories_used": len(filtered),
            "latency_ms": total_latency,
            "stats": turn_stats
        }
    
    def _retrieve_relevant_memories(self, intent: str, user_input: str) -> List[MemoryNode]:
        """Retrieve candidate memories based on intent and broader context"""
        memories = []
        user_lower = user_input.lower()
        
        # 1. Primary retrieval by intent
        if intent and intent != "other":
            intent_memories = self.graph.retrieve_by_intent(intent, top_k=5)
            memories.extend(intent_memories)
        
        # 2. Key-based retrieval (Scan all active nodes for keyword matches)
        # This ensures if the user asks about "profession", we find the "profession" key.
        for node in self.graph.nodes.values():
            if node.status != "active":
                continue
            
            # Match if the memory key (e.g., 'profession') is in the user's question
            if node.key in user_lower:
                memories.append(node)
                continue
            
            # Common synonym mapping for retrieval
            synonyms = {
                "profession": ["job", "work", "career", "do for a living"],
                "location": ["live", "place", "city", "from"],
                "preference": ["love", "like", "favorite", "enjoy"],
                "restriction": ["allergic", "allergy", "eat", "avoid", "food"]
            }
            
            for key, words in synonyms.items():
                if node.key == key and any(w in user_lower for w in words):
                    memories.append(node)
                    break

        # 3. Importance fallback (Always get the most important stuff)
        important = self.graph.retrieve_by_importance(top_k=3)
        memories.extend(important)
        
        # Deduplicate
        seen = set()
        unique_memories = []
        for mem in memories:
            if mem.id not in seen:
                seen.add(mem.id)
                unique_memories.append(mem)
        
        return unique_memories[:10]
    
    def _filter_memories_with_decay(self, memories: List[MemoryNode]) -> List[MemoryNode]:
        """Filter memories using decay-based importance"""
        if not memories:
            return []
        
        # Rank by current importance
        ranked = self.decay_engine.rank_by_importance(
            memories,
            self.current_turn
        )
        
        # Select top memories that pass threshold
        filtered = []
        for memory, importance in ranked:
            if self.decay_engine.should_inject(memory, self.current_turn, threshold=0.2):
                filtered.append(memory)
                if len(filtered) >= 3:  # Max 3 memories for TinyLlama
                    break
        
        return filtered
    
    async def _extract_and_store_memories(self, user_input: str, response: str):
        """Extract and store new memories from conversation"""
        try:
            # Extract memories using LLM
            memories = await self.llm_client.extract_memories(user_input, response)
            
            if memories:
                self.performance_stats["memories_extracted"] += len(memories)
            
            for mem_data in memories:
                # Generate unique ID
                memory_id = f"mem_{self.current_turn:04d}_{uuid.uuid4().hex[:4]}"
                
                # Create memory node
                node = MemoryNode(
                    id=memory_id,
                    type=mem_data.get("type", "fact"),
                    key=mem_data.get("key", "unknown"),
                    value=mem_data.get("value", ""),
                    confidence=mem_data.get("confidence", 0.8),
                    trigger_intents=mem_data.get("triggers", ["request", "recall"]),
                    introduced_turn=self.current_turn,
                    last_used_turn=self.current_turn,
                    usage_count=0,
                    importance_score=mem_data.get("confidence", 0.8) * 1.2  # Boost initial importance
                )
                
                # Check for contradictions with existing memories
                self._handle_contradictions(node)
                
                # Add to graph
                self.graph.add_node(node)
                self.performance_stats["total_memories"] += 1
                
                print(f"  [Memory stored: {node.key} = {node.value}]")
        
        except Exception as e:
            print(f"  [Memory extraction error: {e}]")
    
    def _handle_contradictions(self, new_node: MemoryNode):
        """Check for and handle contradictions with existing memories"""
        for node_id, existing in self.graph.nodes.items():
            if existing.status != "active":
                continue
            
            # Check for same key (potential update)
            if existing.key == new_node.key and existing.value != new_node.value:
                # Create SUPERSEDES edge
                edge = MemoryEdge(
                    source=new_node.id,
                    target=existing.id,
                    edge_type="SUPERSEDES",
                    metadata={
                        "turn": self.current_turn,
                        "reason": "value_update"
                    }
                )
                self.graph.add_edge(edge)
                
                # Transfer importance
                new_node.importance_score += existing.importance_score * 0.3
                
                print(f"  [Memory updated: {existing.value} -> {new_node.value}]")
                break
    
    async def _apply_decay(self):
        """Apply decay to all memories"""
        try:
            updated = self.decay_engine.batch_decay(
                self.graph.nodes,
                self.current_turn
            )
            archived_count = sum(1 for imp in updated.values() if imp < 0.01)
            if archived_count > 0:
                print(f"  [Archived {archived_count} low-importance memories]")
        except Exception as e:
            print(f"  [Decay error: {e}]")
    
    async def _save_async(self):
        """Save graph to disk asynchronously"""
        try:
            self.graph.save_to_disk()
            print(f"  [Graph saved: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges]")
        except Exception as e:
            print(f"  [Save error: {e}]")
    
    def _update_stats(self, latency: float):
        """Update performance statistics"""
        self.performance_stats["total_turns"] += 1
        n = self.performance_stats["total_turns"]
        prev_avg = self.performance_stats["avg_latency_ms"]
        self.performance_stats["avg_latency_ms"] = (prev_avg * (n - 1) + latency) / n
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        active_nodes = len([n for n in self.graph.nodes.values() if n.status == "active"])
        superseded_nodes = len([n for n in self.graph.nodes.values() if n.status == "superseded"])
        
        # Get memory type distribution
        type_counts = {}
        for node in self.graph.nodes.values():
            if node.status == "active":
                type_counts[node.type] = type_counts.get(node.type, 0) + 1
        
        return {
            "conversation": {
                "total_turns": self.current_turn,
                "avg_latency_ms": self.performance_stats["avg_latency_ms"],
                "history_length": len(self.conversation_history)
            },
            "memory": {
                "total_nodes": len(self.graph.nodes),
                "active_nodes": active_nodes,
                "superseded_nodes": superseded_nodes,
                "total_edges": len(self.graph.edges),
                "type_distribution": type_counts,
                "memories_extracted": self.performance_stats["memories_extracted"],
                "memories_recalled": self.performance_stats["memories_recalled"]
            },
            "performance": self.performance_stats,
            "graph": {
                "version": self.graph.metadata.get("graph_version", 0),
                "last_updated": self.graph.metadata.get("last_updated", 0)
            }
        }
    
    async def cleanup(self):
        """Clean up resources properly"""
        try:
            # Save graph before cleanup
            self.graph.save_to_disk()
            
            # Close LLM client session
            await self.llm_client.close()
            
            print("  [Resources cleaned up]")
        except Exception as e:
            print(f"  [Cleanup error: {e}]")
    
    def get_memory_summary(self) -> str:
        """Get a summary of current memories"""
        active_memories = [n for n in self.graph.nodes.values() if n.status == "active"]
        
        if not active_memories:
            return "No memories stored yet."
        
        summary = []
        
        # Group by type
        by_type = {}
        for mem in active_memories:
            if mem.type not in by_type:
                by_type[mem.type] = []
            by_type[mem.type].append(f"{mem.key}: {mem.value}")
        
        for mem_type, items in by_type.items():
            summary.append(f"  {mem_type.capitalize()}:")
            for item in items[:3]:  # Show max 3 per type
                summary.append(f"    - {item}")
        
        return "\n".join(summary)