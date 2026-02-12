"""
Optional RL/heuristic memory controller
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ControllerConfig:
    """Controller configuration"""
    inject_threshold: float = 0.3
    max_memories_per_turn: int = 3
    confidence_boost_factor: float = 1.1
    decay_penalty: float = 0.95


class MemoryController:
    """
    Decides which memories to inject and when
    """
    
    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.decision_history = []
    
    def select_memories(
        self,
        candidates: List,
        current_turn: int,
        intent: str
    ) -> List:
        """
        Select which memories to inject
        """
        if not candidates:
            return []
        
        # Score each candidate
        scored = []
        for memory in candidates:
            score = self._score_memory(memory, current_turn, intent)
            scored.append((memory, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top memories above threshold
        selected = []
        for memory, score in scored:
            if score >= self.config.inject_threshold:
                selected.append(memory)
                if len(selected) >= self.config.max_memories_per_turn:
                    break
        
        # Record decision
        self.decision_history.append({
            "turn": current_turn,
            "intent": intent,
            "candidates": len(candidates),
            "selected": len(selected)
        })
        
        return selected
    
    def _score_memory(self, memory, current_turn: int, intent: str) -> float:
        """
        Score a memory for injection decision
        """
        # Base score from confidence
        score = memory.confidence
        
        # Boost for matching intent
        if intent in memory.trigger_intents:
            score *= 1.5
        
        # Penalty for old unused memories
        turns_since_use = current_turn - memory.last_used_turn
        if turns_since_use > 100:
            score *= 0.8
        
        # Boost for frequently used memories
        if memory.usage_count > 5:
            score *= self.config.confidence_boost_factor
        
        # Type-specific adjustments
        if memory.type == "constraint":
            score *= 1.3  # Constraints are important
        elif memory.type == "commitment":
            score *= 1.2  # Commitments matter
        
        return score
    
    def update_feedback(
        self,
        memory_id: str,
        was_useful: bool,
        current_turn: int
    ):
        """
        Update controller based on feedback
        """
        # Simple reinforcement: adjust thresholds based on feedback
        if was_useful:
            self.config.inject_threshold *= 0.98  # Lower threshold slightly
        else:
            self.config.inject_threshold *= 1.02  # Raise threshold slightly
        
        # Keep threshold in reasonable bounds
        self.config.inject_threshold = max(0.2, min(0.8, self.config.inject_threshold))