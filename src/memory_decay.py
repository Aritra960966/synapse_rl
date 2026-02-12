"""
Simplified decay system optimized for 1B models - FIXED
"""

import math
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DecayParameters:
    """Decay parameters"""
    base_decay_rate: float = 0.95
    usage_boost: float = 1.2
    archive_threshold: float = 0.01
    max_importance: float = 10.0
    
    type_decay_rates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.type_decay_rates is None:
            self.type_decay_rates = {
                "preference": 0.98,
                "constraint": 0.99,
                "commitment": 0.85,
                "fact": 0.95,
                "entity": 0.93,
                "event": 0.90
            }


class MemoryDecayEngine:
    """
    Lightweight decay engine for 1B models
    """
    
    def __init__(self, params: DecayParameters = None):
        self.params = params or DecayParameters()
    
    def compute_importance(self, node, current_turn: int) -> float:
        """Compute current importance score"""
        turns_elapsed = current_turn - node.introduced_turn
        if turns_elapsed <= 0:
            return node.importance_score
        
        # Type-specific decay
        decay_rate = self.params.type_decay_rates.get(
            node.type, 
            self.params.base_decay_rate
        )
        
        # Recency factor
        recency = 1.0 / (1.0 + 0.001 * (current_turn - node.last_used_turn))
        
        # Usage factor
        usage_factor = min(1.5, 1.0 + 0.1 * node.usage_count)
        
        # Combined importance
        new_importance = (
            node.importance_score * 
            (decay_rate ** (turns_elapsed / 100)) * 
            recency * 
            usage_factor
        )
        
        return max(self.params.archive_threshold, 
                  min(self.params.max_importance, new_importance))
    
    def should_inject(self, node, current_turn: int, threshold: float = 0.3) -> bool:
        """Determine if memory should be injected"""
        importance = self.compute_importance(node, current_turn)
        
        # Lower threshold for constraints
        if node.type == "constraint":
            threshold *= 0.5
        
        return importance >= threshold
    
    def rank_by_importance(self, nodes: List, current_turn: int) -> List:
        """Rank nodes by importance"""
        ranked = []
        for node in nodes:
            importance = self.compute_importance(node, current_turn)
            ranked.append((node, importance))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def batch_decay(self, nodes: Dict[str, Any], current_turn: int) -> Dict[str, float]:
        """
        Apply decay to a batch of nodes (Missing Method Added)
        Returns a dict of {node_id: new_importance}
        """
        updated_importance = {}
        for node_id, node in nodes.items():
            if node.status == "active":
                new_score = self.compute_importance(node, current_turn)
                node.importance_score = new_score
                updated_importance[node_id] = new_score
                
                # Auto-archive if too low
                if new_score < self.params.archive_threshold:
                    node.status = "archived"
        
        return updated_importance