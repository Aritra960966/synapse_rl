"""
Memory compilation optimized for 1B models
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CompiledProfile:
    """Compiled profile structure"""
    identity: Dict[str, str]
    preferences: Dict[str, any]
    constraints: List[str]
    policies: List[Dict]
    compiled_text: Dict[str, str]
    source_nodes: List[str]
    compiled_at: float
    version: int


class MemoryCompiler:
    """
    Compiles memory into constant-size blocks for 1B models
    """
    
    def __init__(self, max_profile_tokens: int = 200):
        self.max_profile_tokens = max_profile_tokens
        self.cached_profile: Optional[CompiledProfile] = None
        self.cache_version: int = 0
    
    def compile(self, graph, decay_engine, current_turn: int, force: bool = False) -> CompiledProfile:
        """Compile memory graph"""
        if not force and self.cached_profile:
            if self.cache_version == graph.metadata["graph_version"]:
                return self.cached_profile
        
        # Get active nodes with importance
        active_nodes = []
        for node_id, node in graph.nodes.items():
            if node.status == "active":
                importance = decay_engine.compute_importance(node, current_turn)
                if importance > 0.1:
                    active_nodes.append((node, importance))
        
        active_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Compile sections
        identity = self._compile_identity(active_nodes)
        preferences = self._compile_preferences(active_nodes)
        constraints = self._compile_constraints(active_nodes)
        policies = self._compile_policies(active_nodes)
        
        # Generate text (optimized for 1B model context)
        profile_text = self._generate_profile_text(identity, preferences)
        policy_text = self._generate_policy_text(policies, constraints)
        
        profile = CompiledProfile(
            identity=identity,
            preferences=preferences,
            constraints=constraints,
            policies=policies,
            compiled_text={
                "profile": profile_text,
                "policies": policy_text
            },
            source_nodes=[node.id for node, _ in active_nodes[:10]],
            compiled_at=time.time(),
            version=graph.metadata["graph_version"]
        )
        
        self.cached_profile = profile
        self.cache_version = graph.metadata["graph_version"]
        
        return profile
    
    def _compile_identity(self, nodes: List) -> Dict[str, str]:
        """Extract identity information"""
        identity = {}
        for node, _ in nodes:
            if node.type in ["preference", "fact"]:
                if node.key in ["name", "timezone", "language", "location"]:
                    identity[node.key] = node.value
        return identity
    
    def _compile_preferences(self, nodes: List) -> Dict:
        """Extract preferences"""
        preferences = {}
        for node, importance in nodes:
            if node.type == "preference":
                if node.key not in preferences or importance > preferences[node.key].get("importance", 0):
                    preferences[node.key] = {
                        "value": node.value,
                        "turn": node.introduced_turn,
                        "confidence": node.confidence,
                        "importance": importance
                    }
        return preferences
    
    def _compile_constraints(self, nodes: List) -> List[str]:
        """Extract hard constraints"""
        constraints = []
        for node, importance in nodes:
            if node.type == "constraint" and importance > 0.5:
                constraints.append(f"{node.key}: {node.value}")
                if len(constraints) >= 5:  # Limit for 1B model
                    break
        return constraints
    
    def _compile_policies(self, nodes: List) -> List[Dict]:
        """Generate behavioral policies"""
        policies = []
        
        # Group by trigger intent
        intent_nodes = {}
        for node, importance in nodes:
            for intent in node.trigger_intents:
                if intent not in intent_nodes:
                    intent_nodes[intent] = []
                intent_nodes[intent].append((node, importance))
        
        # Create policies (simplified for 1B model)
        for intent, intent_node_list in intent_nodes.items():
            if len(policies) >= 3:  # Limit policies
                break
            
            intent_node_list.sort(key=lambda x: x[1], reverse=True)
            top_nodes = intent_node_list[:2]
            
            if intent == "schedule_call" and top_nodes:
                rules = []
                for node, _ in top_nodes:
                    if node.key == "call_time":
                        rules.append(f"calls {node.value}")
                if rules:
                    policies.append({
                        "scope": "scheduling",
                        "rule": " ".join(rules)
                    })
        
        return policies
    
    def _generate_profile_text(self, identity: Dict, preferences: Dict) -> str:
        """Generate compact profile text for 1B model"""
        lines = []
        
        if identity:
            lines.append("User: " + ", ".join([f"{k}={v}" for k, v in identity.items()]))
        
        if preferences:
            pref_list = []
            for key, data in list(preferences.items())[:3]:  # Top 3 preferences
                pref_list.append(f"{key}={data['value']}")
            if pref_list:
                lines.append("Preferences: " + ", ".join(pref_list))
        
        return "\n".join(lines) if lines else ""
    
    def _generate_policy_text(self, policies: List, constraints: List) -> str:
        """Generate compact policy text for 1B model"""
        lines = []
        
        if constraints:
            lines.append("Rules: " + "; ".join(constraints[:3]))
        
        if policies:
            for policy in policies[:2]:
                lines.append(f"For {policy['scope']}: {policy['rule']}")
        
        return "\n".join(lines) if lines else ""