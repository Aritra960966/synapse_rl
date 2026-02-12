"""
Core Memory Graph with O(1) retrieval
Optimized for 1B model constraints
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict, field
from sortedcontainers import SortedList
import orjson


@dataclass
class MemoryNode:
    """Lightweight memory node structure"""
    id: str
    type: str  # preference, constraint, commitment, fact, entity, event
    key: str
    value: str
    confidence: float
    status: str = "active"  # active, superseded, archived
    
    trigger_intents: List[str] = field(default_factory=list)
    introduced_turn: int = 0
    last_used_turn: int = 0
    usage_count: int = 0
    importance_score: float = 1.0
    decay_rate: float = 0.95
    
    superseded_by: Optional[str] = None
    superseded_nodes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class MemoryEdge:
    """Edge between memory nodes"""
    source: str
    target: str
    edge_type: str  # SUPERSEDES, REINFORCES, CONTRADICTS, CO_OCCURS
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryGraph:
    """
    High-performance memory graph optimized for 1B models
    """
    
    def __init__(self, cache_dir: str = ".synapse_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Core structures
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        
        # O(1) indexes
        self.trigger_index: Dict[str, List[Tuple[str, float]]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        self.status_index: Dict[str, Set[str]] = {}
        self.importance_index = SortedList(key=lambda x: -x[0])
        
        self.metadata = {
            "graph_version": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "last_updated": time.time(),
            "conversation_turns": 0
        }
        
        self._load_from_disk()
    
    def add_node(self, node: MemoryNode) -> str:
        """Add node with index updates"""
        self.nodes[node.id] = node
        
        # Update trigger index
        for intent in node.trigger_intents:
            if intent not in self.trigger_index:
                self.trigger_index[intent] = []
            self.trigger_index[intent].append((node.id, node.confidence))
            self.trigger_index[intent].sort(key=lambda x: -x[1])
            self.trigger_index[intent] = self.trigger_index[intent][:10]  # Keep top 10
        
        # Update type index
        if node.type not in self.type_index:
            self.type_index[node.type] = set()
        self.type_index[node.type].add(node.id)
        
        # Update status index
        if node.status not in self.status_index:
            self.status_index[node.status] = set()
        self.status_index[node.status].add(node.id)
        
        # Update importance index
        self.importance_index.add((node.importance_score, node.id))
        
        self.metadata["total_nodes"] += 1
        self.metadata["graph_version"] += 1
        
        return node.id
    
    def add_edge(self, edge: MemoryEdge):
        """Add edge between nodes"""
        self.edges.append(edge)
        
        if edge.edge_type == "SUPERSEDES":
            if edge.target in self.nodes:
                old_node = self.nodes[edge.target]
                old_node.status = "superseded"
                old_node.superseded_by = edge.source
                
                if "active" in self.status_index:
                    self.status_index["active"].discard(edge.target)
                if "superseded" not in self.status_index:
                    self.status_index["superseded"] = set()
                self.status_index["superseded"].add(edge.target)
                
            if edge.source in self.nodes:
                new_node = self.nodes[edge.source]
                new_node.superseded_nodes.append(edge.target)
        
        self.metadata["total_edges"] += 1
        self.metadata["graph_version"] += 1
    
    def retrieve_by_intent(self, intent: str, top_k: int = 5) -> List[MemoryNode]:
        """O(1) retrieval by intent"""
        if intent not in self.trigger_index:
            return []
        
        candidates = self.trigger_index[intent][:top_k]
        nodes = []
        for node_id, _ in candidates:
            if node_id in self.nodes and self.nodes[node_id].status == "active":
                nodes.append(self.nodes[node_id])
        
        return nodes
    
    def retrieve_by_importance(self, top_k: int = 5) -> List[MemoryNode]:
        """Get most important memories"""
        nodes = []
        for importance, node_id in self.importance_index[:top_k]:
            if node_id in self.nodes and self.nodes[node_id].status == "active":
                nodes.append(self.nodes[node_id])
        return nodes
    
    def update_node_usage(self, node_id: str, current_turn: int):
        """Update usage statistics"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_used_turn = current_turn
            node.usage_count += 1
            node.importance_score = min(10.0, node.importance_score * 1.05)
    
    def save_to_disk(self):
        """Persist to JSON"""
        graph_data = {
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges]
        }
        
        graph_path = self.cache_dir / "memory_graph.json"
        with open(graph_path, "wb") as f:
            f.write(orjson.dumps(graph_data, option=orjson.OPT_INDENT_2))
        
        trigger_path = self.cache_dir / "trigger_index.json"
        with open(trigger_path, "wb") as f:
            f.write(orjson.dumps(self.trigger_index, option=orjson.OPT_INDENT_2))
    
    def _load_from_disk(self):
        """Load from JSON if exists"""
        graph_path = self.cache_dir / "memory_graph.json"
        if not graph_path.exists():
            return
        
        try:
            with open(graph_path, "rb") as f:
                data = orjson.loads(f.read())
            
            self.metadata = data["metadata"]
            
            for node_data in data["nodes"]:
                node = MemoryNode.from_dict(node_data)
                self.nodes[node.id] = node
                
                for intent in node.trigger_intents:
                    if intent not in self.trigger_index:
                        self.trigger_index[intent] = []
                    self.trigger_index[intent].append((node.id, node.confidence))
                
                if node.type not in self.type_index:
                    self.type_index[node.type] = set()
                self.type_index[node.type].add(node.id)
                
                if node.status not in self.status_index:
                    self.status_index[node.status] = set()
                self.status_index[node.status].add(node.id)
                
                self.importance_index.add((node.importance_score, node.id))
            
            for edge_data in data["edges"]:
                self.edges.append(MemoryEdge(**edge_data))
            
            for intent in self.trigger_index:
                self.trigger_index[intent].sort(key=lambda x: -x[1])
        
        except Exception as e:
            print(f"Error loading graph: {e}")