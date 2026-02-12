"""
Synapse-RL: Long-term memory system for conversational AI
"""

__version__ = "1.0.0"
__author__ = "Synapse-RL Team"

from .memory_graph import MemoryGraph, MemoryNode, MemoryEdge
from .memory_decay import MemoryDecayEngine, DecayParameters
from .memory_compiler import MemoryCompiler, CompiledProfile
from .conversation_manager import ConversationManager
from .llm_client import OllamaClient

__all__ = [
    "MemoryGraph",
    "MemoryNode", 
    "MemoryEdge",
    "MemoryDecayEngine",
    "DecayParameters",
    "MemoryCompiler",
    "CompiledProfile",
    "ConversationManager",
    "OllamaClient"
]