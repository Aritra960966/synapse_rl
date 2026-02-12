"""
Utility functions for Synapse-RL
"""

import json
import hashlib
import time
from typing import Any, Dict, List
from pathlib import Path


def generate_memory_id(turn: int, content: str) -> str:
    """Generate unique memory ID"""
    hash_part = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"mem_{turn:04d}_{hash_part}"


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between vectors"""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def save_json(data: Dict[str, Any], filepath: Path):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON data"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_latency(ms: float) -> str:
    """Format latency for display"""
    if ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms/1000:.1f}s"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def calculate_memory_size(obj: Any) -> int:
    """Calculate approximate memory size of object"""
    import sys
    return sys.getsizeof(json.dumps(obj))


def ensure_directory(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Get current timestamp"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_intent_from_text(text: str) -> str:
    """Simple intent parsing from text"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["call", "phone", "ring"]):
        return "schedule_call"
    elif any(word in text_lower for word in ["prefer", "like", "want"]):
        return "preference"
    elif any(word in text_lower for word in ["remember", "note", "save"]):
        return "inform"
    elif any(word in text_lower for word in ["what", "when", "where", "who"]):
        return "request"
    else:
        return "other"