# Synapse-RL: Long-Term Memory System for Conversational AI

**From Context Windows to Behavioral Programs**

## Overview

Synapse-RL is a production-ready long-term memory system that enables AI assistants to maintain coherent memory across 1000+ conversation turns with constant O(1) latency. Instead of retrieving memories from a database, we compile them into constant-size behavioral constraints.

## Key Features

- **Constant O(1) Latency**: ~150ms at turn 1 or turn 1000
- **100% Offline**: No cloud dependencies, runs entirely local
- **1B Model Optimized**: Designed for TinyLlama 1.1B
- **Smart Memory Decay**: Sophisticated importance-based memory management
- **Automatic Contradiction Resolution**: SUPERSEDES edges handle updates
- **Minimal Storage**: <1MB for 1000 turns

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Aritra960966/synapse-rl.git
cd synapse-rl

# Run setup
chmod +x scripts/setup.sh
./scripts/setup.sh
