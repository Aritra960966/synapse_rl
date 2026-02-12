#!/bin/bash

echo "Setting up Synapse-RL with 1B model..."

# Check Python
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Ollama if needed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama
ollama serve &
sleep 3

# Pull TinyLlama 1.1B model
echo "Pulling TinyLlama 1.1B model..."
ollama pull tinyllama:latest

# Create cache directory
mkdir -p .synapse_cache

echo "Setup complete"
echo "Run: python main.py"