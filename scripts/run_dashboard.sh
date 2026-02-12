#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install additional requirements
pip install -r requirements_streamlit.txt

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address localhost