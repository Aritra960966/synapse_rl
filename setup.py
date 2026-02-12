from setuptools import setup, find_packages

setup(
    name="synapse-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.1.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "streamlit": [
            "streamlit>=1.29.0",
            "plotly>=5.18.0",
            "pandas>=2.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Synapse-RL Team",
    description="Memory-augmented conversational AI system",
    keywords="ai chatbot memory graph rl",
    url="https://github.com/yourusername/synapse-rl",
)
