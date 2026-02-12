### 1️⃣ Install Required Tools

Install these first:

• Python 3.10+ → https://python.org\
✔ During install check: "Add Python to PATH"

• Git → https://git-scm.com

• Ollama → https://ollama.com

Test installation:

    python --version
    git --version
    ollama --version

------------------------------------------------------------------------

### 2️⃣ Clone the Project

Open PowerShell:

    cd D:\
    mkdir hackathon
    cd hackathon
    git clone https://github.com/YOUR_REPO_URL/synapse_rl.git
    cd synapse_rl

------------------------------------------------------------------------

### 3️⃣ Download the AI Model

    ollama pull tinyllama:latest

Wait until download completes.

------------------------------------------------------------------------

### 4️⃣ Setup Virtual Environment

Inside the project folder:

    python -m venv venv
    venv\Scripts\activate

You should now see:

    (venv)

------------------------------------------------------------------------

### 5️⃣ Install Dependencies

    pip install -r requirements.txt

If any module is missing:

    pip install sortedcontainers orjson aiohttp aiofiles sentence-transformers

------------------------------------------------------------------------

## ▶️ Run the AI

Make sure:

• Ollama is running\
• Virtual environment is active

Then run:

    python main.py

------------------------------------------------------------------------

## 💬 How to Use

Tell it something:

    My name is Alex.

Then test memory:

    What is my name?

------------------------------------------------------------------------

## 🛠 Useful Commands

Inside chat:

    stats   → Show memory information
    clear   → Reset memory
    exit    → Save and quit

------------------------------------------------------------------------

## 🔄 Update Project

To get latest updates from GitHub:

    git pull

------------------------------------------------------------------------

## 📂 Memory Location

All memory is stored locally inside:

    .synapse_cache/

Delete this folder if you want to reset everything.

------------------------------------------------------------------------

## ⚠ Common Issues

Ollama not found?

    ollama serve

------------------------------------------------------------------------

## ✅ Done!

You now have a fully offline AI with long-term memory running locally.

No cloud. No API keys. Fully private.
