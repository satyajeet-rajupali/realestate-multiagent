
# 🏠 Federated Multi‑Agent Real Estate System
### Agent‑to‑Agent (A2A) Protocol · LangGraph Orchestration · RAG · Streamlit UI

A fully local, open‑source demonstration of a **federated multi‑agent system** for a real estate platform.  
A central **Concierge Agent** discovers specialized agents via their **Agent Cards**, orchestrates
multi‑step workflows, and aggregates responses. The system uses **LangGraph** for stateful
orchestration, **ChromaDB** for vector storage (RAG), **SQLite** for persistence, and **Ollama**
for local LLM inference and embeddings.

---

## 📐 Architecture

```
 User (CLI / Streamlit)
        │
        ▼
┌─────────────────┐     A2A (REST)     ┌──────────────┬──────────────┬──────────────┐
│  Concierge Agent │ ─────────────────▶ │ Customer     │  Deal        │ Marketing    │
│  (FastAPI +      │                    │ Onboarding   │  Onboarding  │ Intelligence │
│   LangGraph)      │ ◀──────────────── │ (FastAPI)    │  (FastAPI)   │ (FastAPI)    │
│  Port 8000       │     A2A responses  │ Port 8001    │  Port 8002   │ Port 8003    │
└─────────────────┘                    └──────┬───────┴──────┬───────┴──────┬───────┘
                                             │              │              │
                                        SQLite         SQLite        ChromaDB
                                      (customers)    (properties)   (embeddings)
```

- All agents are **independently deployable** (one folder, one process).
- The **Concierge dynamically discovers** agents by fetching their `/card` endpoints.
- The **A2A protocol** uses a standard JSON envelope:
  ```json
  { "status": "success" | "error", "data": { ... }, "error": "..." }
  ```

### Core Workflow (LangGraph)
```
Customer Onboarding → Deal Onboarding → Marketing Intelligence (store in RAG) → Aggregate
                                                                                        ↓
                                                              RAG Query ← Concierge ← User
```

---

## 📋 System Capabilities (All Demonstrated)

| Capability                               | How it’s shown                                                                 |
|------------------------------------------|--------------------------------------------------------------------------------|
| Agent discovery & task delegation        | Concierge fetches `/card` at startup; routes requests by task name.            |
| Structured A2A communication             | All inter‑agent calls use the shared JSON schema.                              |
| End‑to‑end workflow orchestration        | LangGraph state machine automatically chains Customer → Deal → Marketing.      |
| Automatic triggering of downstream agents| Marketing Agent is called immediately after property onboarding (no user step).|
| RAG‑based retrieval & response generation| Marketing stores embeddings in ChromaDB; Concierge queries & synthesises.      |
| Persistent storage & checkpointing       | SQLite for customers/properties; ChromaDB for vectors; LangGraph SQLite checkpoints |
| Logging & observability                 | Every agent logs timestamped messages to stdout.                               |

---

## 🔧 Prerequisites (Windows / Linux / macOS)

- **Python 3.10+** (with `pip`)
- **Ollama** – install from [ollama.com](https://ollama.com)  
  After installation, pull the required models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```
- **Git** (optional, for cloning)

All other dependencies are Python packages (listed in each agent’s `requirements.txt`).

---

## 🚀 Setup & Execution (Step‑by‑Step)

### 1. Clone / Create the Project Directory

Create a root folder (e.g., `realestate-multiagent`) with the following structure:

```
realestate-multiagent/
├── shared/
│   ├── __init__.py
│   ├── a2a_client.py
│   ├── models.py
│   └── logging_config.py
├── customer_agent/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── deal_agent/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── marketing_agent/
│   ├── __init__.py
│   ├── main.py
│   ├── chroma_store.py
│   └── requirements.txt
├── concierge/
│   ├── __init__.py
│   ├── main.py
│   ├── graph.py
│   ├── streamlit_app.py
│   ├── agent_cards_config.json
│   └── requirements.txt
└── README.md
```

> All file contents are available in the repository.  
> Place every file exactly as shown.

### 2. Install Dependencies

Open a terminal (Command Prompt, PowerShell, Bash) in the **project root** and run:

**Windows** (virtual environment recommended):
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r customer_agent/requirements.txt
pip install -r deal_agent/requirements.txt
pip install -r marketing_agent/requirements.txt
pip install -r concierge/requirements.txt
```

**Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r customer_agent/requirements.txt
pip install -r deal_agent/requirements.txt
pip install -r marketing_agent/requirements.txt
pip install -r concierge/requirements.txt
```

### 3. Set PYTHONPATH (so `shared` package is visible)

From the **project root**, set the environment variable:

**Windows (cmd):**
```cmd
set PYTHONPATH=%cd%
```
**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
```
**Linux / macOS (Bash):**
```bash
export PYTHONPATH=$PWD
```

> Alternative: add the project root to your IDE’s PYTHONPATH or install the project in editable mode.

### 4. Start the Agents (4 terminals)

Always from the **project root**.

| Agent                  | Command                                                        | Port  |
|------------------------|----------------------------------------------------------------|-------|
| **Customer Onboarding** | `uvicorn customer_agent.main:app --port 8001 --reload`        | 8001  |
| **Deal Onboarding**     | `uvicorn deal_agent.main:app --port 8002 --reload`            | 8002  |
| **Marketing Intelligence**| `uvicorn marketing_agent.main:app --port 8003 --reload`      | 8003  |
| **Concierge**           | `uvicorn concierge.main:app --port 8000 --reload`             | 8000  |

Wait until each terminal shows `Application startup complete`.

### 5. Launch the Streamlit UI (Browser‑based Chat)

In a **5th terminal**, from the project root:
```bash
streamlit run concierge/streamlit_app.py --server.port 8501
```
Then open your browser to: `http://localhost:8501`

---

## 🧪 Sample Test Cases

### ✅ 1. Onboard a Customer & Property (Core Flow)

**Using `curl`:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Add customer John Doe, email john@example.com, budget 500000. Then add his property 123 Main St, price 450000, 3 bed, 2 bath.",
    "session_id": "test1"
  }'
```

**Expected response (example):**
```json
{
  "status": "success",
  "response": "Customer onboarded with ID 3f7a... Property onboarded with ID a1b2... Market insight preview: The property at 123 Main St shows strong..."
}
```

### ✅ 2. RAG‑based Market Query
Replace `<PROPERTY_ID>` with the actual ID returned above.
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the market risks and opportunities for property <PROPERTY_ID>?",
    "session_id": "test2"
  }'
```

**Expected:** A synthesised answer based on the chunks retrieved from ChromaDB.

### ✅ 3. Error Handling – Incomplete Input
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Add a customer but do not give email", "session_id": "err1"}'
```

**Expected:** An error message explaining that email is required, returned gracefully.

### ✅ 4. Streamlit UI
- Open `http://localhost:8501`.
- Type the same commands as above and see the system orchestrate in real time.
- The “New Session” button resets the conversation and checks the checkpointing (LangGraph resumes on the server side).

---

## 📁 Deliverables (mapped to the task)

| Deliverable                             | Location / Proof                                                                                   |
|-----------------------------------------|----------------------------------------------------------------------------------------------------|
| Concierge Agent                         | `concierge/` – FastAPI app with LangGraph orchestration, SQLite checkpointer                       |
| Customer Onboarding Agent (A2A Server)  | `customer_agent/` – FastAPI, exposes `/onboard` & `/card`                                          |
| Deal Onboarding Agent (A2A Server)      | `deal_agent/` – FastAPI, exposes `/onboard_property` & `/card`                                     |
| Marketing Intelligence Agent (A2A Server)| `marketing_agent/` – FastAPI, uses ChromaDB + Ollama, exposes `/analyze`, `/query`, `/card`       |
| Valid Agent Cards for all agents        | Each agent’s `/card` endpoint; also static `agent_card.json` files                                |
| Shared utilities                        | `shared/` (A2A client, Pydantic models, logging)                                                  |
| README                                  | This document                                                                                      |

---

## 🔍 Observability & Logging

All agents print structured logs to **stdout**:
```
2026-04-24 10:00:00,123 - CustomerAgent - INFO - Onboarded customer 3f7a... (John Doe, john@example.com)
2026-04-24 10:00:01,456 - DealAgent - INFO - Onboarded property a1b2... (123 Main St)
2026-04-24 10:00:02,789 - MarketingAgent - INFO - Generated insight for property a1b2...
2026-04-24 10:00:03,012 - Concierge - INFO - Router decided: onboard_full_flow
```

Checkpoint file `checkpoints.sqlite` is created inside `concierge/` upon first graph execution.

---

## 🔄 Checkpointing & Resume

The Concierge uses **LangGraph’s SQLite checkpointer**.  
If you stop the Concierge during a workflow and restart it, sending a request with the same `session_id` will resume from the last successful node.

---

## 🧰 Tech Stack Summary

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Orchestration      | LangGraph (StateGraph)              |
| Protocol           | A2A (JSON over REST)                |
| Framework          | FastAPI                             |
| LLM                | Ollama – `llama3.2` (3B)            |
| Embeddings         | Ollama – `nomic-embed-text`         |
| Vector DB          | ChromaDB (persistent local)         |
| Persistence        | SQLite (customers, properties)      |
| Checkpointing      | SQLite (LangGraph SQLite saver)     |
| Chat UI            | Streamlit                           |
| Language           | Python 3.10+                        |

All components are **open‑source** and run **entirely locally**.

---

## 💡 Extending the System

The modular design makes it easy to add new agents or capabilities:
1. Create a new agent folder with a FastAPI server.
2. Expose a `/card` endpoint describing its tasks.
3. Add the agent’s URL to `concierge/agent_cards_config.json`.
4. Add new nodes/edges to `concierge/graph.py` to route to the new agent.

No other agent needs to change – the Concierge discovers and routes automatically.

---

## 🤝 Support

If you encounter any issues, check:
- Ollama is running (`ollama list` in terminal).
- All agents are started from the **project root**.
- `PYTHONPATH` is set correctly (or you’re using a virtual environment).
- Windows Firewall has allowed Python/uvicorn to accept connections.