import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import sqlite3
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from shared.a2a_client import A2AClient
from shared.logging_config import setup_logger
from .graph import create_graph

logger = setup_logger("Concierge")
app = FastAPI(title="Concierge Agent")

# Discover agents by reading their /card endpoints
config_path = os.path.join(os.path.dirname(__file__), "agent_cards_config.json")
with open(config_path) as f:
    config = json.load(f)

cards = {}
for agent in config["agents"]:
    try:
        resp = requests.get(agent["url"])
        resp.raise_for_status()
        card = resp.json()
        cards[card["agent_name"]] = card
        logger.info(f"Discovered agent: {card['agent_name']}")
    except Exception as e:
        logger.error(f"Failed to discover agent at {agent['url']}: {e}")

a2a_client = A2AClient(cards)

# Set up checkpointing so failed workflows can be resumed
db_path = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")
sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()

# Build the LangGraph application with all the nodes and edges
compiled_graph = create_graph(a2a_client, checkpointer)

# Cleanup on server shutdown
@app.on_event("shutdown")
def shutdown():
    sqlite_conn.close()
    logger.info("Checkpoint database closed")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(req: ChatRequest):
    # Fresh state for each conversation turn
    initial_state = {
        "session_id": req.session_id,
        "messages": [],
        "user_request": req.message,
        "next_task": "",
        "customer_id": None,
        "property_id": None,
        "property_details": {},
        "insights_preview": None,
        "retrieved_chunks": None,
        "final_response": None,
        "error": None
    }
    config = {"configurable": {"thread_id": req.session_id}}
    try:
        final_state = compiled_graph.invoke(initial_state, config)
        return {"status": "success", "response": final_state.get("final_response", "No response.")}
    except Exception as e:
        logger.exception("Error during graph execution")
        raise HTTPException(status_code=500, detail=str(e))

# Simple static HTML UI for quick testing
@app.get("/ui", response_class=HTMLResponse)
def ui():
    ui_file = os.path.join(os.path.dirname(__file__), "ui.html")
    if os.path.exists(ui_file):
        with open(ui_file, "r") as f:
            html = f.read()
        return HTMLResponse(content=html)
    return HTMLResponse(content="<h1>UI file not found</h1>", status_code=404)