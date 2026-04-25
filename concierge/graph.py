from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import logging
import json
import re
from shared.a2a_client import A2AClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  State                                                              #
# ------------------------------------------------------------------ #
class AgentState(TypedDict):
    session_id: str
    messages: List[Dict[str, str]]
    user_request: str
    next_task: str
    customer_id: Optional[str]
    property_id: Optional[str]
    property_details: Optional[dict]
    insights_preview: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    final_response: Optional[str]
    error: Optional[str]

llm = ChatOllama(model="llama3.2")

# ------------------------------------------------------------------ #
#  Smart validation node (NEW)                                        #
# ------------------------------------------------------------------ #
def validate_intent(state: AgentState, a2a: A2AClient) -> AgentState:
    user_msg = state["user_request"]

    prompt = f"""You are a strict real‑estate assistant. Examine the user's message and output a JSON object with the following keys:

- "relevant": true if the message is about real‑estate (customers, properties, market risks), false otherwise.
- "intent": one of "onboard_full_flow" (if user wants to add a customer or property), "query_insights" (for questions about markets, risks, trends), "other" (anything else).
- "missing_fields": list of required fields that are missing. For "onboard_full_flow", check for customer fields: name, email, budget; and property fields: address, price. If a field is missing and necessary, add it to the list.
- "clarification_question": a polite question asking for the missing information (only if something is missing, otherwise empty string).

Only output the JSON. Do not write any other text.

User message: "{user_msg}"

JSON:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Extract JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    decision = {
        "relevant": False,
        "intent": "other",
        "missing_fields": [],
        "clarification_question": ""
    }
    if json_match:
        try:
            decision = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If irrelevant, stop early
    if not decision.get("relevant", False):
        state["final_response"] = "I'm a real‑estate assistant. Please ask me about properties, customers, or market insights."
        state["error"] = "IRRELEVANT"  # special flag
        logger.info("Validation: message is not real‑estate related.")
        return state

    # If missing fields for onboarding, ask for them
    missing = decision.get("missing_fields", [])
    if missing and decision.get("intent") == "onboard_full_flow":
        question = decision.get("clarification_question") or f"I still need the following information: {', '.join(missing)}. Could you please provide it?"
        state["final_response"] = question
        state["error"] = "INCOMPLETE"
        logger.info(f"Validation: onboarding data incomplete – missing {missing}.")
        return state

    # Pass through: set the detected intent for the next router
    state["next_task"] = decision.get("intent", "onboard_full_flow")
    state["property_details"] = {}
    logger.info(f"Validation passed. Intent: {state['next_task']}")
    return state

# ------------------------------------------------------------------ #
#  Router node (unchanged)                                            #
# ------------------------------------------------------------------ #
def router_node(state: AgentState, a2a: A2AClient) -> AgentState:
    user_msg = state["user_request"].lower()

    # ---------- keyword shortcuts (no LLM needed) ----------
    if any(phrase in user_msg for phrase in ["add customer", "onboard", "new customer", "add his property", "add her property"]):
        state["next_task"] = "onboard_full_flow"
        state["property_details"] = {}
        logger.info("Router decided (keyword): onboard_full_flow")
        return state

    if any(word in user_msg for word in ["risk", "opportunity", "market", "insight"]):
        state["next_task"] = "query_insights"
        state["property_details"] = {}
        logger.info("Router decided (keyword): query_insights")
        return state

    # ---------- LLM fallback ----------
    prompt = (
        'You are a concierge routing assistant. Output ONLY a JSON object with '
        '"task" and "entities". Do NOT write anything else.\n'
        'Tasks: "onboard_full_flow", "query_insights", "get_customer", "get_property"\n'
        f'User request: {state["user_request"]}\nJSON:'
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # extract first JSON object
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    decision = {"task": "unknown"}
    if m:
        try:
            decision = json.loads(m.group())
        except json.JSONDecodeError:
            pass

    state["next_task"] = decision.get("task", "unknown")
    state["property_details"] = decision.get("entities", {})
    logger.info(f"Router decided (LLM): {state['next_task']}")
    return state

# ------------------------------------------------------------------ #
#  Onboarding nodes (unchanged)                                       #
# ------------------------------------------------------------------ #
def customer_onboarding_node(state: AgentState, a2a: A2AClient) -> AgentState:
    user_text = state["user_request"]

    # ---------- Pre‑check: does the message contain an email or a budget? ----------
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_text))
    has_budget = bool(re.search(r'budget\s+\$?(\d[\d,.]*)', user_text, re.IGNORECASE))

    if not has_email and not has_budget:
        state["error"] = (
            "I need a bit more information to onboard the customer. "
            "Please provide the customer's **email** and **budget**, for example:\n"
            "`Add customer Jane Doe, email jane@example.com, budget 350000.`"
        )
        logger.warning("Incomplete customer data – no email or budget found in message.")
        return state

    # ---------- LLM extraction ----------
    extract_prompt = (
        'Extract ONLY the customer details from the following text. '
        'Return a JSON object with exactly three keys: "name", "email", "budget". '
        'Output nothing but the JSON. If a field is missing, use an empty string.\n'
        f'Text: {user_text}'
    )
    resp = llm.invoke([HumanMessage(content=extract_prompt)])
    raw = resp.content.strip()

    details = {}
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            details = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # ---------- Regex fallback ----------
    if not details.get("name"):
        name_match = re.search(r'(?:name\s*)?([A-Z][a-z]+\s[A-Z][a-z]+)', user_text)
        if name_match:
            details["name"] = name_match.group(1).strip()
    if not details.get("email"):
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_text)
        if email_match:
            details["email"] = email_match.group()
    if not details.get("budget"):
        budget_match = re.search(r'budget\s+\$?(\d[\d,.]*)', user_text, re.IGNORECASE)
        if budget_match:
            details["budget"] = budget_match.group(1).replace(",", "")

    # ---------- Final validation ----------
    try:
        budget_val = float(str(details.get("budget", "")).replace(",", ""))
    except (ValueError, TypeError):
        budget_val = None

    if not details.get("name") or not details.get("email") or budget_val is None or budget_val <= 0:
        state["error"] = (
            "Incomplete customer data. Please make sure you include:\n"
            "- **name** (e.g., John Doe)\n"
            "- **email** (e.g., john@example.com)\n"
            "- **budget** (e.g., 500000)\n\n"
            "You can say something like: `Add customer Jane Doe, email jane@example.com, budget 350000.`"
        )
        return state

    # ---------- Call the Customer Agent ----------
    try:
        result = a2a.call("onboard_customer", params={
            "name": str(details["name"]),
            "email": str(details["email"]),
            "budget": budget_val
        })
        if result["status"] == "success":
            state["customer_id"] = result["data"]["customer_id"]
            logger.info(f"Customer onboarded: {state['customer_id']}")
        else:
            state["error"] = result.get("error", "Customer onboarding failed")
    except Exception as e:
        state["error"] = f"Customer onboarding failed: {str(e)}"
    return state

def deal_onboarding_node(state: AgentState, a2a: A2AClient) -> AgentState:
    extract_prompt = (
        'Extract ONLY the property details from the following text. '
        'Return a JSON object with exactly four keys: "address", "price", "bedrooms", "bathrooms". '
        'Output nothing but the JSON. Do not include customer information.\n'
        f'Text: {state["user_request"]}'
    )
    resp = llm.invoke([HumanMessage(content=extract_prompt)])
    raw = resp.content.strip()

    pd = {}
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            pd = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Regex fallback
    if not pd.get("address"):
        addr_match = re.search(r'(\d+\s[\w\s]+(?:Street|St|Ave|Road|Dr|Lane|Way|Blvd)\.?)\b', state["user_request"])
        if addr_match:
            pd["address"] = addr_match.group(1).strip()
    if not pd.get("price"):
        price_match = re.search(r'price\s+\$?(\d+(?:,\d{3})*(?:\.\d+)?)', state["user_request"], re.IGNORECASE)
        if price_match:
            pd["price"] = price_match.group(1).replace(",", "")
    if not pd.get("bedrooms"):
        bed_match = re.search(r'(\d+)\s*bed', state["user_request"])
        if bed_match:
            pd["bedrooms"] = int(bed_match.group(1))
    if not pd.get("bathrooms"):
        bath_match = re.search(r'(\d+)\s*bath', state["user_request"])
        if bath_match:
            pd["bathrooms"] = int(bath_match.group(1))

    if not pd.get("address") or not pd.get("price"):
        state["error"] = "Incomplete property data. Please provide address and price."
        return state

    pd.setdefault("bedrooms", 1)
    pd.setdefault("bathrooms", 1)

    params = {
        "address": str(pd["address"]),
        "price": float(str(pd["price"]).replace(",", "")),
        "bedrooms": int(pd["bedrooms"]),
        "bathrooms": int(pd["bathrooms"]),
        "customer_id": state.get("customer_id")
    }
    try:
        result = a2a.call("onboard_property", params=params)
        if result["status"] == "success":
            state["property_id"] = result["data"]["property_id"]
            state["property_details"] = result["data"]
            logger.info(f"Property onboarded: {state['property_id']}")
        else:
            state["error"] = result.get("error", "Property onboarding failed")
    except Exception as e:
        state["error"] = str(e)
    return state

def marketing_analysis_node(state: AgentState, a2a: A2AClient) -> AgentState:
    try:
        result = a2a.call("analyze_property", params={
            "property_id": state["property_id"],
            "property_data": state["property_details"]
        })
        if result["status"] == "success":
            state["insights_preview"] = result["data"].get("insight_preview", "")
            logger.info("Marketing analysis completed.")
        else:
            state["error"] = result.get("error", "Marketing analysis failed")
    except Exception as e:
        state["error"] = str(e)
    return state

def rag_query_node(state: AgentState, a2a: A2AClient) -> AgentState:
    query = state["user_request"]
    try:
        result = a2a.call("query_market_insights", params={"query": query, "top_k": 3})
        if result["status"] == "success":
            state["retrieved_chunks"] = result["data"].get("chunks", [])
            logger.info(f"Retrieved {len(state['retrieved_chunks'])} chunks from RAG.")
        else:
            state["error"] = result.get("error", "RAG query failed")
    except Exception as e:
        state["error"] = str(e)
    return state

# ------------------------------------------------------------------ #
#  Aggregate & error nodes (unchanged)                                #
# ------------------------------------------------------------------ #
def aggregate_node(state: AgentState) -> AgentState:
    # If a final_response was already set by validation, keep it
    if state.get("final_response"):
        return state

    if state.get("retrieved_chunks"):
        context = "\n".join([c["text"] for c in state["retrieved_chunks"]])
        prompt = f"User: {state['user_request']}\nInsights:\n{context}\nAnswer:"
        response = llm.invoke([HumanMessage(content=prompt)])
        state["final_response"] = response.content
    else:
        parts = []
        if state.get("customer_id"):
            parts.append(f"Customer onboarded with ID {state['customer_id']}.")
        if state.get("property_id"):
            parts.append(f"Property onboarded with ID {state['property_id']}.")
        if state.get("insights_preview"):
            parts.append(f"Market insight preview: {state['insights_preview']}")
        state["final_response"] = " ".join(parts) if parts else "Operation completed."
    return state

def error_node(state: AgentState) -> AgentState:
    state["final_response"] = f"An error occurred: {state.get('error', 'Unknown error')}"
    return state

# ------------------------------------------------------------------ #
#  Direct response node (for validation rejections)                   #
# ------------------------------------------------------------------ #
def respond_directly(state: AgentState) -> AgentState:
    # final_response already set by validate_intent, just pass
    return state

# ------------------------------------------------------------------ #
#  Graph builder (updated entry)                                      #
# ------------------------------------------------------------------ #
def create_graph(a2a_client: A2AClient, checkpointer=None):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("validate_intent", lambda s: validate_intent(s, a2a_client))
    graph.add_node("router", lambda s: router_node(s, a2a_client))
    graph.add_node("customer_onboarding", lambda s: customer_onboarding_node(s, a2a_client))
    graph.add_node("deal_onboarding", lambda s: deal_onboarding_node(s, a2a_client))
    graph.add_node("marketing_analysis", lambda s: marketing_analysis_node(s, a2a_client))
    graph.add_node("rag_query", lambda s: rag_query_node(s, a2a_client))
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("handle_error", error_node)
    graph.add_node("respond_directly", respond_directly)

    graph.set_entry_point("validate_intent")

    # After validation: if error flag (IRRELEVANT/INCOMPLETE) -> direct response; else -> router
    def after_validation(state: AgentState):
        err = state.get("error", "")
        if err in ("IRRELEVANT", "INCOMPLETE"):
            return "respond_directly"
        return "router"

    graph.add_conditional_edges("validate_intent", after_validation, {
        "respond_directly": "respond_directly",
        "router": "router"
    })

    # ---- router → next action ----
    def route_next(state: AgentState):
        task = state.get("next_task", "")
        if task == "onboard_full_flow":
            return "customer_onboarding"
        if task == "query_insights":
            return "rag_query"
        return "aggregate"

    graph.add_conditional_edges("router", route_next, {
        "customer_onboarding": "customer_onboarding",
        "rag_query": "rag_query",
        "aggregate": "aggregate"
    })

    # ---- sequential chain with error checks ----
    def after_customer(state: AgentState):
        return "deal_onboarding" if not state.get("error") else "handle_error"

    def after_deal(state: AgentState):
        return "marketing_analysis" if not state.get("error") else "handle_error"

    def after_marketing(state: AgentState):
        return "aggregate" if not state.get("error") else "handle_error"

    def after_rag(state: AgentState):
        return "aggregate" if not state.get("error") else "handle_error"

    graph.add_conditional_edges("customer_onboarding", after_customer, {
        "deal_onboarding": "deal_onboarding",
        "handle_error": "handle_error"
    })
    graph.add_conditional_edges("deal_onboarding", after_deal, {
        "marketing_analysis": "marketing_analysis",
        "handle_error": "handle_error"
    })
    graph.add_conditional_edges("marketing_analysis", after_marketing, {
        "aggregate": "aggregate",
        "handle_error": "handle_error"
    })
    graph.add_conditional_edges("rag_query", after_rag, {
        "aggregate": "aggregate",
        "handle_error": "handle_error"
    })

    # ---- terminal edges ----
    graph.add_edge("aggregate", END)
    graph.add_edge("handle_error", END)
    graph.add_edge("respond_directly", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled