from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
import logging
from shared.a2a_client import A2AClient

logger = logging.getLogger(__name__)

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

def router_node(state: AgentState, a2a: A2AClient) -> AgentState:
    prompt = f"""You are a concierge routing assistant. Given the user request, determine the next action.
Reply with exactly one JSON object containing "task" and "entities".

Available tasks:
- "onboard_full_flow": if user wants to add a new customer and/or a property.
- "query_insights": if user asks about market risks, trends, opportunities for a property.
- "get_customer": if user wants to look up customer by email or ID.
- "get_property": if user wants to look up property by ID.

User request: {state["user_request"]}

JSON:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        decision = eval(response.content)  # simple eval; in production use json.loads
    except:
        decision = {"task": "unknown"}
    state["next_task"] = decision.get("task", "unknown")
    state["property_details"] = decision.get("entities", {})
    logger.info(f"Router decided: {state['next_task']}")
    return state

def customer_onboarding_node(state: AgentState, a2a: A2AClient) -> AgentState:
    extract_prompt = f"""Extract customer details from this text: "{state['user_request']}".
Return as JSON with keys: name, email, budget. If not present, use empty strings."""
    resp = llm.invoke([HumanMessage(content=extract_prompt)])
    try:
        details = eval(resp.content)
    except:
        details = {}
    if not details.get("name") or not details.get("email") or not details.get("budget"):
        state["error"] = "Incomplete customer data. Please provide name, email, and budget."
        return state
    try:
        result = a2a.call("onboard_customer", params={
            "name": details["name"],
            "email": details["email"],
            "budget": float(details["budget"])
        })
        if result["status"] == "success":
            state["customer_id"] = result["data"]["customer_id"]
            logger.info(f"Customer onboarded: {state['customer_id']}")
        else:
            state["error"] = result.get("error", "Customer onboarding failed")
    except Exception as e:
        state["error"] = str(e)
    return state

def deal_onboarding_node(state: AgentState, a2a: A2AClient) -> AgentState:
    if not state.get("property_details") or not state["property_details"].get("address"):
        extract_prompt = f"""Extract property details from this text: "{state['user_request']}".
Return as JSON with keys: address, price, bedrooms, bathrooms. If not present, use empty strings."""
        resp = llm.invoke([HumanMessage(content=extract_prompt)])
        try:
            state["property_details"] = eval(resp.content)
        except:
            state["property_details"] = {}
    pd = state["property_details"]
    if not pd.get("address") or not pd.get("price"):
        state["error"] = "Incomplete property data. Please provide address and price."
        return state
    params = {
        "address": pd["address"],
        "price": float(pd["price"]),
        "bedrooms": int(pd.get("bedrooms", 1)),
        "bathrooms": int(pd.get("bathrooms", 1)),
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

def aggregate_node(state: AgentState) -> AgentState:
    if state.get("retrieved_chunks"):
        context = "\n".join([c["text"] for c in state["retrieved_chunks"]])
        prompt = f"""Based on the following market insights, answer the user's question:
User: {state["user_request"]}
Insights:
{context}
Answer:"""
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
        if not parts:
            state["final_response"] = "Operation completed."
        else:
            state["final_response"] = " ".join(parts)
    return state

def error_node(state: AgentState) -> AgentState:
    state["final_response"] = f"An error occurred: {state.get('error', 'Unknown error')}"
    return state

def create_graph(a2a_client: A2AClient):
    graph = StateGraph(AgentState)
    
    graph.add_node("router", lambda s: router_node(s, a2a_client))
    graph.add_node("customer_onboarding", lambda s: customer_onboarding_node(s, a2a_client))
    graph.add_node("deal_onboarding", lambda s: deal_onboarding_node(s, a2a_client))
    graph.add_node("marketing_analysis", lambda s: marketing_analysis_node(s, a2a_client))
    graph.add_node("rag_query", lambda s: rag_query_node(s, a2a_client))
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("handle_error", error_node)
    
    graph.set_entry_point("router")
    
    def route_next(state: AgentState):
        task = state.get("next_task", "")
        if task == "onboard_full_flow":
            return "customer_onboarding"
        elif task == "query_insights":
            return "rag_query"
        elif task in ["get_customer", "get_property"]:
            return "aggregate"
        return "aggregate"
    
    graph.add_conditional_edges("router", route_next, {
        "customer_onboarding": "customer_onboarding",
        "rag_query": "rag_query",
        "aggregate": "aggregate"
    })
    graph.add_edge("customer_onboarding", "deal_onboarding")
    graph.add_edge("deal_onboarding", "marketing_analysis")
    graph.add_edge("marketing_analysis", "aggregate")
    graph.add_edge("rag_query", "aggregate")
    
    def check_error(state: AgentState):
        return "handle_error" if state.get("error") else END
    
    graph.add_conditional_edges("customer_onboarding", check_error, {END: "deal_onboarding", "handle_error": "handle_error"})
    graph.add_conditional_edges("deal_onboarding", check_error, {END: "marketing_analysis", "handle_error": "handle_error"})
    graph.add_conditional_edges("marketing_analysis", check_error, {END: "aggregate", "handle_error": "handle_error"})
    graph.add_conditional_edges("rag_query", check_error, {END: "aggregate", "handle_error": "handle_error"})
    
    graph.add_edge("aggregate", END)
    graph.add_edge("handle_error", END)
    
    return graph.compile()