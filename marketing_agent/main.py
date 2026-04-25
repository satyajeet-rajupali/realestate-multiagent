from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.logging_config import setup_logger
from shared.models import AnalysisRequest, QueryRequest, RetrievedChunk, A2AResponse
from .chroma_store import store_insight_chunks, query_insights, check_property_exists

logger = setup_logger("MarketingAgent")
app = FastAPI(title="Marketing Intelligence Agent")

# Local models through Ollama – no cloud dependency
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

@app.get("/card")
def get_card():
    # Advertise our analysis and retrieval capabilities to the Concierge
    return {
        "agent_name": "MarketingIntelligenceAgent",
        "base_url": "http://localhost:8003",
        "capabilities": [
            {
                "task": "analyze_property",
                "endpoint": "/analyze",
                "method": "POST",
                "input_schema": AnalysisRequest.schema(),
                "output_schema": {"status": "str", "insight_preview": "str"}
            },
            {
                "task": "query_market_insights",
                "endpoint": "/query",
                "method": "POST",
                "input_schema": QueryRequest.schema(),
                "output_schema": {"chunks": "list[RetrievedChunk]"}
            }
        ]
    }

@app.post("/analyze", response_model=A2AResponse)
def analyze_property(req: AnalysisRequest):
    # Skip if we already have insights for this property
    if check_property_exists(req.property_id):
        logger.warning(f"Property {req.property_id} already analyzed.")
        return A2AResponse(status="success", data={"status": "duplicate", "message": "Already processed."})

    # Ask the LLM to write a short market report
    try:
        prompt = f"""Generate a market intelligence report for the following property:
Address: {req.property_data.get('address')}
Price: {req.property_data.get('price')}
Bedrooms: {req.property_data.get('bedrooms')}
Bathrooms: {req.property_data.get('bathrooms')}
Include trends, risk signals, and opportunity indicators.
"""
        response = llm.invoke(prompt)
        insight_text = response.content
        logger.info(f"Generated insight for property {req.property_id}")
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail="Insight generation error")

    # Split into chunks and create embeddings for storage
    chunks = text_splitter.split_text(insight_text)
    try:
        chunk_embeddings = embeddings.embed_documents(chunks)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Embedding error")

    # Persist everything to ChromaDB
    store_insight_chunks(req.property_id, chunks, chunk_embeddings)

    # Return a short preview so the Concierge can show something immediately
    preview = insight_text[:200] + "..." if len(insight_text) > 200 else insight_text
    return A2AResponse(status="success", data={
        "status": "generated",
        "insight_preview": preview
    })

@app.post("/query", response_model=A2AResponse)
def query_market(req: QueryRequest):
    # Embed the user's question and retrieve the most relevant chunks
    try:
        q_embedding = embeddings.embed_query(req.query)
        retrieved = query_insights(q_embedding, req.top_k)
        chunks = [RetrievedChunk(text=c["text"], metadata=c["metadata"]) for c in retrieved]
        return A2AResponse(status="success", data={"chunks": [c.dict() for c in chunks]})
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))