import chromadb
from typing import List, Dict

CHROMA_PATH = "./chroma_db"

# Persistent local vector store – survives restarts
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="market_insights")

def store_insight_chunks(property_id: str, chunks: List[str], embeddings: List[List[float]]):
    """Replace any existing chunks for this property with the new ones (dedup)."""
    existing = collection.get(where={"property_id": property_id})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])
    
    ids = [f"{property_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"property_id": property_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_insights(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """Return the top‑k most similar chunks for a given query embedding."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    if not results or not results["ids"][0]:
        return []
    chunks = []
    for idx, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][idx] if results["metadatas"] else {}
        })
    return chunks

def check_property_exists(property_id: str) -> bool:
    """Quick check to see if a property already has insights stored."""
    existing = collection.get(where={"property_id": property_id})
    return len(existing.get("ids", [])) > 0