from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List

# Customer
class CustomerOnboardRequest(BaseModel):
    name: str
    email: EmailStr
    budget: float = Field(gt=0)

class CustomerResponse(BaseModel):
    customer_id: str
    name: str
    email: str
    budget: float

# Property / Deal
class PropertyOnboardRequest(BaseModel):
    address: str
    price: float = Field(gt=0)
    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    customer_id: str

class PropertyResponse(BaseModel):
    property_id: str
    address: str
    price: float
    bedrooms: int
    bathrooms: int
    customer_id: str

# Marketing Analysis
class AnalysisRequest(BaseModel):
    property_id: str
    property_data: dict

class AnalysisResponse(BaseModel):
    status: str
    insight_preview: Optional[str] = None
    message: Optional[str] = None

# Marketing Query (RAG)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class RetrievedChunk(BaseModel):
    text: str
    metadata: dict

class QueryResponse(BaseModel):
    chunks: List[RetrievedChunk]

# A2A envelope
class A2AResponse(BaseModel):
    status: str  # "success", "error"
    data: Optional[dict] = None
    error: Optional[str] = None