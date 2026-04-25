from fastapi import FastAPI, HTTPException
from shared.logging_config import setup_logger
from shared.models import PropertyOnboardRequest, A2AResponse
from .models import init_db, insert_property, get_property

logger = setup_logger("DealAgent")
app = FastAPI(title="Deal Onboarding Agent")

@app.on_event("startup")
def startup():
    # Ensure the properties table is ready before we accept requests
    init_db()
    logger.info("Deal Agent started. DB ready.")

@app.get("/card")
def get_card():
    # Publish our capabilities so the Concierge can discover us
    return {
        "agent_name": "DealOnboardingAgent",
        "base_url": "http://localhost:8002",
        "capabilities": [
            {
                "task": "onboard_property",
                "endpoint": "/onboard_property",
                "method": "POST",
                "input_schema": PropertyOnboardRequest.schema(),
                "output_schema": {"property_id": "str"}
            },
            {
                "task": "get_property",
                "endpoint": "/property/{property_id}",
                "method": "GET"
            }
        ]
    }

@app.post("/onboard_property", response_model=A2AResponse)
def onboard_property(req: PropertyOnboardRequest):
    # Store a new property and return its details
    try:
        pid = insert_property(req.address, req.price, req.bedrooms, req.bathrooms, req.customer_id)
        logger.info(f"Onboarded property {pid} ({req.address})")
        return A2AResponse(status="success", data={
            "property_id": pid,
            "address": req.address,
            "price": req.price,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "customer_id": req.customer_id
        })
    except Exception as e:
        logger.error(f"Property onboarding error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/property/{property_id}", response_model=A2AResponse)
def get_property_endpoint(property_id: str):
    # Look up a property by its ID; return 404 if not found
    prop = get_property(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return A2AResponse(status="success", data=prop)