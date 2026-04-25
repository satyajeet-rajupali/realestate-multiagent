from fastapi import FastAPI, HTTPException
from shared.logging_config import setup_logger
from shared.models import CustomerOnboardRequest, A2AResponse
from .models import init_db, insert_customer, get_customer

logger = setup_logger("CustomerAgent")
app = FastAPI(title="Customer Onboarding Agent")

@app.on_event("startup")
def startup():
    init_db()
    logger.info("Customer Agent started. DB ready.")

@app.get("/card")
def get_card():
    return {
        "agent_name": "CustomerOnboardingAgent",
        "base_url": "http://localhost:8001",
        "capabilities": [
            {
                "task": "onboard_customer",
                "endpoint": "/onboard",
                "method": "POST",
                "input_schema": CustomerOnboardRequest.schema(),
                "output_schema": {"customer_id": "str"}
            },
            {
                "task": "get_customer",
                "endpoint": "/customer/{customer_id}",
                "method": "GET"
            }
        ]
    }

@app.post("/onboard", response_model=A2AResponse)
def onboard_customer(req: CustomerOnboardRequest):
    try:
        cid = insert_customer(req.name, req.email, req.budget)
        logger.info(f"Onboarded customer {cid} ({req.name}, {req.email})")
        return A2AResponse(status="success", data={"customer_id": cid})
    except Exception as e:
        logger.error(f"Failed to onboard customer: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{customer_id}", response_model=A2AResponse)
def get_customer_endpoint(customer_id: str):
    cust = get_customer(customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    return A2AResponse(status="success", data=cust)