import sqlite3
from fastapi import FastAPI, HTTPException
from shared.logging_config import setup_logger
from shared.models import CustomerOnboardRequest, A2AResponse
from .models import init_db, insert_customer, get_customer

logger = setup_logger("CustomerAgent")
app = FastAPI(title="Customer Onboarding Agent")

@app.on_event("startup")
def startup():
    init_db()   # make sure the customers table exists
    logger.info("Customer Agent started. DB ready.")

@app.get("/card")
def get_card():
    # Let the Concierge (or any client) know what this agent can do
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
    # If the email already exists, return the existing ID instead of an error
    conn = sqlite3.connect("customer.db")
    c = conn.cursor()
    c.execute("SELECT customer_id FROM customers WHERE email=?", (req.email,))
    row = c.fetchone()
    conn.close()
    if row:
        existing_id = row[0]
        logger.info(f"Customer with email {req.email} already exists, returning existing ID {existing_id}")
        return A2AResponse(status="success", data={"customer_id": existing_id, "is_new": False})

    # Fresh customer – insert and return the new ID
    try:
        cid = insert_customer(req.name, req.email, req.budget)
        logger.info(f"Onboarded customer {cid} ({req.name}, {req.email})")
        return A2AResponse(status="success", data={"customer_id": cid, "is_new": True})
    except Exception as e:
        logger.error(f"Failed to onboard customer: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{customer_id}", response_model=A2AResponse)
def get_customer_endpoint(customer_id: str):
    cust = get_customer(customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    return A2AResponse(status="success", data=cust)