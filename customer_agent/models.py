import sqlite3
import uuid
from typing import Optional

DB_PATH = "customer.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            budget REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_customer(name: str, email: str, budget: float) -> str:
    cid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO customers VALUES (?,?,?,?)", (cid, name, email, budget))
    conn.commit()
    conn.close()
    return cid

def get_customer(cid: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM customers WHERE customer_id=?", (cid,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"customer_id": row[0], "name": row[1], "email": row[2], "budget": row[3]}
    return None