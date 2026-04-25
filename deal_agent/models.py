import sqlite3
import uuid
from typing import Optional

DB_PATH = "deal.db"

def init_db():
    """Create the properties table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            property_id TEXT PRIMARY KEY,
            address TEXT NOT NULL,
            price REAL NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms INTEGER NOT NULL,
            customer_id TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_property(address: str, price: float, bedrooms: int, bathrooms: int, customer_id: str) -> str:
    """Generate a random ID and insert a new property record."""
    pid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO properties VALUES (?,?,?,?,?,?)",
        (pid, address, price, bedrooms, bathrooms, customer_id)
    )
    conn.commit()
    conn.close()
    return pid

def get_property(pid: str) -> Optional[dict]:
    """Fetch a single property by ID; return None if not found."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM properties WHERE property_id=?", (pid,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "property_id": row[0],
            "address": row[1],
            "price": row[2],
            "bedrooms": row[3],
            "bathrooms": row[4],
            "customer_id": row[5]
        }
    return None