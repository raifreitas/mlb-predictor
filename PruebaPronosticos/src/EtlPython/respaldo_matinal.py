import os
import sqlite3
import sys

ayer = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AYER", "")
db = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "mlb.db")
)
conn = sqlite3.connect(db)
n = conn.execute(
    "SELECT COUNT(*) FROM GameLog WHERE EsFinal=1 AND Fecha=?", (ayer,)
).fetchone()[0]
print("0" if n > 0 else "1")
