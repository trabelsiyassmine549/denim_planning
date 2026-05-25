import os
import pyodbc
from contextlib import contextmanager
from typing import List, Dict, Any

# ── Connection string ─────────────────────────────────────────────────────────
# Set CHATBOT_DB_CONN in your environment, or edit the default below.
_DEFAULT_CONN = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=CommandesDB;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

DB_CONN_STR = os.getenv("CHATBOT_DB_CONN", _DEFAULT_CONN)


@contextmanager
def get_connection():
    """Yield a pyodbc connection, auto-close on exit."""
    conn = pyodbc.connect(DB_CONN_STR, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """
    Execute a SELECT query and return a list of dicts.
    Each dict maps column_name -> value.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cols = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(cols, row)))
        return rows


def query_one(sql: str, params: tuple = ()) -> Dict[str, Any] | None:
    """Return the first row as a dict, or None."""
    results = query(sql, params)
    return results[0] if results else None


def scalar(sql: str, params: tuple = ()):
    """Return the first column of the first row (scalar value)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None