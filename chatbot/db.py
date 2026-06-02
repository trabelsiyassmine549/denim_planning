import os
import pyodbc
from contextlib import contextmanager
from typing import List, Dict, Any

#  Connection string 
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
    conn = pyodbc.connect(DB_CONN_STR, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cols = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(cols, row)))
        return rows


def query_one(sql: str, params: tuple = ()) -> Dict[str, Any] | None:
    results = query(sql, params)
    return results[0] if results else None


def scalar(sql: str, params: tuple = ()):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None