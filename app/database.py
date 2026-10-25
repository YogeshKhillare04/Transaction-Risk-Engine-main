import sqlite3
from datetime import datetime

from src.config import DATABASE_PATH


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_time REAL,
            amount REAL,
            fraud_probability REAL,
            risk_band TEXT,
            decision TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def insert_scored_transaction(transaction_time, amount, fraud_probability, risk_band, decision):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO scored_transactions (
            transaction_time,
            amount,
            fraud_probability,
            risk_band,
            decision,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            transaction_time,
            amount,
            fraud_probability,
            risk_band,
            decision,
            datetime.utcnow().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def fetch_scored_transactions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            transaction_time,
            amount,
            fraud_probability,
            risk_band,
            decision,
            created_at
        FROM scored_transactions
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return rows