CREATE TABLE IF NOT EXISTS scored_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_time REAL,
    amount REAL,
    fraud_probability REAL,
    risk_band TEXT,
    decision TEXT,
    created_at TEXT
);