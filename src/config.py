from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

DATABASE_PATH = BASE_DIR / "fraud_engine.db"

RANDOM_STATE = 42
TEST_SIZE = 0.2

REVIEW_THRESHOLD = 0.30
DECLINE_THRESHOLD = 0.70

FRAUD_LOSS_MULTIPLIER = 1.0
MANUAL_REVIEW_COST = 5.0
FALSE_POSITIVE_COST = 10.0