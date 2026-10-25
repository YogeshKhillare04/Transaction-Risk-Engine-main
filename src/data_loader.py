import pandas as pd
from src.config import DATA_PATH


def load_credit_card_data() -> pd.DataFrame:
    """
    Loads the real Kaggle Credit Card Fraud Detection dataset.

    Expected columns:
    Time, V1-V28, Amount, Class

    Class:
        0 = genuine transaction
        1 = fraudulent transaction
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it using Kaggle CLI and place creditcard.csv inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    return df


def basic_data_report(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nFraud percentage:")
    print(df["Class"].mean() * 100)

    print("\nMissing values:")
    print(df.isnull().sum().sum())