import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_credit_card_data, basic_data_report
from src.evaluate import evaluate_model, save_metrics
from src.threshold_tuning import tune_thresholds, apply_business_decisions


def build_features(df: pd.DataFrame):
    """
    The dataset has PCA-transformed features V1-V28.
    We scale Time and Amount because they are not PCA-transformed.
    """

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def train_models():
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_credit_card_data()
    basic_data_report(df)

    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )
    X_test_scaled[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    all_metrics = {}
    best_model_name = None
    best_pr_auc = -1
    best_model = None
    best_y_proba = None

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = evaluate_model(y_test, y_proba, threshold=0.5)
        all_metrics[name] = metrics

        print(name, metrics)

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model_name = name
            best_model = model
            best_y_proba = y_proba

    print(f"\nBest model: {best_model_name}")

    threshold_results = tune_thresholds(
        y_true=y_test,
        y_proba=best_y_proba,
        amounts=X_test["Amount"],
        manual_review_cost=5.0,
        false_positive_cost=10.0,
    )

    best_threshold_row = threshold_results.iloc[0].to_dict()

    print("\nBest business thresholds:")
    print(best_threshold_row)

    scored_test = X_test.copy()
    scored_test["actual_fraud"] = y_test.values
    scored_test["fraud_probability"] = best_y_proba

    scored_test = apply_business_decisions(
        scored_test,
        probability_col="fraud_probability",
        review_threshold=best_threshold_row["review_threshold"],
        decline_threshold=best_threshold_row["decline_threshold"],
    )

    scored_test.to_csv("data/processed/scored_test_transactions.csv", index=False)
    threshold_results.to_csv("data/processed/threshold_tuning_results.csv", index=False)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    final_report = {
        "best_model": best_model_name,
        "model_metrics": all_metrics,
        "best_business_thresholds": best_threshold_row,
    }

    save_metrics(final_report, METRICS_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)
    print("Metrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    train_models()