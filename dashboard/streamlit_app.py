import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import streamlit as st
import plotly.express as px

from app.database import create_tables, fetch_scored_transactions
from src.config import METRICS_PATH

st.set_page_config(
    page_title="Fraud Risk Decision Dashboard",
    layout="wide",
)

st.title("Transaction Risk Scoring and Fraud Decision Dashboard")

create_tables()


@st.cache_data
def load_scored_test_data():
    path = "data/processed/scored_test_transactions.csv"
    return pd.read_csv(path)


@st.cache_data
def load_threshold_results():
    path = "data/processed/threshold_tuning_results.csv"
    return pd.read_csv(path)


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


tab1, tab2, tab3, tab4 = st.tabs([
    "Model Performance",
    "Business Decisions",
    "Threshold Tuning",
    "Live API Transactions",
])

with tab1:
    st.header("Model Performance")

    metrics = load_metrics()

    st.subheader("Best Model")
    st.write(metrics["best_model"])

    model_metrics = metrics["model_metrics"]

    metric_rows = []
    for model_name, values in model_metrics.items():
        metric_rows.append({
            "model": model_name,
            "roc_auc": values["roc_auc"],
            "pr_auc": values["pr_auc"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1_score"],
        })

    metric_df = pd.DataFrame(metric_rows)

    st.dataframe(metric_df)

    fig = px.bar(
        metric_df,
        x="model",
        y=["roc_auc", "pr_auc", "precision", "recall", "f1_score"],
        barmode="group",
        title="Model Comparison",
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Business Decision Summary")

    df = load_scored_test_data()

    total_transactions = len(df)
    fraud_rate = df["actual_fraud"].mean()
    approval_rate = (df["decision"] == "APPROVE").mean()
    review_rate = (df["decision"] == "REVIEW").mean()
    decline_rate = (df["decision"] == "DECLINE").mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Rate", f"{fraud_rate:.3%}")
    col3.metric("Approval Rate", f"{approval_rate:.2%}")
    col4.metric("Review Rate", f"{review_rate:.2%}")
    col5.metric("Decline Rate", f"{decline_rate:.2%}")

    decision_counts = df["decision"].value_counts().reset_index()
    decision_counts.columns = ["decision", "count"]

    fig = px.pie(
        decision_counts,
        names="decision",
        values="count",
        title="Approve vs Review vs Decline"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Score Distribution")
    fig2 = px.histogram(
        df,
        x="fraud_probability",
        color="decision",
        nbins=50,
        title="Fraud Probability Distribution by Decision"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sample Scored Transactions")
    st.dataframe(df.head(100))

with tab3:
    st.header("Threshold Tuning Results")

    threshold_df = load_threshold_results()

    st.subheader("Best Threshold Combinations")
    st.dataframe(threshold_df.head(20))

    fig = px.scatter(
        threshold_df,
        x="fraud_capture_rate",
        y="total_cost",
        color="review_rate",
        size="decline_rate",
        hover_data=[
            "review_threshold",
            "decline_threshold",
            "approval_rate",
            "review_rate",
            "decline_rate",
        ],
        title="Fraud Capture Rate vs Total Business Cost",
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Live API Scored Transactions")

    rows = fetch_scored_transactions()

    live_df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "transaction_time",
            "amount",
            "fraud_probability",
            "risk_band",
            "decision",
            "created_at",
        ],
    )

    if live_df.empty:
        st.info("No live API transactions scored yet.")
    else:
        st.dataframe(live_df)

        decision_counts = live_df["decision"].value_counts().reset_index()
        decision_counts.columns = ["decision", "count"]

        fig = px.bar(
            decision_counts,
            x="decision",
            y="count",
            title="Live API Decision Counts",
        )

        st.plotly_chart(fig, use_container_width=True)