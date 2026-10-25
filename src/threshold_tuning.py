import numpy as np
import pandas as pd


def assign_decision(probability: float, review_threshold: float, decline_threshold: float) -> str:
    if probability >= decline_threshold:
        return "DECLINE"
    elif probability >= review_threshold:
        return "REVIEW"
    else:
        return "APPROVE"


def tune_thresholds(
    y_true,
    y_proba,
    amounts,
    review_thresholds=None,
    decline_thresholds=None,
    manual_review_cost=5.0,
    false_positive_cost=10.0,
):
    """
    Business threshold tuning.

    Fraud loss:
        If fraud is approved, company loses transaction amount.

    Manual review cost:
        Every reviewed transaction costs operational money.

    False positive cost:
        Genuine declined transaction creates customer friction.

    Objective:
        Minimize expected business cost.
    """

    if review_thresholds is None:
        review_thresholds = np.arange(0.05, 0.55, 0.05)

    if decline_thresholds is None:
        decline_thresholds = np.arange(0.55, 0.95, 0.05)

    results = []

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    amounts = np.array(amounts)

    for review_t in review_thresholds:
        for decline_t in decline_thresholds:
            if review_t >= decline_t:
                continue

            decisions = np.array([
                assign_decision(p, review_t, decline_t)
                for p in y_proba
            ])

            approved = decisions == "APPROVE"
            reviewed = decisions == "REVIEW"
            declined = decisions == "DECLINE"

            fraud = y_true == 1
            genuine = y_true == 0

            fraud_approved = approved & fraud
            fraud_declined = declined & fraud
            fraud_reviewed = reviewed & fraud

            genuine_declined = declined & genuine
            genuine_reviewed = reviewed & genuine

            fraud_loss = amounts[fraud_approved].sum()
            review_cost = reviewed.sum() * manual_review_cost
            false_positive_cost_total = genuine_declined.sum() * false_positive_cost

            total_cost = fraud_loss + review_cost + false_positive_cost_total

            fraud_capture_rate = (
                (fraud_declined.sum() + fraud_reviewed.sum()) / fraud.sum()
                if fraud.sum() > 0 else 0
            )

            approval_rate = approved.mean()
            review_rate = reviewed.mean()
            decline_rate = declined.mean()

            results.append({
                "review_threshold": round(float(review_t), 3),
                "decline_threshold": round(float(decline_t), 3),
                "fraud_capture_rate": round(float(fraud_capture_rate), 4),
                "approval_rate": round(float(approval_rate), 4),
                "review_rate": round(float(review_rate), 4),
                "decline_rate": round(float(decline_rate), 4),
                "fraud_loss": round(float(fraud_loss), 2),
                "review_cost": round(float(review_cost), 2),
                "false_positive_cost": round(float(false_positive_cost_total), 2),
                "total_cost": round(float(total_cost), 2),
                "fraud_approved": int(fraud_approved.sum()),
                "fraud_reviewed": int(fraud_reviewed.sum()),
                "fraud_declined": int(fraud_declined.sum()),
                "genuine_declined": int(genuine_declined.sum()),
                "genuine_reviewed": int(genuine_reviewed.sum()),
            })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_cost", ascending=True)

    return result_df


def apply_business_decisions(df, probability_col, review_threshold, decline_threshold):
    df = df.copy()
    df["risk_score"] = df[probability_col]
    df["decision"] = df["risk_score"].apply(
        lambda p: assign_decision(p, review_threshold, decline_threshold)
    )

    df["risk_band"] = df["decision"].map({
        "APPROVE": "LOW",
        "REVIEW": "MEDIUM",
        "DECLINE": "HIGH",
    })

    return df