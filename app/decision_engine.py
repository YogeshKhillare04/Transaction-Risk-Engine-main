from src.config import REVIEW_THRESHOLD, DECLINE_THRESHOLD


def get_risk_band(probability: float) -> str:
    if probability >= DECLINE_THRESHOLD:
        return "HIGH"
    elif probability >= REVIEW_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def get_decision(probability: float) -> str:
    if probability >= DECLINE_THRESHOLD:
        return "DECLINE"
    elif probability >= REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "APPROVE"


def get_reason_codes(transaction: dict, probability: float):
    """
    Simple explainability layer for business users.
    Since V1-V28 are PCA features, they are not human-readable.
    But Time and Amount are interpretable.
    """

    reasons = []

    amount = transaction.get("Amount", 0)

    if amount > 500:
        reasons.append("High transaction amount")

    if probability >= DECLINE_THRESHOLD:
        reasons.append("Fraud probability exceeds decline threshold")
    elif probability >= REVIEW_THRESHOLD:
        reasons.append("Fraud probability exceeds manual review threshold")
    else:
        reasons.append("Fraud probability is within approval range")

    return reasons