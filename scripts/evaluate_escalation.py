import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import classify_intent, retrieve_evidence

INPUT_FILE = "data/golden/golden_set.csv"
OUTPUT_FILE = "outputs/escalation_evaluation.csv"


def get_similarity(evidence):
    """Handle retrieve_evidence() returning a list."""
    if not evidence:
        return 0.0

    # Expected format: [(score, customer_message, response), ...]
    if isinstance(evidence, list):
        first = evidence[0]

        if isinstance(first, (tuple, list)) and len(first) > 0:
            try:
                return float(first[0])
            except:
                return 0.0

        if isinstance(first, dict):
            return float(first.get("similarity", first.get("score", 0)))

    if isinstance(evidence, dict):
        return float(
            evidence.get("similarity",
            evidence.get("score", 0))
        )

    return 0.0


def get_confidence(prediction):
    if isinstance(prediction, dict):
        return float(prediction.get("confidence", 0))

    if isinstance(prediction, (tuple, list)):
        for value in prediction:
            if isinstance(value, (int, float)):
                return float(value)

    return 0.0


def get_intent(prediction):
    if isinstance(prediction, dict):
        return str(
            prediction.get("intent",
            prediction.get("label", "general_support"))
        )

    if isinstance(prediction, (tuple, list)):
        for value in prediction:
            if isinstance(value, str):
                return value

    return "general_support"


def decide_escalation(confidence, similarity):
    if confidence < 0.60:
        return "ESCALATE", "Low intent confidence"

    if similarity <= 0:
        return "ESCALATE", "No historical evidence found"

    if similarity < 0.08:
        return "ESCALATE", "Weak historical evidence"

    return "AUTO_HANDLE", "Sufficient confidence and historical evidence"


def main():

    print("=" * 70)
    print("HIVER SUPPORT AGENT - ESCALATION EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Examples: {len(df)}")

    results = []

    for i, row in df.iterrows():

        message = str(row["customer_message"])

        try:
            prediction = classify_intent(message)

            intent = get_intent(prediction)
            confidence = get_confidence(prediction)

            evidence = retrieve_evidence(message)
            similarity = get_similarity(evidence)

            decision, reason = decide_escalation(
                confidence,
                similarity
            )

            results.append({
                "customer_message": message,
                "human_intent": row.get("human_label", ""),
                "predicted_intent": intent,
                "confidence": confidence,
                "decision": decision,
                "reason": reason,
                "similarity": similarity
            })

        except Exception as e:

            results.append({
                "customer_message": message,
                "human_intent": row.get("human_label", ""),
                "predicted_intent": "ERROR",
                "confidence": 0,
                "decision": "ESCALATE",
                "reason": f"System error: {e}",
                "similarity": 0
            })

        if (i + 1) % 25 == 0:
            print(f"Processed: {i + 1}/{len(df)}")

    result_df = pd.DataFrame(results)

    os.makedirs("outputs", exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("DECISION COUNTS")
    print("=" * 70)
    print(result_df["decision"].value_counts())

    print()
    print("=" * 70)
    print("ESCALATION REASONS")
    print("=" * 70)
    print(result_df["reason"].value_counts())

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(OUTPUT_FILE)

    print()
    print("ESCALATION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()