import os
import json
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    ROOT, "data", "processed", "amazonhelp_pairs.csv"
)

GOLDEN_FILE = os.path.join(
    ROOT, "data", "golden", "golden_set.csv"
)

OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(os.path.dirname(GOLDEN_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTENTS = [
    "delivery_issue",
    "refund_request",
    "order_issue",
    "payment_issue",
    "account_issue",
    "subscription_issue",
    "product_issue",
    "return_issue",
    "price_question",
    "general_support",
]


def load_classifier():
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))

    from agent import classify_intent

    return classify_intent


def create_golden_set():
    print("Creating golden evaluation set...")

    df = pd.read_csv(DATA_FILE)

    sample_size = min(200, len(df))

    golden = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    classify_intent = load_classifier()

    predictions = []
    confidences = []

    for message in golden["customer_message"].fillna(""):
        try:
            result = classify_intent(str(message))

            if isinstance(result, tuple):
                intent = result[0]
                confidence = result[1]
            else:
                intent = result
                confidence = 0.0

        except Exception:
            intent = "general_support"
            confidence = 0.0

        predictions.append(intent)
        confidences.append(confidence)

    golden["predicted_intent"] = predictions
    golden["confidence"] = confidences
    golden["human_label"] = ""

    golden.to_csv(GOLDEN_FILE, index=False)

    print(f"Created: {GOLDEN_FILE}")
    print(f"Examples: {len(golden)}")
    print()
    print("Now run:")
    print("python scripts\\label_golden_set.py")


def evaluate():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - EVALUATION")
    print("=" * 70)

    df = pd.read_csv(GOLDEN_FILE)

    df["human_label"] = (
        df["human_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    labeled = df[df["human_label"] != ""].copy()

    print(f"Total examples : {len(df)}")
    print(f"Labeled        : {len(labeled)}")
    print(f"Remaining      : {len(df) - len(labeled)}")

    if len(labeled) < 150:
        print()
        print("LABELING REQUIRED")
        print("=" * 70)
        print(
            f"Need at least 150 human-labelled examples. "
            f"Currently have {len(labeled)}."
        )
        print()
        print("Run:")
        print("python scripts\\label_golden_set.py")
        return

    y_true = labeled["human_label"]
    y_pred = labeled["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print()
    print("PER-INTENT RESULTS")
    print("-" * 70)

    report = classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        zero_division=0,
        output_dict=True
    )

    for intent in INTENTS:
        if intent in report:
            print(
                f"{intent:20s} "
                f"precision={report[intent]['precision']:.3f} "
                f"recall={report[intent]['recall']:.3f} "
                f"f1={report[intent]['f1-score']:.3f} "
                f"n={int(report[intent]['support'])}"
            )

    # Majority-class baseline
    majority_class = y_true.value_counts().idxmax()

    majority_predictions = [
        majority_class
    ] * len(y_true)

    majority_accuracy = accuracy_score(
        y_true,
        majority_predictions
    )

    majority_f1 = f1_score(
        y_true,
        majority_predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("BASELINE: MAJORITY CLASS")
    print("=" * 70)

    print(f"Majority intent : {majority_class}")
    print(f"Accuracy        : {majority_accuracy:.4f}")
    print(f"Macro F1        : {majority_f1:.4f}")

    # Save metrics
    metrics = {
        "dataset": "AmazonHelp",
        "golden_examples": len(labeled),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "majority_baseline_accuracy": float(majority_accuracy),
        "majority_baseline_macro_f1": float(majority_f1),
        "majority_class": majority_class,
        "per_intent": report,
    }

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    )

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Failure analysis
    failures = labeled[
        labeled["human_label"] != labeled["predicted_intent"]
    ].copy()

    failure_file = os.path.join(
        OUTPUT_DIR,
        "failure_analysis.csv"
    )

    failures.to_csv(
        failure_file,
        index=False
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(metrics_file)
    print(failure_file)

    print()
    print("EVALUATION COMPLETE")


def main():
    if not os.path.exists(GOLDEN_FILE):
        create_golden_set()
        return

    # IMPORTANT:
    # Never recreate the golden set if it already exists.
    # This preserves human labels.
    print(f"Existing golden set found: {GOLDEN_FILE}")
    print("Human labels will NOT be overwritten.")

    evaluate()


if __name__ == "__main__":
    main()