import os
import sys
import pandas as pd

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_FILE = os.path.join(
    ROOT, "data", "golden", "golden_set.csv"
)

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


def show_intents():
    print("\nAvailable intents:")
    for i, intent in enumerate(INTENTS, 1):
        print(f"  {i}. {intent}")


def main():
    if not os.path.exists(GOLDEN_FILE):
        print("ERROR: golden_set.csv was not found.")
        print(f"Expected: {GOLDEN_FILE}")
        return

    df = pd.read_csv(GOLDEN_FILE)

    if "human_label" not in df.columns:
        df["human_label"] = ""

    # Convert NaN to empty strings
    df["human_label"] = df["human_label"].fillna("").astype(str)

    remaining = [
        i for i in range(len(df))
        if not df.loc[i, "human_label"].strip()
    ]

    print("=" * 70)
    print("HIVER GOLDEN SET - HUMAN LABELING")
    print("=" * 70)
    print(f"Total examples : {len(df)}")
    print(f"Already labeled: {len(df) - len(remaining)}")
    print(f"Remaining      : {len(remaining)}")

    print("\nHow to label:")
    print("  ENTER = accept the predicted intent")
    print("  1-10  = choose a different intent")
    print("  q     = save and quit")
    print("  s     = skip this example")
    print("=" * 70)

    show_intents()

    for position, idx in enumerate(remaining, 1):
        row = df.loc[idx]

        message = str(row.get("customer_message", ""))
        predicted = str(row.get("predicted_intent", "general_support"))
        confidence = row.get("confidence", "")

        print("\n" + "=" * 70)
        print(f"EXAMPLE {position}/{len(remaining)}")
        print("=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print(message)

        print("\nMODEL PREDICTION:")
        print(f"Intent     : {predicted}")
        print(f"Confidence: {confidence}")

        print("\nChoose:")
        print("  ENTER = accept prediction")
        print("  1-10  = correct/change label")
        print("  s     = skip")
        print("  q     = save & quit")

        choice = input("\nYour label: ").strip().lower()

        # Quit
        if choice == "q":
            df.to_csv(GOLDEN_FILE, index=False)
            print("\nProgress saved.")
            print(f"File: {GOLDEN_FILE}")
            break

        # Skip
        if choice == "s":
            print("Skipped.")
            continue

        # Accept prediction
        if choice == "":
            if predicted in INTENTS:
                df.loc[idx, "human_label"] = predicted
            else:
                print("Prediction is not a valid intent. Please choose 1-10.")
                continue

        # Choose numbered intent
        elif choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            selected = INTENTS[int(choice) - 1]
            df.loc[idx, "human_label"] = selected
            print(f"Human label: {selected}")

        else:
            print("Invalid input. Use ENTER, 1-10, s, or q.")
            continue

        # Save after EVERY label
        df.to_csv(GOLDEN_FILE, index=False)

        labeled_count = df["human_label"].str.strip().ne("").sum()
        print(f"Saved. Progress: {labeled_count}/{len(df)}")

    else:
        print("\nAll examples have been reviewed.")

    # Final status
    df = pd.read_csv(GOLDEN_FILE)
    df["human_label"] = df["human_label"].fillna("").astype(str)

    labeled = df["human_label"].str.strip().ne("").sum()

    print("\n" + "=" * 70)
    print("LABELING STATUS")
    print("=" * 70)
    print(f"Labeled: {labeled}/{len(df)}")
    print(f"Remaining: {len(df) - labeled}")
    print("=" * 70)


if __name__ == "__main__":
    main()