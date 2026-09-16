import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "amazonhelp_pairs.csv"

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset file not found:\n{DATA_FILE}\n\n"
        "Run: python scripts\\prepare_data.py"
    )

df = pd.read_csv(DATA_FILE)

df = df.fillna("")

print(f"Loaded {len(df):,} historical AmazonHelp conversations.")


# ---------------------------------------------------------
# INTENT TAXONOMY
# ---------------------------------------------------------

INTENT_KEYWORDS = {
    "delivery_issue": [
        "delivery",
        "delivered",
        "delivery date",
        "late",
        "delayed",
        "package",
        "parcel",
        "courier",
        "shipment",
        "shipping",
        "arrive",
        "arrived",
        "where is my package",
        "not arrived",
        "missing package",
        "tracking",
    ],

    "refund_request": [
        "refund",
        "money back",
        "give my money back",
        "return money",
        "reimburse",
        "reimbursement",
        "refunded",
        "refund me",
    ],

    "order_issue": [
        "order",
        "ordered",
        "cancel order",
        "cancel my order",
        "wrong order",
        "order status",
        "order number",
        "purchase",
        "item i ordered",
    ],

    "payment_billing": [
        "payment",
        "paid",
        "charge",
        "charged",
        "charged twice",
        "double charge",
        "billing",
        "bill",
        "card",
        "credit card",
        "debit card",
        "transaction",
        "payment failed",
        "payment issue",
        "money deducted",
    ],

    "account_issue": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "username",
        "locked account",
        "account locked",
        "cannot access",
        "can't access",
    ],

    "subscription_issue": [
        "prime",
        "membership",
        "subscription",
        "subscribe",
        "unsubscribe",
        "prime membership",
        "prime video",
        "free month",
        "renewal",
        "renew",
    ],

    "product_issue": [
        "product",
        "item",
        "device",
        "broken",
        "damaged",
        "defective",
        "not working",
        "doesn't work",
        "does not work",
        "quality",
        "wrong item",
    ],

    "price_question": [
        "price",
        "cost",
        "expensive",
        "discount",
        "offer",
        "deal",
        "sale",
        "coupon",
        "promo",
        "promotion",
        "cheaper",
    ],

    "general_support": [
        "help",
        "question",
        "information",
        "please help",
        "support",
        "amazon help",
        "what can i do",
    ],
}


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize customer text for keyword matching and retrieval.
    """
    if text is None:
        return ""

    text = str(text).lower()

    # Decode a few common HTML entities
    text = text.replace("&amp;", " and ")
    text = text.replace("&quot;", " ")
    text = text.replace("&apos;", "'")
    text = text.replace("&nbsp;", " ")

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter handles
    text = re.sub(r"@\w+", " ", text)

    # Keep alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

def tokenize(text):
    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


# ---------------------------------------------------------
# INTENT CLASSIFICATION
# ---------------------------------------------------------

def classify_intent(message):
    """
    Lightweight deterministic intent classifier.

    Returns:
        intent, confidence
    """

    text = normalize_text(message)

    if not text:
        return "general_support", 0.0

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():

        score = 0

        for keyword in keywords:
            keyword_normalized = normalize_text(keyword)

            if not keyword_normalized:
                continue

            if keyword_normalized in text:
                # Multi-word phrases get slightly higher weight
                if " " in keyword_normalized:
                    score += 2
                else:
                    score += 1

        scores[intent] = score

    # -----------------------------------------------------
    # SPECIAL CASES
    # -----------------------------------------------------

    # Duplicate charge / payment is clearly billing related
    if (
        "charged twice" in text
        or "double charge" in text
        or "charged two times" in text
        or "charged 2 times" in text
    ):
        return "payment_billing", 0.85

    # Refund wording
    if "refund" in text or "money back" in text:
        return "refund_request", 0.80

    # Delivery wording
    if (
        "package not arrived" in text
        or "package hasn't arrived" in text
        or "package has not arrived" in text
        or "where is my package" in text
    ):
        return "delivery_issue", 0.80

    # Find highest score
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # No keyword matched
    if best_score == 0:
        return "general_support", 0.40

    # Calculate a bounded confidence
    confidence = min(0.40 + (best_score * 0.10), 0.95)

    # If multiple intents have the same score, reduce confidence
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if len(sorted_scores) > 1:
        if sorted_scores[0][1] == sorted_scores[1][1]:
            confidence = max(0.35, confidence - 0.15)

    return best_intent, round(confidence, 3)


# ---------------------------------------------------------
# JACCARD SIMILARITY
# ---------------------------------------------------------

def similarity(a, b):
    """
    Simple lexical Jaccard similarity.
    """

    a_words = tokenize(a)
    b_words = tokenize(b)

    if not a_words or not b_words:
        return 0.0

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    if union == 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------
# HISTORICAL RETRIEVAL
# ---------------------------------------------------------

def retrieve_evidence(message, top_k=3):
    """
    Retrieve the strongest historical AmazonHelp example.

    IMPORTANT:
    Returns a dictionary so evaluation scripts can safely
    use evidence.get(...).
    """

    scores = []

    for _, row in df.iterrows():

        customer_message = str(
            row.get("customer_message", "")
        )

        historical_response = str(
            row.get("historical_response", "")
        )

        score = similarity(message, customer_message)

        if score > 0:
            scores.append(
                {
                    "similarity": round(score, 3),
                    "customer_example": customer_message,
                    "historical_response": historical_response,
                }
            )

    if not scores:
        return {
            "similarity": 0.0,
            "customer_example": "",
            "historical_response": "",
            "examples": [],
        }

    scores.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    top_examples = scores[:top_k]

    best = top_examples[0]

    return {
        "similarity": best["similarity"],
        "customer_example": best["customer_example"],
        "historical_response": best["historical_response"],
        "examples": top_examples,
    }


# ---------------------------------------------------------
# ESCALATION DECISION
# ---------------------------------------------------------

def decide_escalation(intent, confidence, evidence):
    """
    Decide AUTO_HANDLE vs ESCALATE.
    """

    evidence_similarity = 0.0

    if isinstance(evidence, dict):
        evidence_similarity = float(
            evidence.get("similarity", 0.0)
        )

    # Sensitive payment/billing cases
    if intent == "payment_billing":
        return (
            "ESCALATE",
            "Human review recommended for sensitive payment_billing case."
        )

    # Low confidence
    if confidence < 0.60:
        return (
            "ESCALATE",
            "Human review recommended because intent confidence is below the safe threshold."
        )

    # No useful evidence
    if evidence_similarity <= 0:
        return (
            "ESCALATE",
            "Human review recommended because no historical support evidence was found."
        )

    # Weak evidence
    if evidence_similarity < 0.08:
        return (
            "ESCALATE",
            "Human review recommended because the best historical match is weak."
        )

    return (
        "AUTO_HANDLE",
        "Sufficient confidence and historical evidence."
    )


# ---------------------------------------------------------
# OPENAI RESPONSE GENERATION
# ---------------------------------------------------------

def generate_llm_reply(message, intent, evidence):
    """
    Generate a grounded reply using OpenAI when an API key
    is available.

    Falls back to historical response if unavailable.
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    historical_response = ""

    if isinstance(evidence, dict):
        historical_response = str(
            evidence.get("historical_response", "")
        )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if not api_key or api_key == "your_api_key_here":

        if historical_response:
            return historical_response

        return (
            "I understand your concern. Please check the relevant "
            "details in your Amazon account. If the issue continues, "
            "Amazon Support can review it further."
        )

    # -----------------------------------------------------
    # OPENAI
    # -----------------------------------------------------

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a customer support assistant for AmazonHelp.

Customer message:
{message}

Detected intent:
{intent}

Historical customer message:
{evidence.get("customer_example", "")}

Historical support response:
{historical_response}

Write a concise, polite customer-support reply.

Rules:
- Use the historical response as grounding evidence.
- Do not invent order details.
- Do not claim access to the customer's account.
- Do not invent refunds, delivery dates or transactions.
- If the issue requires account-specific information, ask the customer to contact support or provide the necessary information.
- Keep the answer professional and concise.
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        reply = getattr(response, "output_text", None)

        if reply:
            return reply.strip()

    except Exception:
        pass

    # -----------------------------------------------------
    # FINAL FALLBACK
    # -----------------------------------------------------

    if historical_response:
        return historical_response

    return (
        "I understand your concern. Please check the relevant "
        "details in your Amazon account. If the issue continues, "
        "Amazon Support can review it further."
    )


# ---------------------------------------------------------
# DRAFT REPLY
# ---------------------------------------------------------

def draft_reply(message, intent, evidence):
    """
    Public function used by evaluation scripts.
    """

    return generate_llm_reply(
        message,
        intent,
        evidence
    )


# ---------------------------------------------------------
# COMPLETE AGENT PIPELINE
# ---------------------------------------------------------

def run_agent(message):
    """
    Run the complete support-agent pipeline.
    """

    intent, confidence = classify_intent(message)

    evidence = retrieve_evidence(
        message,
        top_k=3
    )

    decision, reason = decide_escalation(
        intent,
        confidence,
        evidence
    )

    reply = draft_reply(
        message,
        intent,
        evidence
    )

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": confidence,
        "decision": decision,
        "reason": reason,
        "reply": reply,
        "evidence": evidence,
        "similarity": evidence.get("similarity", 0.0),
    }


# ---------------------------------------------------------
# INTERACTIVE TERMINAL
# ---------------------------------------------------------

def main():

    print()
    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 60)
    print("Brand: AmazonHelp")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key and api_key != "your_api_key_here":
        print("LLM: ENABLED")
    else:
        print("LLM: DISABLED - historical-response fallback")

    print("=" * 60)

    while True:

        try:
            message = input(
                "\nCustomer message (type 'exit' to stop): "
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not message:
            continue

        if message.lower() in {
            "exit",
            "quit",
            "q"
        }:
            print("Exiting.")
            break

        result = run_agent(message)

        intent = result["intent"]
        confidence = result["confidence"]
        decision = result["decision"]
        reason = result["reason"]
        reply = result["reply"]
        evidence = result["evidence"]

        print()
        print("=" * 60)

        print(f"\nCustomer: {message}")

        print("\n" + "=" * 60)

        print("INTENT")
        print(intent)

        print("\nINTENT CONFIDENCE")
        print(f"{confidence:.3f}")

        print("\nBEST HISTORICAL MATCH")

        if evidence:
            print(
                f"{evidence.get('similarity', 0.0):.3f}"
            )
        else:
            print("0.000")

        print("\nESCALATION DECISION")
        print(decision)

        print("\nREASON")
        print(reason)

        print("\nDRAFT REPLY")
        print("-" * 60)
        print(reply)

        print("\n" + "=" * 60)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()