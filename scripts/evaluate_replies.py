import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import retrieve_evidence

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOLDEN_FILE = "data/golden/golden_set.csv"
OUTPUT_FILE = "outputs/reply_quality.csv"

def get_evidence_text(message):
    evidence = retrieve_evidence(message)

    if isinstance(evidence, dict):
        return str(evidence)

    if isinstance(evidence, list):
        parts = []
        for item in evidence[:3]:
            parts.append(str(item))
        return "\n".join(parts)

    return str(evidence)


def generate_reply(message, evidence):
    prompt = f"""
You are an AI customer support agent for Amazon.

Customer message:
{message}

Historical support evidence:
{evidence}

Draft a concise customer-support reply.

Rules:
- Use the historical evidence as grounding.
- Do not invent policies, refunds, dates, prices, or guarantees.
- Be polite and professional.
- If the evidence is insufficient, clearly say the issue needs human support.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text.strip()


def judge_reply(message, reply, evidence):
    prompt = f"""
Evaluate this customer-support reply.

CUSTOMER:
{message}

REPLY:
{reply}

HISTORICAL EVIDENCE:
{evidence}

Score each criterion from 1 to 5:

Groundedness:
Does the reply stay supported by the historical evidence?

Relevance:
Does it directly address the customer's issue?

Correctness:
Is the response factually appropriate based on the evidence?

Helpfulness:
Would this reasonably help the customer move forward?

Brand consistency:
Is it professional and suitable for a customer-support interaction?

Return ONLY valid JSON:

{{
  "groundedness": 1,
  "relevance": 1,
  "correctness": 1,
  "helpfulness": 1,
  "brand_consistency": 1,
  "overall": 1,
  "reason": "short explanation"
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "groundedness": 0,
            "relevance": 0,
            "correctness": 0,
            "helpfulness": 0,
            "brand_consistency": 0,
            "overall": 0,
            "reason": text
        }


def main():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - REPLY QUALITY EVALUATION")
    print("=" * 70)

    df = pd.read_csv(GOLDEN_FILE)

    # Evaluate 30 examples to keep runtime/cost low.
    sample = df.head(30).copy()

    results = []

    for i, row in sample.iterrows():

        message = str(row["customer_message"])

        print(f"Processing {len(results) + 1}/30")

        try:
            evidence = get_evidence_text(message)
            reply = generate_reply(message, evidence)
            judge = judge_reply(message, reply, evidence)

            results.append({
                "customer_message": message,
                "intent": row.get("human_label", ""),
                "reply": reply,
                "evidence": evidence,
                "groundedness": judge.get("groundedness", 0),
                "relevance": judge.get("relevance", 0),
                "correctness": judge.get("correctness", 0),
                "helpfulness": judge.get("helpfulness", 0),
                "brand_consistency": judge.get("brand_consistency", 0),
                "overall": judge.get("overall", 0),
                "judge_reason": judge.get("reason", "")
            })

        except Exception as e:
            print("Error:", e)

    result_df = pd.DataFrame(results)

    os.makedirs("outputs", exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("REPLY QUALITY COMPLETE")
    print("=" * 70)

    if len(result_df) > 0:
        print("Examples evaluated:", len(result_df))
        print()
        print("Average scores:")
        print("Groundedness:", round(result_df["groundedness"].mean(), 2))
        print("Relevance:", round(result_df["relevance"].mean(), 2))
        print("Correctness:", round(result_df["correctness"].mean(), 2))
        print("Helpfulness:", round(result_df["helpfulness"].mean(), 2))
        print("Brand consistency:", round(result_df["brand_consistency"].mean(), 2))
        print("Overall:", round(result_df["overall"].mean(), 2))

    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()