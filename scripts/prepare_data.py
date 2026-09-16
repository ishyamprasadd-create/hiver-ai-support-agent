import pandas as pd
import os
import re

INPUT = r"data\raw\twcs\twcs\twcs.csv"
OUTPUT = r"data\processed\amazonhelp_pairs.csv"

BRAND = "AmazonHelp"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("=" * 60)
print("HIVER SUPPORT AGENT - DATA PREPARATION")
print("=" * 60)
print(f"Brand: {BRAND}")
print("Step 1/2: Finding AmazonHelp responses...")

# Store AmazonHelp tweet IDs and text
support = {}

for chunk in pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "created_at",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    chunksize=100000,
    dtype=str
):
    brand_rows = chunk[chunk["author_id"] == BRAND]

    for _, row in brand_rows.iterrows():
        support[str(row["tweet_id"])] = str(row["text"])

print(f"Found {len(support):,} AmazonHelp support messages.")

print("Step 2/2: Finding customer messages and matching responses...")

pairs = []

for chunk in pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "created_at",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    chunksize=100000,
    dtype=str
):
    # Customer/inbound messages
    customers = chunk[chunk["inbound"].astype(str).str.lower() == "true"]

    for _, row in customers.iterrows():

        response_ids = str(row["response_tweet_id"])

        if response_ids in ("nan", "None", ""):
            continue

        # Dataset can contain comma-separated response IDs
        ids = re.split(r"[, ]+", response_ids)

        for response_id in ids:
            response_id = response_id.strip()

            if response_id in support:
                pairs.append({
                    "customer_message": str(row["text"]),
                    "historical_response": support[response_id],
                    "customer_tweet_id": str(row["tweet_id"]),
                    "response_tweet_id": response_id
                })

                if len(pairs) >= 5000:
                    break

        if len(pairs) >= 5000:
            break

    if len(pairs) >= 5000:
        break

df = pd.DataFrame(pairs)

# Remove empty/duplicate messages
df = df.dropna(subset=["customer_message", "historical_response"])
df = df.drop_duplicates(subset=["customer_message"])

# Basic cleaning
df["customer_message"] = df["customer_message"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

df["historical_response"] = df["historical_response"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

df = df[
    (df["customer_message"].str.len() >= 10) &
    (df["historical_response"].str.len() >= 5)
]

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print(f"SUCCESS!")
print(f"Saved {len(df):,} customer/support pairs")
print(f"Output: {OUTPUT}")
print("=" * 60)

print("\nExample:")
print("CUSTOMER:", df.iloc[0]["customer_message"])
print("RESPONSE:", df.iloc[0]["historical_response"])