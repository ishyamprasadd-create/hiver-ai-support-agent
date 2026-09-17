AI-Powered Customer Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment, using the Customer Support on Twitter (TWCS) dataset. The system focuses on the AmazonHelp support brand and processes incoming customer messages by identifying intent, retrieving similar historical support conversations, generating a grounded draft response, and deciding whether a case can be auto-handled or should be escalated to a human.

---

Project Overview

Customer-support messages are often short, informal, and incomplete, making them difficult to classify through simple keyword matching alone. For example: "Why was I charged twice for my Amazon order?"

A useful support system should not only identify the issue, but also:

- Understand the customer's intent
- Find similar historical support cases
- Use previous resolutions as supporting evidence
- Generate a relevant support response
- Decide whether the response can be automatically handled
- Escalate sensitive or uncertain cases to a human
- Provide a reason for the escalation decision
- Evaluate system quality using a manually labelled golden set

This project implements that complete workflow for the AmazonHelp customer-support brand.

---

Problem Statement

The objective is to build an AI customer-support agent capable of handling incoming customer messages through three primary tasks:

Task 1 — Intent Classification
Classifies the customer's message into a predefined set of support intents.

Task 2 — Grounded Reply Generation
Retrieves historically similar AmazonHelp conversations and uses them to produce a support-oriented draft response.

Task 3 — Auto-Handle vs. Human Escalation
Determines whether the request can be automatically handled or should be reviewed by a human.

The final output includes:

- Customer message
- Predicted intent
- Intent confidence
- Historical similarity score
- Escalation decision and reason
- Draft reply
- Historical supporting evidence

---

Dataset

This project uses the Customer Support on Twitter (TWCS) dataset, which contains real customer-support conversations between users and companies on Twitter. The selected support brand is AmazonHelp.

Data Processing Pipeline:

1. Load the TWCS dataset
2. Filter conversations involving AmazonHelp
3. Group customer messages with corresponding support responses
4. Remove unnecessary information
5. Create customer–response pairs
6. Store the processed data for retrieval and evaluation

Processed data is stored at: data/processed/amazonhelp_pairs.csv

Note: The large raw TWCS dataset is intentionally excluded from GitHub due to file-size restrictions.

---

Support Intent Taxonomy

| Intent | Description |
|---|---|
| delivery_issue | Problems related to delayed, missing, or undelivered orders |
| refund_request | Requests or questions regarding refunds |
| order_issue | Problems with an order or received order |
| payment_billing | Payment, billing, duplicate charge, or transaction issues |
| account_issue | Problems involving the customer's account |
| subscription_issue | Issues involving subscriptions or recurring services |
| product_issue | Problems involving a product or product condition |
| price_question | Questions about product pricing or price changes |
| general_support | General support requests that do not fit another category |

The taxonomy is intentionally kept compact to provide actionable support categories rather than an excessive number of fine-grained labels.

---

System Architecture

The system follows this pipeline:

Customer Message
→ Intent Classifier
→ Confidence Score
→ Historical Retrieval (AmazonHelp Conversations)
→ Evidence Selection
→ Draft Reply Generation
→ Escalation Policy
→ Final Agent Output

This combines intent classification, historical retrieval, evidence-grounded response generation, escalation rules, and automated evaluation using human-labelled data.

---

Intent Classification

The first stage identifies the likely intent of an incoming customer message using lightweight lexical and rule-based signals designed for fast execution.

Examples:

Customer: "Why was I charged twice for my Amazon order?"
Predicted intent: payment_billing

Customer: "My package hasn't arrived yet."
Predicted intent: delivery_issue

Customer: "I returned my order but haven't received my refund."
Predicted intent: refund_request

The classifier also produces a confidence value, e.g.:

Intent: payment_billing
Intent Confidence: 0.850

This confidence score is later used as one of the signals in the escalation decision.

---

Historical Evidence Retrieval and Reply Generation

After identifying intent, the system searches historical AmazonHelp conversations for similar customer messages using lexical similarity.

Example:

Customer: "Why was I charged twice for my Amazon order?"
Historical customer message: "I'm seeing an unexpected charge on my account."
Historical response: "I'm sorry about the unexpected charge. We can't access your account from Twitter, but we can look into this with you here."

Best Historical Match: 0.569

The historical response is then used as evidence for drafting the final support response. If an OpenAI API key is configured, the system generates a more natural response using this retrieved evidence; otherwise, it falls back to a historical-response-based reply. This ensures the core workflow functions without requiring an external LLM service.

---

Escalation Decision

Not every generated response is sent automatically. The system determines whether a case should be AUTO_HANDLE or ESCALATE, based on:

Sensitive intent
Payment and billing issues are escalated for human review, as they may involve sensitive transaction or account details.

Low confidence
Cases with intent confidence below the configured threshold are escalated.

Weak historical evidence
If no useful historical evidence is found, the system escalates rather than generating an unsupported response.

Low similarity
If the best historical match has insufficient similarity, human review is recommended.

Example output:

Escalation Decision: ESCALATE
Reason: Human review recommended for sensitive payment_billing case.

This conservative approach reduces the risk of automatically responding to uncertain or sensitive customer requests.

---

Evaluation

The project includes a manually labelled golden evaluation set of 200 customer examples, each reviewed and assigned a human intent label. The evaluation pipeline measures intent-classification performance using accuracy, macro precision, macro recall, macro F1, weighted F1, per-intent performance, and failure analysis.

Intent Evaluation Results:

| Metric | Result |
|---|---:|
| Accuracy | 12.00% |
| Macro Precision | 16.71% |
| Macro Recall | 15.69% |
| Macro F1 | 9.84% |
| Weighted F1 | 10.48% |

These results are reported transparently. The low classification performance indicates that the lightweight lexical classifier is not yet robust enough for production-quality intent classification, and this evaluation is valuable precisely because it exposes where the current approach falls short rather than presenting an inflated result.

Baseline Comparisons:

Majority-Class Baseline — always predicts the most frequent intent, providing a trivial reference point.

TF-IDF + Logistic Regression — a traditional text-classification baseline offering a stronger classical comparison.

These baselines help quantify the improvement gained from moving beyond simple frequency-based and traditional text-classification approaches.

---

Limitations, Failure Modes, and Future Improvements

Current Limitations:

1. Ambiguous Customer Messages — Short messages (e.g., "Please help with my order.") may lack enough information to determine correct intent.
2. Multiple Issues in One Message — A single-intent classifier may struggle when customers describe more than one problem at once.
3. Lexical Classification Limitations — The classifier relies heavily on surface wording, so semantically similar messages phrased differently may be misclassified.
4. Historical Retrieval Mismatch — A textually similar historical response may not actually address the same underlying issue.
5. Historical Response Variability — Retrieved responses can vary in wording, completeness, and quality.

What the Headline Number Doesn't Capture:

Intent accuracy alone should not be treated as a complete measure of the support agent's overall quality. The golden set contains only 200 examples relative to the much larger source dataset, the classifier is intentionally lightweight, and class imbalance can affect aggregate metrics. Intent classification also does not directly measure reply quality, and results should not be interpreted as representative of production-level performance.

The purpose of this evaluation is to expose the current system's strengths and weaknesses and to provide a reproducible baseline for future improvement.

Planned Future Improvements:

1. Transformer-based intent classification
2. Sentence embeddings for semantic retrieval
3. FAISS or another vector database for large-scale retrieval
4. Improved conversation-level train/test splitting
5. Multi-intent classification
6. Confidence calibration
7. Improved escalation classification
8. Stronger grounding constraints for generated responses
9. Human evaluation of generated replies
10. More extensive error analysis
11. Better handling of multilingual customer messages
12. Production monitoring and feedback loops

---

Project Structure

hiver-ai-support-agent/
├── data/
│   ├── golden/
│   │   └── golden_set.csv
│   ├── processed/
│   │   └── amazonhelp_pairs.csv
│   └── raw/
│       └── twcs/
├── outputs/
│   ├── escalation_evaluation.csv
│   ├── failure_analysis.csv
│   ├── metrics.json
│   └── reply_quality.csv
├── report/
│   └── report.md
├── scripts/
│   ├── evaluate_escalation.py
│   ├── evaluate_replies.py
│   ├── label_golden_set.py
│   ├── prepare_data.py
│   └── run_evaluation.py
├── src/
│   └── agent.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md

---

Installation

1. Clone the Repository

git clone https://github.com/ishyamprasadd-create/hiver-ai-support-agent.git
cd hiver-ai-support-agent

2. Create a Virtual Environment

python -m venv .venv

Activate on Windows PowerShell:

.venv\Scripts\Activate.ps1

3. Install Dependencies

pip install -r requirements.txt

---

Running the Agent

python src/agent.py

The application will prompt for a customer message, for example:

"Why was I charged twice for my Amazon order?"

The system produces structured output similar to:

============================================================
Customer: Why was I charged twice for my Amazon order?
============================================================

Intent: payment_billing
Intent Confidence: 0.850
Best Historical Match: 0.569

Escalation Decision: ESCALATE
Reason: Human review recommended for sensitive payment_billing case.

Draft Reply:
------------------------------------------------------------
I understand your concern about the payment or charge. Please check the transaction details in your Amazon account. If the charge is still unclear, Amazon Support can review the transaction.

Similar Amazon support guidance:
I'm sorry about the unexpected charge. We can't access your account from Twitter but we can look into this with you here.

Human review is recommended before sending a final response, as this case may involve account, payment, or refund information.
============================================================

---

Running Evaluation

Run intent evaluation:
python scripts/run_evaluation.py

Run escalation evaluation:
python scripts/evaluate_escalation.py

Generated results are stored under the outputs/ directory.

---

Golden Evaluation Set

Location: data/golden/golden_set.csv
Size: 200 examples, manually reviewed and assigned human intent labels

This set is used to measure the difference between the system's predictions and human-labelled intent categories.

---

Configuration

The project can optionally use an OpenAI API key for LLM-based reply generation.

Create a local .env file:

OPENAI_API_KEY=your_api_key_here

Do not commit the .env file to GitHub. The repository's .gitignore excludes secrets and the large raw dataset.

---

Example Customer Queries

Delivery: "My package hasn't arrived yet. Where is my order?"
Refund: "I returned my order but haven't received my refund."
Payment: "Why was I charged twice for my Amazon order?"
Order: "I received the wrong item in my order."
Product: "The product I received is damaged."
Account: "I can't log into my Amazon account."
Price: "Why did the price of this product change?"

---

Key Design Decisions

1. AmazonHelp was selected as the support brand.
2. A small intent taxonomy was used instead of an excessive number of categories.
3. Historical customer-support conversations are used as grounding evidence.
4. Sensitive payment-related cases are escalated.
5. Low-confidence predictions are escalated.
6. Weak retrieval evidence triggers human review.
7. A 200-example human-labelled golden set is used for evaluation.
8. Simple baselines are included for comparison.
9. Failure analysis is included rather than reporting aggregate accuracy alone.
10. The large raw dataset is excluded from GitHub due to repository file-size constraints.
11. API secrets are excluded from version control.
12. The system is designed as a lightweight prototype that can later be upgraded with transformer models and semantic retrieval.

---

What This Project Demonstrates

- Natural Language Processing
- Text classification
- Customer-support automation
- Information retrieval and historical conversation retrieval
- Evidence-grounded response generation
- Human-in-the-loop escalation
- Evaluation dataset creation and model evaluation
- Error analysis
- Python, Pandas, Scikit-learn
- OpenAI API integration
- Git and GitHub

---

Conclusion

This project provides an end-to-end prototype for an AI-powered customer-support agent. Rather than simply generating an answer, the system follows a complete support workflow:

Customer Message → Intent Classification → Historical Evidence Retrieval → Draft Support Reply → Escalation Decision → Final Structured Output

The current evaluation also highlights an important limitation: lightweight lexical intent classification is not yet sufficiently accurate for reliable production use. The architecture nonetheless provides a solid foundation for future improvement through semantic embeddings, transformer-based classifiers, stronger retrieval, better confidence calibration, and human feedback.

---

Author

Shyam Prasad S
B.Tech — Computer Engineering
Specialization: Artificial Intelligence & Data Science

GitHub: https://github.com/ishyamprasadd-create