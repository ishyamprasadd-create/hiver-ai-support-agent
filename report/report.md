# Hiver SDE Intern Take-Home Assignment
## AI Support Agent — AmazonHelp

## 1. Framing

The goal of this project is to build a lightweight AI customer-support agent for one brand from the Customer Support on Twitter dataset.

I selected AmazonHelp and built an end-to-end prototype that:

1. classifies an incoming customer message into a support intent;
2. retrieves similar historical AmazonHelp interactions;
3. uses historical responses as supporting evidence;
4. determines whether the request can be auto-handled or should be escalated.

The prototype prioritizes traceability and reproducibility over production-scale infrastructure.

---

## 2. What Good Means

A useful support agent should:

- identify the customer's primary intent;
- provide a response grounded in historical support resolutions;
- avoid unsupported claims;
- expose the evidence used for the response;
- escalate uncertain requests rather than confidently responding incorrectly.

For this prototype, success is evaluated separately for intent classification and escalation behavior.

---

## 3. Dataset and Sampling

The source dataset is:

`thoughtvector/customer-support-on-twitter`

The selected brand is:

`AmazonHelp`

A deterministic sample of 200 customer messages was created for evaluation.

Each example was manually reviewed and assigned a human intent label.

Golden set:

`data/golden/golden_set.csv`

Total manually reviewed examples:

**200**

---

## 4. Intent Taxonomy

The prototype uses nine support intents:

- account_issue
- delivery_issue
- general_support
- order_issue
- payment_issue
- price_question
- product_issue
- refund_request
- subscription_issue

The taxonomy was kept intentionally small so that the prototype could provide interpretable routing decisions.

---

## 5. Architecture

```text
Incoming Customer Message
          |
          v
Intent Classification
          |
          v
Confidence Estimation
          |
          v
Historical Interaction Retrieval
          |
          v
Evidence Assessment
          |
       +--+--+
       |     |
       v     v
 Strong    Weak/No
 Evidence  Evidence
       |     |
       v     v
 Auto      Escalate
 Handle
       |
       v
Grounded Response