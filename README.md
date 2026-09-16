\# Hiver AI Support Agent



An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment.



\## Overview



This project builds an AI support workflow for Amazon Help customer-support conversations from the Customer Support on Twitter dataset.



The system:



\- Classifies incoming customer messages into support intents.

\- Retrieves similar historical customer-support conversations.

\- Uses historical resolutions as evidence for drafting replies.

\- Decides whether a case should be automatically handled or escalated.

\- Provides a reason for the escalation decision.

\- Includes a manually labelled 200-example golden evaluation set.

\- Includes evaluation scripts for intent classification and escalation.



\## Problem Statement



Customer support teams receive a large number of customer messages. The goal of this project is to build a lightweight AI support agent that can understand incoming messages, identify their intent, use historical support conversations as evidence, draft an appropriate response, and determine whether the case should be handled automatically or escalated to a human.



\## Selected Brand



\*\*Amazon Help (`AmazonHelp`)\*\*



The project uses Amazon Help conversations from the Customer Support on Twitter dataset.



A sampled subset of the dataset is used for development and evaluation. The complete raw dataset is not included in GitHub because of GitHub file-size limitations.



\## System Architecture



```text

Customer Message

&#x20;      |

&#x20;      v

Intent Classification

&#x20;      |

&#x20;      v

Historical Evidence Retrieval

&#x20;      |

&#x20;      v

Draft Reply Generation

&#x20;      |

&#x20;      v

Escalation Decision

&#x20;      |

&#x20;      v

Final Support Response

