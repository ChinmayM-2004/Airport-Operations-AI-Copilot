# ✈️ Airport Operations AI Copilot

## Uber Global Airport Operations & Supply Disruption Resolver

An AI-powered Airport Operations Copilot that investigates airport operational issues, retrieves applicable policies, analyzes operational data, generates recommendations, applies safety guardrails, and requests human approval before executing sensitive operational actions.

The project combines:

* Generative AI
* Retrieval-Augmented Generation (RAG)
* Vector search
* Multi-agent orchestration
* ReAct reasoning
* Function calling
* Operational analytics
* Safety guardrails
* Human-in-the-loop approval
* Audit logging
* Conversation memory
* Streamlit

---

## 🎯 Project Objective

The goal is to build an end-to-end AI Copilot capable of supporting airport operations teams when they need to investigate supply or service disruptions.

The complete workflow is:

```text
User Query
    ↓
Understand
    ↓
Retrieve
    ↓
Investigate
    ↓
Reason
    ↓
Recommend
    ↓
Validate
    ↓
Approve
    ↓
Execute
```

The system is designed to ensure that operational actions are not executed blindly by an AI system.

High-risk actions are validated through safety guardrails and require human approval before execution.

---

## 🏗️ System Architecture

```text
                         USER
                           |
                           v
                  ORCHESTRATOR AGENT
                           |
            +--------------+--------------+
            |              |              |
            v              v              v
      OPERATIONS      POLICY &        RESOLUTION
      INVESTIGATOR    COMPLIANCE        AGENT
            |              |              |
            v              v              v
        OPERATIONS        RAG         RECOMMENDATION
           TOOLS        KNOWLEDGE
                          BASE
            \              |              /
             \             |             /
              +------------+------------+
                           |
                           v
                    SAFETY GUARDRAILS
                           |
                           v
                   HUMAN APPROVAL
                           |
                  +--------+--------+
                  |                 |
               APPROVE           REJECT
                  |                 |
                  v                 v
               EXECUTE             STOP
```

---

## 🤖 Core AI Agents

### 1. Operations Investigator

Responsible for understanding the current operational situation.

It:

* Detects the airport from the user query
* Retrieves current operational metrics
* Analyzes supply and demand conditions
* Determines operational severity
* Identifies contributing factors
* Produces structured investigation findings

Example metrics include:

* Completion rate
* Average ETA
* Active drivers
* Driver cancellation rate
* Queue size
* Surge multiplier
* Request volume

### 2. Policy & Compliance Agent

Responsible for retrieving and interpreting applicable airport policies.

It uses:

* Document retrieval
* Semantic search
* FAISS vector search
* Sentence Transformer embeddings
* RAG
* Gemini-based policy interpretation
* Deterministic fallback logic

The agent determines whether a requested operational action is:

* Allowed
* Not allowed
* Requires additional approval
* Missing sufficient policy information

### 3. Resolution Agent

The Resolution Agent converts investigation and policy findings into an operational recommendation.

It considers:

* Current airport conditions
* Contributing factors
* Retrieved policies
* Requested operational changes
* Safety constraints
* Approval requirements

Example recommendation:

```text
Increase SFO surge multiplier from 1.26x to 1.40x.
```

The recommendation is not automatically executed when the action is considered high-risk.

### 4. Orchestrator Agent

The Orchestrator coordinates the complete workflow.

It manages:

* Agent handoffs
* Tool execution
* Policy retrieval
* ReAct reasoning
* Safety validation
* Human approval
* Execution
* Conversation memory
* Final response generation

The ReAct workflow is controlled with a maximum iteration limit to prevent uncontrolled agent loops.

---

## 📚 Retrieval-Augmented Generation

The project contains a synthetic airport policy knowledge base covering:

* SFO
* LAX
* JFK

The knowledge base contains **7 policy documents** and **65 policy chunks**.

### RAG Pipeline

```text
Policy Documents
       ↓
Document Loading
       ↓
Cleaning
       ↓
Section-Aware Chunking
       ↓
Embeddings
       ↓
FAISS Vector Index
       ↓
Semantic Retrieval
       ↓
Relevant Policy Evidence
       ↓
LLM Policy Interpretation
```

### Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

with:

```text
384-dimensional embeddings
```

The embeddings are normalized before being stored in the FAISS index.

### Retrieval Validation

The policy retrieval system was tested against 5 representative questions.

```text
5 / 5 appropriate policy retrievals
```

---

## 📄 Airport Policy Knowledge Base

The policy documents cover operational and pricing rules for:

```text
SFO
├── Operations
├── Pricing
└── Driver Policy

LAX
├── Operations
└── Pricing

JFK
├── Operations
└── Pricing
```

Examples of policy rules include:

* Airport-specific surge thresholds
* Driver staging rules
* Operational restrictions
* Approval requirements

---

## 📊 Operational Data

Synthetic airport telemetry is generated for:

* SFO
* LAX
* JFK

The dataset contains:

* **72 records**
* **3 airports**
* **24 hourly records per airport**
* **9 fields**

### Operational Dataset

| Field                      | Description                    |
| -------------------------- | ------------------------------ |
| `airport_code`             | Airport identifier             |
| `completion_rate`          | Trip completion rate           |
| `average_eta`              | Average estimated arrival time |
| `active_drivers`           | Number of active drivers       |
| `driver_cancellation_rate` | Driver cancellation rate       |
| `queue_size`               | Current queue size             |
| `surge_multiplier`         | Current surge multiplier       |
| `request_volume`           | Number of ride requests        |
| `timestamp`                | Telemetry timestamp            |

---

## 🛠️ Tools & Function Calling

The system provides operational tools that can be invoked by the AI agents.

### Get Airport Metrics

```python
get_airport_metrics(airport_code)
```

Retrieves the latest operational metrics for an airport.

### Calculate Driver Incentive

```python
calculate_driver_incentive(driver_count, severity_level)
```

Calculates a recommended driver incentive based on operational severity.

### Trigger Surge Override

```python
trigger_surge_override(airport_code, new_multiplier, reason)
```

A mock operational execution tool used to demonstrate safe action execution.

Tool calls use structured inputs and outputs with validation and error handling.

---

## 🧠 ReAct Reasoning

The system uses a controlled ReAct-style workflow:

```text
User Query
    ↓
Thought / Investigation
    ↓
Tool Call
    ↓
Observation
    ↓
Policy Retrieval
    ↓
Reasoning
    ↓
Recommendation
```

The ReAct controller has a maximum iteration limit of:

```text
5 iterations
```

This prevents uncontrolled reasoning loops.

---

## 🛡️ AI Safety & Guardrails

Safety is a core component of the application.

The system validates:

* Airport codes
* Surge multipliers
* Driver counts
* Incentive amounts
* Severity levels
* Required tool parameters
* AI recommendations
* Policy compliance

Invalid requests are blocked before execution.

Examples:

```text
airport_code = UNKNOWN
surge_multiplier = 100
driver_count = -50
```

are rejected by the input validation layer.

---

## ⚠️ Risk-Based Action Classification

Operational actions are classified based on risk.

| Action                    | Risk   | Approval    |
| ------------------------- | ------ | ----------- |
| Read airport metrics      | Low    | No          |
| Search policy             | Low    | No          |
| Calculate incentive       | Medium | Conditional |
| Increase surge below 1.3x | Medium | Yes         |
| Increase surge ≥ 1.3x     | High   | Yes         |
| High-value incentive      | High   | Yes         |

The thresholds are project-level safety assumptions used for demonstration.

---

## 🔐 Policy Guardrails

AI recommendations are checked against retrieved policy evidence before execution.

For example, if SFO policy permits standard surge only up to:

```text
1.5x
```

and the AI recommends:

```text
2.0x
```

the recommendation is blocked as a policy violation.

The AI cannot bypass policy constraints simply because a user requests the action.

---

## 👤 Human-in-the-Loop

High-risk actions require explicit human approval.

The execution lifecycle is:

```text
AI Recommendation
        ↓
Guardrail Check
        ↓
High-Risk Action?
        ↓
      YES
        ↓
Human Approval
     ↙     ↘
 APPROVE   REJECT
    ↓         ↓
 EXECUTE     STOP
```

Before approval:

```text
execution_status = awaiting_approval
```

After approval:

```text
execution_status = executed
```

If rejected:

```text
execution_status = rejected
```

This ensures that sensitive operational actions are not executed automatically.

---

## 🧾 Audit Logging

The system maintains an audit trail for operational decisions.

Audit information includes:

* User request
* Agents invoked
* Tools called
* Retrieved policy evidence
* Recommendation
* Risk level
* Approval requirement
* Approval decision
* Final action
* Execution result

Audit records are stored in:

```text
data/audit_logs/audit_trail.jsonl
```

---

## 🧠 Conversation Memory

The application includes short-term conversation memory.

The memory allows the system to:

* Remember previous investigations
* Resolve airport context from previous messages
* Answer follow-up questions
* Recall previous recommendations
* Avoid unnecessarily rerunning investigations

Example:

```text
User:
Investigate SFO and increase surge to 1.4x.

Assistant:
Increase surge multiplier to 1.4x.
Execution status: awaiting approval.

User:
What was the surge recommendation from our previous investigation?

Assistant:
Increase surge multiplier to 1.4x.
```

Memory recall is handled without unnecessarily executing a new investigation.

---

## 🧪 Distilled Training Data

Successful AI interactions are stored as structured JSONL records.

The distilled dataset captures useful examples such as:

* User request
* Investigation
* Retrieved policies
* Recommendation
* Safety decision
* Approval outcome
* Execution result

Fine-tuning is not required for the current implementation.

The dataset can be used as a foundation for future model improvement.

---

## 🖥️ Streamlit Application

The project includes a Streamlit-based user interface.

The application provides:

* Natural-language airport queries
* Operational dashboard
* Agent activity
* Investigation results
* RAG policy evidence
* Recommendations
* Safety status
* Human approval workflow
* Conversation memory
* Execution results
* Audit information

### Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
airport_ai_copilot/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── airport_policies/
│   │   ├── jfk_operations.md
│   │   ├── jfk_pricing.md
│   │   ├── lax_operations.md
│   │   ├── lax_pricing.md
│   │   ├── sfo_driver_policy.md
│   │   ├── sfo_operations.md
│   │   └── sfo_pricing.md
│   │
│   ├── operational_data/
│   │   └── airport_telemetry.csv
│   │
│   ├── audit_logs/
│   │   └── audit_trail.jsonl
│   │
│   ├── distilled_training_data.jsonl
│   └── policy_data_dictionary.md
│
├── notebooks/
│   └── day2_function_calling_demo.md
│
├── src/
│   ├── agents/
│   │   ├── investigator.py
│   │   ├── orchestrator.py
│   │   ├── policy_agent.py
│   │   ├── react_controller.py
│   │   ├── resolution_agent.py
│   │   └── state.py
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── audit_logger.py
│   ├── distillation_logger.py
│   ├── document_chunker.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── function_calling.py
│   ├── generate_operational_data.py
│   ├── guardrails.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── safe_executor.py
│   ├── tools.py
│   └── vector_store.py
│
└── tests/
    └── __init__.py
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ChinmayM-2004/Airport-Operations-AI-Copilot.git
cd Airport-Operations-AI-Copilot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 🔍 Example Queries

The application supports natural-language operational questions such as:

```text
What's happening at SFO right now?
```

```text
Does the current surge require approval?
```

```text
Investigate SFO and increase the surge multiplier to 1.4x.
```

```text
Investigate JFK and determine whether increasing surge to 2.0x is allowed.
```

```text
What was the surge recommendation from our previous investigation?
```

---

## 🧪 Validation

The project includes validation across the major components.

### RAG

```text
5 / 5 policy retrieval tests passed
```

### Operational Tools

```text
Tool validation passed
```

### Function Calling

```text
Function calling workflow validated
```

### Guardrails

```text
12 / 12 guardrail tests passed
```

### Safe Executor

```text
Safe executor integration tests passed
```

### Agent Orchestration

```text
Multi-agent workflow validated
```

### Human Approval

```text
High-risk actions correctly wait for human approval
```

### Memory

```text
Previous investigation recommendations can be recalled
```

---

## 🗓️ Development Roadmap

### Day 1 — Project Setup & RAG

Completed:

* Project structure
* Git setup
* Airport policy documents
* Prompt engineering
* Document loading
* Chunking
* Embeddings
* FAISS vector search
* Semantic retrieval
* RAG pipeline
* Policy validation

### Day 2 — Operational Data & Tools

Completed:

* Synthetic airport telemetry
* Data validation
* Operational tools
* Structured tool inputs and outputs
* Function calling
* Tool error handling

### Day 3 — Multi-Agent Copilot

Completed:

* Operations Investigator
* Policy & Compliance Agent
* Resolution Agent
* Orchestrator
* ReAct reasoning
* Agent handoffs
* Tool integration
* RAG integration
* Conversation memory

### Day 4 — Safety & Human-in-the-Loop

Completed:

* Risk classification
* Input guardrails
* Output validation
* Policy guardrails
* Human approval workflow
* Safe execution
* Audit logging
* Distilled training data

### Day 5 — Application

Completed:

* Streamlit application
* End-to-end workflow
* Operational dashboard
* Agent activity
* Safety integration
* Memory integration
* Final validation
* GitHub deployment

---

## 🔒 Safety Principles

The system follows several core safety principles:

1. Never invent operational metrics.
2. Use retrieved policy evidence for policy decisions.
3. Validate AI recommendations before execution.
4. Block policy-violating actions.
5. Require human approval for high-risk actions.
6. Never treat pending approval as rejection.
7. Maintain an audit trail for operational decisions.
8. Use conversation memory for contextual follow-ups.
9. Keep operational execution behind explicit safety checks.

---

## 🚀 Future Improvements

Potential future improvements include:

* Real airport telemetry integration
* Real-time streaming data
* Production vector database
* More advanced agent planning
* Role-based approval workflows
* Production authentication
* Persistent enterprise memory
* Automated monitoring and alerting
* Cloud deployment
* More sophisticated simulation environments
* Evaluation datasets for agent performance
* Automated regression testing
* Production-grade observability

---

## 📌 Project Status

| Day    | Component                           | Status     |
| ------ | ----------------------------------- | ---------- |
| Day 1  | RAG & Policy Knowledge Base         | ✅ Complete |
| Day 2  | Operational Data & Function Calling | ✅ Complete |
| Day 3  | Multi-Agent AI Copilot              | ✅ Complete |
| Day 4  | Safety & Human-in-the-Loop          | ✅ Complete |
| Day 5  | Streamlit Application               | ✅ Complete |
| GitHub | Repository Published                | ✅ Complete |

The project demonstrates an end-to-end **Generative AI + RAG + Agentic AI + Tool Calling + Safety + Human-in-the-Loop** workflow for airport operations.

---

## 👨‍💻 Author

**Chinmay M**

GitHub:

https://github.com/ChinmayM-2004
