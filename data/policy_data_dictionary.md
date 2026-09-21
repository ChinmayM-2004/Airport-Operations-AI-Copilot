# Airport Operations Policy Knowledge Base

## 1. Purpose

This document describes the synthetic policy knowledge base used by
the Uber Global Airport Operations & Supply Disruption Resolver
(Agentic Copilot).

The knowledge base is used by the Retrieval-Augmented Generation (RAG)
pipeline to retrieve airport-specific operational policies and provide
grounded answers to user questions.

All policies in this project are synthetic training policies created
specifically for this capstone project. They are not real Uber or
airport policies.

---

## 2. Airports Covered

The knowledge base currently covers three airports:

| Airport | Code | Location |
|---|---|---|
| San Francisco International Airport | SFO | San Francisco, California |
| Los Angeles International Airport | LAX | Los Angeles, California |
| John F. Kennedy International Airport | JFK | New York, New York |

---

## 3. Policy Documents

The knowledge base contains seven synthetic policy documents.

| File | Airport | Policy Type | Policy ID |
|---|---|---|---|
| `sfo_operations.md` | SFO | Airport Operations | SFO-OPS-001 |
| `sfo_pricing.md` | SFO | Pricing and Surge | SFO-PRC-001 |
| `sfo_driver_policy.md` | SFO | Driver Queue Policy | SFO-DRV-001 |
| `lax_operations.md` | LAX | Airport Operations | LAX-OPS-001 |
| `lax_pricing.md` | LAX | Pricing and Surge | LAX-PRC-001 |
| `jfk_operations.md` | JFK | Airport Operations | JFK-OPS-001 |
| `jfk_pricing.md` | JFK | Pricing and Surge | JFK-PRC-001 |

---

## 4. Policy Categories

### 4.1 Airport Operations

Airport operations policies define operational rules such as:

- Pickup locations
- Drop-off procedures
- Driver staging areas
- Airport queue requirements
- Temporary operational restrictions
- Operational release procedures
- Airport-specific restrictions

---

### 4.2 Pricing and Surge

Pricing policies define rules for surge pricing and approval requirements.

The current synthetic policies define the following standard maximum
surge multipliers without additional approval:

| Airport | Maximum Surge Without Additional Approval |
|---|---:|
| SFO | 1.5x |
| LAX | 1.8x |
| JFK | 2.0x |

Increasing surge above the airport-specific threshold requires
Airport Operations Manager approval.

---

### 4.3 Driver Queue Policy

The SFO driver policy defines rules related to the airport queue,
including:

- Designated staging requirements
- Queue position
- Leaving the staging area
- Voluntary queue abandonment
- Operational releases
- Maintaining queue priority

For example, voluntarily leaving the SFO airport queue results in loss
of queue position.

---

## 5. Important Policy Rules

### SFO Surge

The maximum standard surge multiplier permitted at SFO without
additional approval is:

**1.5x**

Any increase above 1.5x requires approval from the:

**Airport Operations Manager**

The approval request must include:

- Current demand conditions
- Available driver supply
- Estimated supply-demand imbalance
- Reason for the proposed increase
- Expected duration

---

### LAX Surge

The maximum surge multiplier permitted at LAX without additional
approval is:

**1.8x**

Any increase above 1.8x requires Airport Operations Manager approval.

---

### JFK Surge

The maximum surge multiplier permitted at JFK without additional
approval is:

**2.0x**

Any increase above 2.0x requires Airport Operations Manager approval.

---

### SFO Driver Queue

Drivers must use the designated airport staging area.

A driver who voluntarily leaves the airport queue loses their queue
position.

A driver may not abandon the queue while retaining their existing
queue position.

Operational releases initiated by authorized airport operations
processes may preserve queue priority according to the applicable
policy.

---

## 6. Document Metadata

Each policy document contains metadata used by the document ingestion
and retrieval pipeline.

| Field | Description |
|---|---|
| Policy ID | Unique identifier for the policy |
| Airport | Airport to which the policy applies |
| Policy Type | Category of the policy |
| Version | Version number of the policy |
| Effective Date | Date from which the policy applies |
| File Name | Original policy document name |
| Section | Section containing the retrieved information |

---

## 7. Chunk Metadata

After document chunking, each chunk contains the following metadata:

| Field | Description |
|---|---|
| `chunk_id` | Unique identifier for the chunk |
| `document_id` | Identifier of the source document |
| `file_name` | Source policy file |
| `airport` | Airport associated with the policy |
| `policy_id` | Source policy identifier |
| `policy_type` | Type of policy |
| `version` | Policy version |
| `effective_date` | Policy effective date |
| `section` | Source policy section |
| `text` | Actual policy text used for retrieval |

This metadata allows the RAG system to provide source information with
retrieved answers.

---

## 8. RAG Processing Pipeline

The policy knowledge base follows this processing pipeline:

```text
Policy Documents
       ↓
Document Loading
       ↓
Document Cleaning
       ↓
Section-Aware Chunking
       ↓
65 Policy Chunks
       ↓
Sentence Transformer Embeddings
       ↓
65 × 384 Embeddings
       ↓
FAISS IndexFlatIP
       ↓
Semantic Retrieval
       ↓
Relevant Policy Chunks
       ↓
Grounded Gemini Prompt
       ↓
Policy Answer + Source