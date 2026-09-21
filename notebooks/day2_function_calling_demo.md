
# Day 2 — Operational Data, Tools & Function Calling

## Uber Global Airport Operations & Supply Disruption Resolver

### Objective

Give the AI copilot access to mock airport operational data and allow the LLM to select and execute tools when additional information is required.

### Day 2 workflow

**User Query → Gemini decides whether a tool is needed → Function Call → Local Tool Execution → Tool Response → Grounded Final Response**

### Three operational tools

1. `get_airport_metrics(airport_code)`
2. `calculate_driver_incentive(driver_count, severity_level)`
3. `trigger_surge_override(airport_code, new_multiplier, reason)`

> The surge override is a mock execution on Day 2. Approval and permission controls will be added on Day 4.

---

## 1. Synthetic Operational Data

The project contains:

`data/operational_data/airport_telemetry.csv`

Dataset validation:

- 72 records
- 9 columns
- 24 hourly records for SFO
- 24 hourly records for LAX
- 24 hourly records for JFK

Required columns:

- `airport_code`
- `completion_rate`
- `average_eta`
- `active_drivers`
- `driver_cancellation_rate`
- `queue_size`
- `surge_multiplier`
- `request_volume`
- `timestamp`

Dataset verification result:

```text
(72, 9)

JFK    24
LAX    24
SFO    24
