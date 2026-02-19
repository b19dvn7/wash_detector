# Step 01 — Scope, Guardrails, and Acceptance Criteria

Status: DONE
Date: 2026-02-11

## Why this step exists
Before adding implementation, lock requirements and boundaries so the rebuild stays reliable and isolated.

## Hard Guardrails
1. Project must remain standalone at `/home/bigdan7/Projects/wash_detector`.
2. Do **not** modify `/home/bigdan7/Projects/wash_trading`.
3. Move slowly: one explicit step at a time with evidence.
4. No silent assumptions; document all schema and detection assumptions.

## Rebuild Success Criteria (v1)
A v1 is considered working only if all are true:

1. **Standalone execution**
   - Can run from this project only.
   - No imports or runtime dependency on files from `wash_trading`.

2. **Deterministic pipeline**
   - Explicit stages: ingest -> feature build -> detect -> report.
   - Same input produces same output (excluding timestamps/metadata).

3. **Data contract clarity**
   - Input schema and required/optional fields documented.
   - Missing fields handled explicitly (not hidden placeholders pretending real values).

4. **Explainable alerts**
   - Every alert contains:
     - detector name
     - trigger condition
     - supporting values
     - risk contribution

5. **Validation quality gates**
   - Include at least one synthetic dataset with known labels.
   - Report precision, recall, and false positive rate.
   - Include threshold sensitivity notes.

6. **Operational safety**
   - Read-only by default for source DB inputs.
   - Output artifacts isolated under this project.

## Out of Scope (for initial v1)
- Compliance-grade ownership attribution without account/counterparty linkage.
- Exchange-order lifecycle reconstruction when raw events are unavailable.

## Next Step
Step 02: define canonical schema contract module and schema validation checks.
