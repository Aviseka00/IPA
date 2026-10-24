# 08 — Performance Qualification (PQ) Protocol

## Objective
Demonstrate the system performs reliably with representative production-like microscopy data, including AI + human review + e-signed release.

## Test sets
Define N images (recommend ≥ 10) with known approximate infected cell counts from SME.

| Test ID | Procedure | Acceptance criteria | Pass/Fail |
|---------|-----------|---------------------|-----------|
| PQ-AI-01 | Run selected production model on test set | All images complete without crash; AI counts recorded | |
| PQ-REVIEW-01 | Reviewer marks FN/FP as needed | Human count = AI + FN − FP for each image | |
| PQ-TABLE-01 | Verify results table | Columns S.No, name, AI, human, FN, FP, total correct | |
| PQ-ESIG-01 | E-sign publish each image | All published; signatures linked | |
| PQ-AUDIT-01 | Export/review audit for batch | Complete chain of inference → draft → publish_esigned | |
| PQ-MODEL-01 | Record model_id/version used | Matches intended production model | |
| PQ-COMPARE-01 | SME compares human totals vs reference | Within agreed tolerance (define: e.g. ±10% or absolute Δ) | |

**Define tolerance here before execution:** _______________

## Approval
| Role | Name | Date | Signature |
|------|------|------|-----------|
| SME / Reviewer | | | |
| QA | | | |
