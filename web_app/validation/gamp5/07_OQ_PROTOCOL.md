# 07 — Operational Qualification (OQ) Protocol

## Objective
Demonstrate Part 11 technical controls and core workflows operate as specified.

| Test ID | Procedure | Expected | Pass/Fail |
|---------|-----------|----------|-----------|
| OQ-AUTH-01 | Login with valid credentials | Token issued; audit `login_success` | |
| OQ-AUTH-02 | Login with wrong password ×5 | Account locked; audit `login_failed` | |
| OQ-AUTH-03 | Open registration (non-first user) | Rejected when Part 11 mode on | |
| OQ-PWD-01 | Set password without special char | Rejected | |
| OQ-SESS-01 | Wait beyond JWT expiry | API returns session expired | |
| OQ-MODEL-01 | List models | Available models returned | |
| OQ-AI-01 | Run analysis on sample image | Batch created; AI count stored | |
| OQ-REVIEW-01 | Add FN and FP markers; save draft | Counts update; audit `save_draft` | |
| OQ-ESIG-01 | Publish without signature | Rejected | |
| OQ-ESIG-02 | Publish with wrong password | Rejected; audit `esignature_failed` | |
| OQ-ESIG-03 | Publish with valid e-signature | Status published; signature stored | |
| OQ-AUDIT-01 | Fetch `/api/compliance/audit` | Hash-chained entries present | |
| OQ-COMP-01 | Open Compliance page | Control matrix shows implemented | |
| OQ-PUB-01 | Attempt to edit markers after publish | Rejected | |

## Approval
| Role | Name | Date | Signature |
|------|------|------|-----------|
| Executor | | | |
| QA | | | |
