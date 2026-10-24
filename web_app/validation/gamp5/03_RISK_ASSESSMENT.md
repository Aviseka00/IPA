# 03 — Risk Assessment (GAMP 5)

Risk ranking: Impact (1–3) × Probability (1–3) → Score. High ≥ 6 requires enhanced testing.

| Risk ID | Scenario | Impact | Prob | Score | Mitigation | Test |
|---------|----------|--------|------|-------|------------|------|
| R-01 | Unauthorized access alters counts | 3 | 2 | 6 | Auth, lockout, roles, e-sign | OQ-AUTH, OQ-ESIG |
| R-02 | Audit trail missing / editable | 3 | 1 | 3 | Append-only hash chain | OQ-AUDIT |
| R-03 | Publish without human review | 3 | 2 | 6 | Draft workflow + e-sign required | OQ-PUB |
| R-04 | AI over/under count accepted blindly | 3 | 3 | 9 | FN/FP markers + dual counts | PQ-REVIEW |
| R-05 | Wrong model used | 2 | 2 | 4 | Model registry + batch model_id | OQ-MODEL |
| R-06 | Session hijack / long session | 3 | 2 | 6 | JWT expiry 30 min | OQ-SESS |
| R-07 | Weak passwords | 3 | 2 | 6 | Complexity + history | OQ-PWD |
| R-08 | Model update changes performance | 3 | 2 | 6 | Change control + PQ on new weights | PQ-MODEL |
| R-09 | Data loss | 3 | 1 | 3 | Backup SOP + MongoDB durability | IQ-BACKUP |
| R-10 | Clock drift invalidates timestamps | 2 | 2 | 4 | NTP SOP (procedural) | IQ-TIME |

**AI/ML specific:** Model weights treated as validated configuration; any new `.pt` file requires change control and PQ re-execution on representative data.
