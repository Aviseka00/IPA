# IPA Cell Analysis Platform — GAMP 5 Computer System Validation Package

**System:** IPA Cell Analysis Platform (AI-driven infected cell detection)  
**Version:** 1.1.0-part11  
**Framework:** ISPE GAMP 5 (2nd Edition) — risk-based approach  
**Regulations:** 21 CFR Part 11 (Electronic Records; Electronic Signatures)  
**Software category:** Category 4 (Configured Product) with Category 5 (Custom / AI-ML) components  

---

## Package contents

| Document | File | Purpose |
|----------|------|---------|
| Validation Plan | `01_VALIDATION_PLAN.md` | Scope, strategy, responsibilities |
| User Requirements (URS) | `02_URS.md` | What the system must do |
| Risk Assessment | `03_RISK_ASSESSMENT.md` | GAMP 5 risk-based focus |
| Functional Specification | `04_FUNCTIONAL_SPEC.md` | How requirements are met |
| Traceability Matrix | `05_TRACEABILITY_MATRIX.md` | URS ↔ FS ↔ Test |
| IQ Protocol | `06_IQ_PROTOCOL.md` | Installation Qualification |
| OQ Protocol | `07_OQ_PROTOCOL.md` | Operational Qualification |
| PQ Protocol | `08_PQ_PROTOCOL.md` | Performance Qualification |
| Part 11 Assessment | `09_PART11_ASSESSMENT.md` | Electronic records/signatures gap close |
| SOP Outline | `10_SOP_OUTLINE.md` | Procedural controls required by QA |

---

## Important compliance statement

Software **technical controls** alone do **not** constitute a completed CSV or an FDA inspection-ready Part 11 claim.

**Full qualification requires:**

1. Execution and signed completion of IQ / OQ / PQ protocols by trained personnel  
2. Approved SOPs (access control, backup, change control, deviation, training)  
3. Training records for operators, reviewers, and QA  
4. Change control for model weight updates and software releases  
5. Documented time synchronization / backup / retention procedures  

Until those are executed and approved by your Quality Unit, the system posture is:

> **Technically controlled — pending CSV execution and QA approval**

Open **Compliance → Part 11** in the application for the live control matrix and audit trail.
