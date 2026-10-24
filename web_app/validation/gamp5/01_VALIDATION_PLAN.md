# 01 — Validation Plan (GAMP 5)

## 1. Purpose
Define the computer system validation strategy for the IPA Cell Analysis Platform used for AI-assisted infected cell counting on microscopy images, supporting GxP decisions under a risk-based GAMP 5 approach and 21 CFR Part 11 where electronic records/signatures apply.

## 2. System description
- Web application (FastAPI + React) for authenticated upload, AI inference, human review (FN/FP markers), electronic signature on publish, and audit reporting.
- AI components: YOLOv8 segmentation, U-Net, dual-branch fusion (classical + DL).
- Data store: MongoDB (users, batches, analysis_items, system_audit, review_audit).
- Files: stored under controlled storage paths per employee/batch.

## 3. GAMP categorization
| Component | Category | Rationale |
|-----------|----------|-----------|
| Ultralytics / PyTorch runtime | Cat 1 / infrastructure | COTS runtime |
| MongoDB Atlas / MongoDB | Cat 1 | Database platform |
| FastAPI / React application | Cat 4 / Cat 5 | Configured + custom application code |
| Trained model weights (.pt) | Cat 5 (AI/ML) | Custom model artifacts; require change control & PQ |

## 4. Validation approach
1. Document URS → risk assessment → FS  
2. IQ of environment and configuration  
3. OQ of Part 11 technical controls and workflows  
4. PQ with representative microscopy datasets and acceptance criteria  
5. Maintain traceability matrix and change control for releases/models  

## 5. Roles
| Role | Responsibility |
|------|----------------|
| System Owner | Business ownership, URS approval |
| IT / Dev | Build, deploy, IQ evidence |
| QA | Protocol approval, CSV oversight |
| Operator / Reviewer | PQ execution, e-signed results |
| Validation Lead | Protocol drafting, report |

## 6. Deliverables
See package README. Validation Summary Report to be written after IQ/OQ/PQ execution.

## 7. Approval (sign when executed)
| Role | Name | Date | Signature |
|------|------|------|-----------|
| System Owner | | | |
| QA | | | |
| Validation Lead | | | |
