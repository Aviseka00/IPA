# 02 — User Requirements Specification (URS)

## URS-01 Authentication
Unique Employee ID and password; failed login lockout; session timeout.

## URS-02 Authorization
Role-based access (operator, reviewer, qa, admin). Open self-registration disabled in Part 11 mode.

## URS-03 AI Analysis
User can select registered model, upload images, obtain AI infected cell counts and overlays.

## URS-04 Human Review
User can mark false negatives (green) and false positives (red); draft state before publish.

## URS-05 Electronic Signature
Publishing final human-verified count requires password re-authentication, printed name, and meaning text (Part 11).

## URS-06 Audit Trail
System records who/what/when for login, logout, inference, draft save, e-sign publish; append-only with integrity hash chain.

## URS-07 Results Table
Results presented with S.No, image name, AI count, human verified, FN, FP, total.

## URS-08 Data Integrity
Published records are immutable for markers; AI counts retained separately from human counts (ALCOA+).

## URS-09 Retention
Electronic records retained per configured retention period (default 7 years) via backup SOP.

## URS-10 Validation Visibility
Application exposes compliance status and control evidence for CSV.

## URS-11 Password Policy
Passwords meet complexity and history rules aligned with Part 11 §11.300.

## URS-12 Model Traceability
Each batch stores model_id / model_name used for inference.
