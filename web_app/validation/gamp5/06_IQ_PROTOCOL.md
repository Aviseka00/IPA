# 06 — Installation Qualification (IQ) Protocol

## Objective
Verify the system is installed and configured as specified in a controlled environment.

## Prerequisites
- Windows host (or approved server) with Python 3.11+, Node.js 20+, network to MongoDB
- Valid `.env` with `SECRET_KEY`, `MONGODB_URI`, `PART11_MODE=true`

## Tests

| Step | Action | Expected | Pass/Fail | Initials |
|------|--------|----------|-----------|----------|
| IQ-01 | Record OS, hostname, IP | Documented | | |
| IQ-02 | Confirm `web_app/backend/.env` Part 11 settings | `PART11_MODE=true`, strong `SECRET_KEY` | | |
| IQ-03 | Start app via `scripts/start_daemon.ps1` | `/api/health` returns ok | | |
| IQ-04 | Confirm MongoDB connectivity | Login works; users collection accessible | | |
| IQ-05 | Confirm indexes created | `users.employee_id`, `system_audit` indexes | | |
| IQ-06 | Confirm model weights present | Models listed as available in UI | | |
| IQ-07 | Confirm NTP / time sync policy | Written SOP reference attached | | |
| IQ-08 | Confirm backup location documented | Backup SOP path recorded | | |
| IQ-09 | Confirm validation package present | `web_app/validation/gamp5/` exists | | |
| IQ-10 | Record software version | `/api/compliance/status` → `app_version` | | |

## Approval
| Role | Name | Date | Signature |
|------|------|------|-----------|
| Executor | | | |
| Reviewer (QA) | | | |
