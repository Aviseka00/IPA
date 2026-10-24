# 04 — Functional Specification (FS)

| FS ID | Behavior | Implementation |
|-------|----------|----------------|
| FS-01 | Login with unique ID/password | `POST /api/auth/login` |
| FS-02 | Lock after N failures | `max_failed_logins`, `locked_until` |
| FS-03 | JWT session with expiry | `access_token_expire_minutes` |
| FS-04 | Password complexity enforced | `validate_password_complexity` |
| FS-05 | Open registration gated | `part11_mode` + `allow_open_registration=false` |
| FS-06 | Run AI analysis | `POST /api/analysis/run` |
| FS-07 | Store AI + human counts | `analysis_items` fields |
| FS-08 | FN/FP markers | Review API + ReviewPanel |
| FS-09 | E-signature on publish | `PublishRequest.signature` + password verify |
| FS-10 | Hash-chained audit | `system_audit` via `write_system_audit` |
| FS-11 | Results table | `ResultsTable` component |
| FS-12 | Compliance dashboard | `GET /api/compliance/status` |
| FS-13 | Roles on user | `role` field: operator/reviewer/qa/admin |
