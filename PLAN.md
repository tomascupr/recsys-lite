# Execution Plan

- [x] **Step 1 – Stabilise Incremental Updates**
  - Audit how models persist user/item mappings and factors.
  - Align `UpdateWorker` with persisted mappings, replace deprecated pandas usage, and ensure Faiss only receives new vectors.
  - Add/extend tests validating incremental ALS updates and Faiss growth.
  - Verify via targeted pytest module and scripted ingest→train→update simulation.

- [x] **Step 2 – Harden Batch & Streaming Ingest**
  - Refactor `ingest_data` for idempotent table creation/insertion with schema validation and parameterised queries.
  - Expand tests to cover repeat loads, schema mismatches, and CLI success paths.
  - Confirm via `poetry run recsys-lite ingest` using sample data.

- [x] **Step 3 – Make Vector Retrieval Fail Fast**
  - Replace random-vector fallbacks with explicit errors or configurable fallback flag.
  - Update API error handling/tests to expect deterministic responses.
  - Run refreshed API pytest suite.

- [x] **Step 4 – Implement Real Cache Invalidation**
  - Track cache keys for LRU/Redis, implement actual deletion in invalidate endpoints.
  - Back with unit tests and API endpoint assertions.

- [ ] **Step 5 – Remove Test-Only Behaviour From Faiss Builder**
  - Gate `_new_ids` override behind explicit test flag and update tests accordingly.
  - Re-run Faiss index tests to confirm determinism without altering production behaviour.

- [ ] **Step 6 – Strengthen Automated Tests & Verification**
  - Limit heavy dependency stubs to fallback mode, add integration workflow/target using real libs.
  - Unskip API happy-path tests with deterministic fixtures.
  - Run `poetry run pytest`, lint, type-check, and verify local FastAPI serve with sample data.

- [ ] **Step 7 – Final Validation**
  - Execute formatters (`black`, `isort` if needed), linter, and mypy.
  - Perform end-to-end smoke: ingest → train → serve → curl `/recommend`, and confirm widget compatibility if feasible.
  - Mark plan complete once all preceding steps are finished and validated.
