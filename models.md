# Models Roadmap and Implementation Plan

This document defines a concrete, engineering‑level plan for the recommender models shipped with RecSys‑Lite: what we will support in core, what we move to optional extensions, and exactly how we will implement, test, and ship them.

## Portfolio Decision

- Core (shipped by default, CPU‑friendly, fully production‑ready):
  - `ease` – Real EASE‑R closed‑form item‑item model
  - `als` – Implicit ALS with incremental updates
  - `lightfm` – Hybrid matrix factorization with user/item features
  - `text_embedding` – Content‑based recommendations via sentence‑transformers

- Optional Extensions (install via extras; loaded lazily):
  - Sessions: `sasrec` (preferred) or keep `gru4rec` as legacy; extra name: `recsys-lite[sessions]`
  - Graph CF: `lightgcn`; extra name: `recsys-lite[graph]`
  - Classic baselines: `bpr`, `item2vec`; extra name: `recsys-lite[classical]`
  - Ensemble: `hybrid` remains available; it depends on other models being present and stays in core, but must degrade gracefully when components are missing.

Rationale: Four core models cover the majority of small‑shop scenarios (implicit CF, hybrid with features, pure content/cold‑start). Session and graph models are valuable but heavier in dependencies and tuning; keep as optional to reduce maintenance and installation friction.

## API and Architecture Invariants

- Keep `BaseRecommender.fit(...)` and `recommend(...)` stable across models.
- Factorization models should implement `FactorizationModelMixin` and expose `get_item_factors()` for ANN.
- Content/embedding models should implement a `get_item_vectors(item_ids)` for indexing.
- Add a single ID mapping utility (see Deliverables) so models never assume `str(user_id)` is castable to `int`.
- Persistence uses `ModelPersistenceMixin` plus model‑specific save/load when needed (e.g., LightFM/Text embeddings).

## Deliverables by Model (Core)

### 1) EASE‑R (replace placeholder with real implementation)

- Algorithm:
  - Let `X` be the CSR binary/weighted user×item matrix (float32)
  - Compute `G = X.T @ X` (item Gram). Add ridge: `G += λ * I`.
  - Solve `B = G^{-1}` using Cholesky (`cho_factor/cho_solve`) or `solveh_banded` if we band‑approx; ensure float32 but compute in float64 when feasible for stability, then cast to float32.
  - Compute weights: `W = -B / diag(B)`; set `diag(W) = 0`.
  - Score for a user row `x`: `scores = x @ W`.
- Performance:
  - Memory budget: up to ~100k items requires care; use blockwise Gram accumulation and optionally top‑K pruning after `W` (store sparse top‑K per item, e.g., K=200).
  - Threading: rely on NumPy/BLAS with controlled threadpools (`OPENBLAS_NUM_THREADS=1` in training code if needed) to avoid oversubscription.
- Implementation plan:
  - Replace `src/recsys_lite/models/ease.py` toy version with real EASE‑R.
  - Add `topk: int | None` constructor arg; if set, keep only top‑K per column of `W` (CSR store).
  - Add `dtype` control and safe upcast to float64 during factorization.
  - Persistence: store `lambda_`, `topk`, and `W` (CSR) via pickle or npz.
  - Tests: numerical sanity (symmetry, zero diag), determinism, HR@10/NDCG@10 on a small fixture; performance test (fit < 30s on sample; inference < 2ms @ 100k items on CPU).

### 2) ALS (implicit)

- Fixes/changes:
  - Correct `partial_fit_users` call signature to `(user_ids, user_items_for_those_users)`.
  - Provide `partial_fit_items(...)` passthrough as well.
  - Add robust `IdMapper` usage so `user_id`/`item_id` mapping is explicit and consistent across fit/serve.
  - Ensure `recommend(...)` falls back to all‑items scoring when `implicit.recommend` cannot filter; keep `filter_already_liked_items=True`.
- Persistence: store factors and hyperparams; on `load`, rehydrate `model.user_factors`/`item_factors`.
- Tests: incremental updates correctness (factors change only for targeted users/items), metrics parity vs. baseline, threadpool sanity (no oversubscription warnings in tests).

### 3) LightFM (hybrid)

- Improvements:
  - Ensure `recommend(...)` handles UUIDs via mapping; do not `int()`-cast arbitrary IDs.
  - Support both rank‑based losses (WARP/BPR) and logistic; expose `predict_rank` when appropriate.
  - Efficient inference path that scores all items and applies mask for already‑interacted items.
- Persistence: current pickle is fine; include features if provided; document versioning expectations.
- Tests: feature/no‑feature modes; ranking metrics; cold‑start with item features; save/load integrity.

### 4) Text Embedding (sentence‑transformers)

- Simplifications:
  - Remove/guard the ONNX path; only enable if `optimum` + `onnxruntime` are present and API is compatible. Default to standard ST inference.
  - Strong caching: `text_embeddings.npy` and `item_ids.json` with integrity checks (shape, count).
  - Add offline‑friendly mode (env to force local models or skip network).
- Retrieval:
  - Provide `get_item_vectors(item_ids)` and full item matrix for ANN.
  - Normalize embeddings (L2) for cosine dot.
- Tests: cache hit/miss, batch encoding shape and dtype, deterministic outputs for fixed seeds, basic accuracy sanity via nearest neighbors.

## Optional Extensions (Extras)

### Sessions (`recsys-lite[sessions]`)

- Preferred: `sasrec` (self‑attention for sequences), PyTorch implementation.
- Transitional: keep `gru4rec` behind the same extra; mark as legacy in docs.
- Provide `recommend(session=..., k=...)` interface and a thin adapter from user history when sessions aren’t explicitly supplied.
- Tests: sequential split evaluation (HR@K/NDCG@K) and robustness to variable session lengths.

### Graph CF (`recsys-lite[graph]`)

- `lightgcn` implementation for implicit feedback graphs.
- Export user/item embeddings for ANN; ensure training converges on CPU with small catalogs.
- Tests: accuracy vs. ALS baseline, memory/time budgets.

### Classic (`recsys-lite[classical]`)

- `bpr` (existing using implicit): keep as optional baseline.
- `item2vec`: guard gensim import; handle zero‑vector users; normalize vectors; consider pruning in‑repo if text embeddings cover content sufficiently.

## Cross‑Cutting: ID Mapping and Data Contracts

- Add `src/recsys_lite/utils/id_mapping.py`:
  - `IdMapper.fit(user_ids, item_ids)` builds bijective maps.
  - `to_internal(external_id)` and `to_external(index)` methods.
  - Persist maps alongside model artifacts.
- All models must accept external IDs at API/CLI layers and work internally on indices only.

## Persistence and Artifacts

- Standardize artifact folder layout under `model_artifacts/<model_type>/`:
  - `model.pkl` or `<model>_model.pkl` for params and learned weights.
  - Optional: `embeddings.npy`, `item_ids.json`, `id_mapping.pkl`.
- Version stamp and schema in an adjacent `METADATA.json` (model_type, version, created_at, dataset hash).

## Evaluation and QA

- Add `tests/models/test_metrics.py` with HR@K/NDCG@K utilities and smoke metrics per model on fixtures in `test_data/`.
- Add `test_scripts/bench_models.py` to report training time, inference P95, and memory footprint for a fixed synthetic dataset.
- Target budgets (CPU only, small demo data):
  - Fit: EASE < 30s, ALS < 60s, LightFM < 90s, TextEmb < 60s (caching amortized)
  - Inference: < 5 ms P95 for top‑K=10 per user
  - Memory: < 1 GB resident for 100k×10k (varies by model)

## Dependency Policy

- Core depends only on: numpy, scipy, implicit, lightfm, sentence‑transformers (CPU). Keep versions pinned in `pyproject.toml`.
- Optional extras declare heavy deps (torch, recbole/pyg, gensim, onnxruntime) behind extras.
- Ensure imports are guarded so CI and default installs do not break.

## Documentation Plan

- Update `README.md` “Six Production‑Ready Algorithms” to “Four Core Models + Optional Extensions”.
- Add a “When to Use Which Model” decision table with data shape cues (implicit only, session‑heavy, rich content, new users/items).
- Add a migration note for existing users of `gru4rec`, `item2vec`, `bpr`.

## Rollout Plan (Milestones)

- P0 (Hardening):
  - Replace EASE toy with real EASE‑R, including top‑K sparsification and tests.
  - Fix ALS incremental update signature; add tests.
  - LightFM recommendation/mapping cleanup; add tests.
  - Text Embedding: remove brittle ONNX path; strengthen caching; add tests.
  - Add ID mapping utility; integrate in ALS/LightFM/TextEmb.

- P1 (Quality & Docs):
  - Add metrics harness and benches; update README/docs.
  - Standardize persistence layout and metadata; update loaders/savers.
  - Ensure Hybrid combines at least two core models and handles cold‑start via content boosting.

- P2 (Optional Extensions):
  - Add `sessions` extra: keep `gru4rec` and roadmap `sasrec`.
  - Add `graph` extra: introduce `lightgcn`.
  - Move `bpr` and `item2vec` under `classical` extra; guard imports and registry.
  - Wire extras into `pyproject.toml` and lazy registry loading.

## Acceptance Criteria (for “Production‑Ready”)

- Correctness: deterministic output (where expected), stable API, complete persistence, robust ID mapping.
- Performance: meets budgets above; no threadpool oversubscription; vectorized inference paths.
- Reliability: unit + integration tests per model; golden metrics on sample datasets.
- Documentation: clear usage guidance, tuning tips, and limitations.

---

If we need to further simplify, we can trim `item2vec` from the optional set (covered by text embeddings) and keep `bpr` as a single classical baseline. Sessions and graph models should remain optional due to dependency weight and operational complexity.

