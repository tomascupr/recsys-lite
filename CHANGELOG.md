# Changelog

All notable changes to the RecSys-Lite project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-07-26

## [0.3.2] - 2025-09-17

### Added
- Contributor onboarding guide (`AGENTS.md`) outlining workflow, coding style, and testing expectations.
- Unit suites covering incremental update worker and cache invalidation behaviour to prevent regressions.

### Changed
- Stabilised incremental update worker by respecting persisted mappings, persisting new users, and safely updating Faiss indices.
- Refactored batch ingestion to validate schemas and replace DuckDB tables atomically for repeatable loads.
- Made vector retrieval deterministic by surfacing `VectorRetrievalError` when models lack vectors, with an opt-in random fallback flag.
- Implemented real cache invalidation with key tracking for user/item recommendations and vectors, and exposed CLI/API docs for managing cache state.
- Unified lightweight Redis stub across environments to ensure predictable cache fallbacks.

### Security
- Updated FastAPI from 0.100.x to 0.116.x to address Starlette security vulnerabilities
- Updated uvicorn from 0.23.x to 0.35.x for security improvements
- Updated requests from 2.32.3 to 2.32.4 to fix CVE
- Updated urllib3 from 2.4.0 to 2.5.0 for security patches
- Updated certifi to latest version for certificate bundle updates
- Updated all development dependencies to latest secure versions

### Changed
- Expanded dependency version ranges to allow minor updates
- Updated Poetry lock file with all security patches
- Modernized README with compelling developer-focused messaging

## [0.3.0] - 2025-04-20

### Added
- Pagination support for recommendation and similar-items endpoints
- Comprehensive filtering capabilities for recommendation endpoints:
  - Category-based filtering
  - Brand-based filtering
  - Price range filtering (min/max price)
  - Item exclusion and inclusion lists
- Enhanced response format with pagination and filter information
- Automatic buffer sizing when filtering to maintain result count
- Response caching with time-to-live (TTL) expiry for recommendation endpoints
- Cache management endpoints for clearing and configuring caches
- Personalized reranking capabilities:
  - Diversity-focused reranking to provide variety across categories
  - Novelty-based reranking to surface less obvious recommendations
  - Popularity-weighted reranking for trending items
  - Hybrid reranking combining multiple strategies
- Explanations for recommendations based on reranking strategy
- Configurable weights for reranking components
- Cache hit/miss statistics and metrics

### Changed
- Recommendation endpoints now use a more consistent response format
- Similar-items endpoint now returns same format as recommendations endpoint
- Improved error handling for pagination and filter parameters
- More efficient recommendation processing with caching
- Enhanced API metrics with cache performance statistics

## [0.2.0] - 2025-04-19

### Added
- Text embedding model using all-MiniLM-L6-v2 for content-based recommendations
- Hybrid model for combining multiple recommendation approaches
- ONNX runtime acceleration for improved inference performance
- Dynamic model weighting based on user interaction patterns
- Field weighting for improved item text representation
- `train_hybrid` command for creating hybrid models
- LLM dependencies group (`recsys-lite[llm]`)
- Cold-start user handling strategies
- Weighted average of item embeddings for user profile generation

### Changed
- Updated CLI interface to support new model types
- Updated ModelType enum to include TEXT_EMBEDDING and HYBRID options
- Improved model persistence for maintaining vector representations
- Enhanced recommendation algorithm to consider item content
- Optimized hyperparameter spaces for text embedding models

### Fixed
- Properly handle empty user interaction histories 
- Type annotation improvements for static analysis

## [0.1.0] - Initial Release

### Added
- Core recommendation system functionality
- ALS, BPR, Item2Vec, LightFM, GRU4Rec, and EASE models
- FastAPI-based recommendation service
- DuckDB data storage
- CLI tools for data ingestion and model training
- React recommendation widget
- FAISS indexing for fast similarity search
- Hyperparameter optimization with Optuna
- Incremental model updating
- GDPR compliance tools
