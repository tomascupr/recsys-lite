# RecSys-Lite Operations Runbook

This runbook provides detailed operational guidance for deploying, maintaining, and troubleshooting the RecSys-Lite recommendation system v0.3.3. RecSys-Lite is a lightweight recommendation engine designed for small e-commerce shops, providing personalized product recommendations with minimal computational resources.

## Table of Contents

1. [System Overview](#system-overview)
2. [Initial Setup](#initial-setup)
3. [Data Ingestion](#data-ingestion)
4. [Model Training](#model-training)
5. [API Operations](#api-operations)
6. [Update Worker](#update-worker)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring](#monitoring)
9. [Backup and Recovery](#backup-and-recovery)
10. [GDPR Compliance](#gdpr-compliance)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance Schedule](#maintenance-schedule)

## System Overview

RecSys-Lite is a lightweight recommendation system for small e-commerce shops. It consists of:

- **Data Storage**: DuckDB database storing user-item interactions and item metadata
- **Models**: Various recommendation algorithms (ALS, BPR, item2vec, LightFM, GRU4Rec)
- **API**: FastAPI service for recommendation requests
- **Update Worker**: Background process for incremental model updates
- **React Widget**: Frontend component for displaying recommendations

## Initial Setup

### Prerequisites

- 8+ CPU cores, 16GB+ RAM
- Docker and Docker Compose
- 10GB+ disk space
- Network connectivity for initial container pull only

### Deployment

#### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/tomascupr/recsys-lite.git
   cd recsys-lite
   ```

2. Create directories for data persistence:
   ```bash
   mkdir -p data/incremental model_artifacts/{als,bpr,item2vec,lightfm}
   ```

3. Configure environment (optional):
   ```bash
   # Create .env file with configuration
   cat > .env << EOL
   MODEL_TYPE=als
   API_PORT=8000
   WORKER_INTERVAL=60
   LOG_LEVEL=INFO
   EOL
   ```

4. Start the system:
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

5. Verify deployment:
   ```bash
   curl http://localhost:8000/health
   ```

Expected response:
```json
{
  "status": "healthy"
}
```

#### Manual Deployment

1. Install dependencies:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install Python dependencies
   pip install -e .
   # Or with Poetry
   poetry install
   ```

2. Create directory structure:
   ```bash
   mkdir -p data/incremental model_artifacts/{als,bpr,item2vec,lightfm}
   ```

3. Start API service:
   ```bash
   # Using CLI (located in src/recsys_lite/cli/serve.py)
   recsys-lite serve --model-dir model_artifacts/als --port 8000

   # Or using uvicorn directly
   uvicorn recsys_lite.api.main:app --host 0.0.0.0 --port 8000
   ```

4. Start update worker (in a separate terminal):
   ```bash
   recsys-lite worker --model-dir model_artifacts/als --db data/recsys.db --interval 60
   ```

## Data Ingestion

RecSys-Lite expects two types of data:
1. **Events**: User-item interactions with timestamps and quantities
2. **Items**: Item metadata (categories, brands, prices, etc.)

### Data Format Requirements

#### Events (Parquet)
```
user_id: string      # Unique user identifier
item_id: string      # Unique item identifier
ts: int64            # Unix timestamp in seconds
qty: int             # Quantity (e.g., number of items purchased, viewed)
```

Sample data:
```
┌────────────┬────────────┬────────────┬──────┐
│  user_id   │  item_id   │     ts     │ qty  │
├────────────┼────────────┼────────────┼──────┤
│ user_12345 │ item_67890 │ 1626533119 │ 1    │
│ user_12345 │ item_24680 │ 1626533135 │ 2    │
│ user_67890 │ item_13579 │ 1626533257 │ 1    │
└────────────┴────────────┴────────────┴──────┘
```

#### Items (CSV)
```
item_id: string      # Unique item identifier (primary key)
category: string     # Item category
brand: string        # Item brand
price: float         # Item price
img_url: string      # URL to item image
```

Sample data:
```
┌────────────┬──────────────┬────────────┬───────┬───────────────────────────────┐
│  item_id   │   category   │   brand    │ price │           img_url             │
├────────────┼──────────────┼────────────┼───────┼───────────────────────────────┤
│ item_67890 │ Electronics  │ TechBrand  │ 99.99 │ https://example.com/img1.jpg  │
│ item_24680 │ Clothing     │ FashionCo  │ 49.99 │ https://example.com/img2.jpg  │
│ item_13579 │ Home         │ HomeMakers │ 29.99 │ https://example.com/img3.jpg  │
└────────────┴──────────────┴────────────┴───────┴───────────────────────────────┘
```

### CLI Structure

RecSys-Lite CLI is organized into modules in `src/recsys_lite/cli/`:
- **gdpr.py**: GDPR compliance commands (export-user, delete-user)
- **ingest.py**: Data ingestion commands (ingest, stream-ingest)
- **model.py**: Model training and evaluation commands (train, evaluate)
- **optimize.py**: Hyperparameter optimization commands (optimize)
- **serve.py**: API server commands (serve, worker)

### Initial Data Load

Load initial data into the DuckDB database:

```bash
# Using Docker
docker exec -it recsys-lite_api recsys-lite ingest \
  --events /data/events.parquet \
  --items /data/items.csv \
  --db /data/recsys.db

# Direct CLI usage (ingest command in src/recsys_lite/cli/ingest.py)
recsys-lite ingest \
  --events data/events.parquet \
  --items data/items.csv \
  --db data/recsys.db
```

The ingest command validates required columns and replaces the existing `events`
and `items` tables atomically, so re-running it with a corrected dataset safely
refreshes your DuckDB instance.

### Incremental Data Loading

For incremental data loading, place new event data in Parquet format in the `/data/incremental` directory with filenames containing timestamps (e.g., `events_20230315.parquet`).

The update worker will automatically detect and process these files based on their modification time. This works in both development and production environments.

#### Real‑time streaming helper

If you prefer a **push based** approach you can run the built‑in streaming
ingest loop which watches a directory and appends any newly created parquet
file to the ``events`` table:

```bash
# Poll every 5 s (default) for new parquet files.
recsys-lite stream-ingest data/incremental --db data/recsys.db
```

This requires no extra dependencies – the implementation relies on simple
directory polling so it works in containerised as well as serverless
environments.  Already processed files are remembered for the lifetime of the
process to avoid duplicate imports.

```bash
# Manual incremental load (if needed)
recsys-lite ingest \
  --events data/incremental/events_$(date +%Y%m%d).parquet \
  --db data/recsys.db
```

### Schema Management

The database schema is automatically created during ingestion. To inspect the schema:

```bash
# Connect to DuckDB database directly
duckdb data/recsys.db "SELECT * FROM information_schema.tables; SELECT * FROM information_schema.columns WHERE table_name = 'events';"
```

## Model Training

RecSys-Lite supports multiple recommendation models. Each model has specific strengths and use cases.

### Model Selection Guide

| Model | Strengths | Ideal For | Production Status |
|-------|-----------|-----------|-------------------|
| ALS | Fast, scalable, general-purpose | Most use cases, sparse data | Production ready |
| BPR | Optimized for ranking quality | Ranked display of recommendations | Production ready |
| item2vec | Captures item similarity | Similar item recommendations | Production ready |
| LightFM | Handles cold start, uses item features | New items, content-rich catalogs | Mock implementation in CI |
| GRU4Rec | Captures sequential patterns | Session-based recommendations | Mock implementation in CI |
| EASE | High accuracy, closed‑form linear model | General CF (no user factors) | Production ready |

**Note on Mock Implementations**: Some models (GRU4Rec, LightFM) have mock implementations for CI environments to avoid dependency conflicts. In production, these models provide full functionality. For development and testing, you can use ALS, BPR, item2vec, or EASE models which are fully implemented.

### Initial Training

Train your first model after data ingestion:

```bash
# Train ALS model
recsys-lite train als \
  --db data/recsys.db \
  --output model_artifacts/als

# Train item2vec model
recsys-lite train item2vec \
  --db data/recsys.db \
  --output model_artifacts/item2vec

# Train LightFM model (with item features)
recsys-lite train lightfm \
  --db data/recsys.db \
  --output model_artifacts/lightfm \
  --use-item-features

# Train GRU4Rec model (session-based)
recsys-lite train gru4rec \
  --db data/recsys.db \
  --output model_artifacts/gru4rec

# Train EASE‑R model
recsys-lite train ease \
  --db data/recsys.db \
  --output model_artifacts/ease
```

### Hyperparameter Optimization

Improve model performance with hyperparameter optimization:

```bash
# Optimize ALS model (20 trials)
recsys-lite optimize als \
  --db data/recsys.db \
  --trials 20 \
  --metric "ndcg@20" \
  --output model_artifacts/als_optimized
```

The optimization will explore different parameter combinations:
- For ALS: factors, regularization, alpha
- For BPR: factors, learning_rate, regularization
- For item2vec: vector_size, window, min_count
- For LightFM: no_components, learning_rate, item_alpha, user_alpha
- For GRU4Rec: hidden_size, dropout, learning_rate

### Model Evaluation

After training, evaluate the model's performance:

```bash
# Evaluate model
recsys-lite evaluate \
  --model-dir model_artifacts/als \
  --db data/recsys.db
```

This will output performance metrics such as:
- Hit Rate at K (HR@10, HR@20)
- Normalized Discounted Cumulative Gain (NDCG@10, NDCG@20)
- Coverage and diversity metrics

### Full Retraining Schedule

Schedule full retraining weekly or monthly using a crontab entry:

```bash
# Create retraining script
cat > retrain.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
MODEL_TYPE="als"  # or bpr, item2vec, lightfm
DB_PATH="data/recsys.db"
OUTPUT_DIR="model_artifacts/${MODEL_TYPE}_${DATE}"
CURRENT_DIR="model_artifacts/${MODEL_TYPE}"

# Train new model
recsys-lite train $MODEL_TYPE \
  --db $DB_PATH \
  --output $OUTPUT_DIR

# If successful, replace current model
if [ $? -eq 0 ]; then
  cp -r $OUTPUT_DIR/* $CURRENT_DIR/
  echo "Model updated successfully"
else
  echo "Training failed, keeping current model"
fi
EOF

chmod +x retrain.sh
```

Add this script to crontab for weekly execution:
```bash
# Run every Sunday at 2 AM
0 2 * * 0 /path/to/retrain.sh >> /path/to/retrain.log 2>&1
```

## API Operations

The RecSys-Lite API provides endpoints for recommendations and monitoring.

### API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/recommend` | GET | Get user recommendations | `user_id`, `k`, `use_faiss`, `page`, `page_size`, `categories`, `brands`, `min_price`, `max_price`, `exclude_items`, `include_items` |
| `/similar-items` | GET | Get similar items | `item_id`, `k`, `page`, `page_size`, `categories`, `brands`, `min_price`, `max_price`, `exclude_items` |
| `/health` | GET | Health check | None |
| `/metrics` | GET | Performance metrics | None |
| `/cache/stats` | GET | Get cache statistics | None |
| `/cache/warm` | POST | Warm cache with popular users/items | `popular_users`, `popular_items` (query params) |
| `/cache/clear` | POST | Clear all cache entries | None |
| `/cache/invalidate/user` | POST | Invalidate user cache | `user_id` (optional query param) |
| `/cache/invalidate/item` | POST | Invalidate item cache | `item_id` (optional query param) |
| `/cache/invalidate/model` | POST | Invalidate model cache | `model_version` (optional query param) |

### Example Requests

#### Basic Recommendations
```bash
# Get recommendations for a user
curl "http://localhost:8000/recommend?user_id=user123&k=10"

# Get similar items
curl "http://localhost:8000/similar-items?item_id=item456&k=5"

# Check API health
curl "http://localhost:8000/health"

# Get performance metrics
curl "http://localhost:8000/metrics"
```

#### Recommendations with Pagination
```bash
# Get page 2 of recommendations (5 items per page)
curl "http://localhost:8000/recommend?user_id=user123&k=20&page=2&page_size=5"

# Get page 1 of similar items with custom page size
curl "http://localhost:8000/similar-items?item_id=item456&k=15&page=1&page_size=10"
```

#### Recommendations with Filtering
```bash
# Filter by categories and price range
curl "http://localhost:8000/recommend?user_id=user123&k=10&categories=Electronics&categories=Clothing&min_price=20&max_price=100"

# Filter by brands and exclude specific items
curl "http://localhost:8000/recommend?user_id=user123&k=10&brands=TechBrand&brands=FashionCo&exclude_items=item789&exclude_items=item456"

# Limit to specific items only
curl "http://localhost:8000/recommend?user_id=user123&k=10&include_items=item111&include_items=item222&include_items=item333"
```

#### Cache Management
```bash
# Get cache statistics
curl "http://localhost:8000/cache/stats"

# Clear all cache
curl -X POST "http://localhost:8000/cache/clear"

# Warm cache with popular users and items
curl -X POST "http://localhost:8000/cache/warm?popular_users=user123&popular_users=user456&popular_items=item789&popular_items=item123"

# Invalidate cache for specific user
curl -X POST "http://localhost:8000/cache/invalidate/user?user_id=user123"

# Invalidate cache for all users
curl -X POST "http://localhost:8000/cache/invalidate/user"

# Invalidate cache for specific item
curl -X POST "http://localhost:8000/cache/invalidate/item?item_id=item456"

# Invalidate model cache after model update
curl -X POST "http://localhost:8000/cache/invalidate/model?model_version=v2.1.0"
```

### Response Format

Recommendations are returned in JSON format with item metadata, pagination, and filtering information:

```json
{
  "user_id": "user123",
  "recommendations": [
    {
      "item_id": "item789",
      "score": 0.85,
      "title": "Product Name",
      "category": "Electronics",
      "brand": "BrandName",
      "price": 99.99,
      "img_url": "https://example.com/images/item789.jpg"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_items": 25,
    "total_pages": 3,
    "has_next": true,
    "has_previous": false
  },
  "filters": {
    "categories": ["Electronics", "Clothing"],
    "brands": ["TechBrand"],
    "min_price": 20.0,
    "max_price": 100.0,
    "items_filtered": 5,
    "items_before_filtering": 30
  }
}
```

#### Cache Statistics Response
```json
{
  "cache_stats": {
    "total_keys": 150,
    "hit_count": 1200,
    "miss_count": 300,
    "hit_rate": 0.8
  },
  "cache_enabled": true
}
```

#### Cache Management Responses
```json
{
  "cleared": true,
  "message": "Cache cleared successfully"
}
```

```json
{
  "users_warmed": 50,
  "items_warmed": 100,
  "message": "Cache warmed: 50 users, 100 items"
}
```

```json
{
  "status": "success",
  "message": "Invalidated cache for user: user123"
}
```

### Error Responses

The API may return these error responses:

| Status Code | Description | Example |
|-------------|-------------|---------|
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | User ID or Item ID not found (UserNotFoundError, ItemNotFoundError) |
| 422 | Unprocessable Entity | Invalid request format |
| 500 | Internal Server Error | ModelNotInitializedError, VectorRetrievalError, RecSysError |

#### Error Types
- **UserNotFoundError**: User ID not found in the system
- **ItemNotFoundError**: Item ID not found in the system
- **ModelNotInitializedError**: Recommendation model is not properly initialized
- **VectorRetrievalError**: Failed to retrieve vectors from Faiss index
- **RecSysError**: General recommendation system error

Example error response:
```json
{
  "detail": "User not found: user123",
  "error_code": "USER_NOT_FOUND"
}
```

### API Configuration

The API can be configured using environment variables:

```bash
# Update API configuration with environment variables
export MODEL_DIR=model_artifacts/als
export DB_PATH=data/recsys.db
export LOG_LEVEL=INFO
export MAX_RECOMMENDATIONS=50
export CACHE_EXPIRY=300

# Or create a .env file
cat > .env << EOL
MODEL_DIR=model_artifacts/als
DB_PATH=data/recsys.db
LOG_LEVEL=INFO
MAX_RECOMMENDATIONS=50
CACHE_EXPIRY=300
EOL
```

### API Security

The API is designed for internal use and doesn't include authentication by default. For production deployment, consider:

1. Deploying behind an API gateway or reverse proxy with authentication
2. Implementing rate limiting to prevent abuse
3. Restricting network access to trusted clients only

## Update Worker

The update worker keeps recommendations fresh by incrementally updating models with new data.

### Update Process

1. Worker polls for new events every `INTERVAL` seconds (default: 60)
2. New events are gathered from DuckDB as well as the incremental Parquet drop directory
3. Events are converted into a sparse matrix aligned with the trained user/item mappings
4. Previously unseen users are appended to the mapping and persisted alongside the model artifacts
5. User factors are updated with `partial_fit_users`
6. New vectors for existing catalog items are added to the Faiss index; items missing from the trained mapping are logged for the next retrain

### Starting the Worker

```bash
# Start worker
recsys-lite worker \
  --model-dir model_artifacts/als \
  --db data/recsys.db \
  --interval 60 \
  --incremental-dir data/incremental
```

### Worker Configuration

Configure the worker using command-line parameters or environment variables:

```bash
# Using environment variables
export MODEL_DIR=model_artifacts/als
export DB_PATH=data/recsys.db
export WORKER_INTERVAL=60
export BATCH_SIZE=1000
export INCREMENTAL_DIR=data/incremental

# Start with custom configuration
recsys-lite worker \
  --model-dir $MODEL_DIR \
  --db $DB_PATH \
  --interval $WORKER_INTERVAL \
  --batch-size $BATCH_SIZE \
  --incremental-dir $INCREMENTAL_DIR
```

### Monitoring Worker Status

```bash
# Check worker process
ps aux | grep "recsys-lite worker"

# View worker logs
tail -f logs/worker.log
```

### Restarting the Worker

```bash
# Find worker process
ps aux | grep "recsys-lite worker"

# Kill worker process
kill <PID>

# Restart worker
recsys-lite worker \
  --model-dir model_artifacts/als \
  --db data/recsys.db \
  --interval 60

## Cache Management

RecSys-Lite includes a comprehensive cache management system for optimizing recommendation performance. The cache system supports both in-memory (LRU) and Redis-based caching.

### Cache Architecture

The cache manager provides several layers of caching:
- **Recommendation caching**: Stores user recommendations to reduce computation
- **Similar items caching**: Caches item similarity results
- **Model prediction caching**: Caches model outputs
- **Metadata caching**: Stores frequently accessed item metadata

### Cache Statistics

Monitor cache performance with the stats endpoint:

```bash
curl "http://localhost:8000/cache/stats"
```

Response includes:
- Total cache keys
- Hit/miss counts and rates
- Cache enabled status

### Cache Operations

#### Warming Cache
Pre-populate cache with popular users and items:

```bash
curl -X POST "http://localhost:8000/cache/warm?popular_users=user123&popular_users=user456&popular_items=item789"
```

#### Clearing Cache
Remove all cache entries:

```bash
curl -X POST "http://localhost:8000/cache/clear"
```

#### Selective Invalidation

**User-specific invalidation:**
```bash
# Invalidate specific user
curl -X POST "http://localhost:8000/cache/invalidate/user?user_id=user123"

# Invalidate all users
curl -X POST "http://localhost:8000/cache/invalidate/user"
```

**Item-specific invalidation:**
```bash
# Invalidate specific item
curl -X POST "http://localhost:8000/cache/invalidate/item?item_id=item456"

# Invalidate all items
curl -X POST "http://localhost:8000/cache/invalidate/item"
```

**Model cache invalidation:**
```bash
# Invalidate when model is updated
curl -X POST "http://localhost:8000/cache/invalidate/model?model_version=v2.1.0"
```

### Cache Configuration

Configure cache behavior via environment variables:

```bash
export CACHE_ENABLED=true
export CACHE_TYPE=redis  # or 'memory' for LRU
export CACHE_EXPIRY=300  # seconds
export CACHE_MAX_SIZE=1000  # for LRU cache
export REDIS_URL=redis://localhost:6379  # for Redis cache
```

All cache operations delete the underlying cache keys (LRU or Redis) and after invalidation, the next read will repopulate data from the source of truth.

## Vector Service Architecture

RecSys-Lite uses a sophisticated vector service for fast similarity search and recommendations. The vector service is built on top of Faiss (Facebook AI Similarity Search) for efficient nearest neighbor search.

### Components

- **Vector Index**: Faiss-based index for fast similarity search
- **Item Embeddings**: Vector representations of items from trained models
- **User Embeddings**: Vector representations of users (when available)
- **Similarity Engine**: Handles vector similarity computations

### Vector Operations

The vector service provides several key operations:

1. **Vector Retrieval**: Get vector representations for users and items
2. **Similarity Search**: Find similar items using vector similarity
3. **Batch Operations**: Process multiple vectors efficiently
4. **Index Management**: Update and rebuild vector indices

### Error Handling

Vector operations may raise:
- **VectorRetrievalError**: When vectors cannot be retrieved from the index
- **ModelNotInitializedError**: When the vector service is not properly initialized

### Configuration

```bash
# Faiss index configuration
export FAISS_NLIST=100      # Number of clusters
export FAISS_NPROBE=10      # Clusters to search (speed vs accuracy)
export FAISS_USE_THREADS=1  # Enable threading
export VECTOR_CACHE_SIZE=1000  # Cache size for vectors
```

## Performance Tuning

Optimize RecSys-Lite for your specific workload:

### Memory Optimization

```bash
# Reduce memory usage for smaller deployments
export FAISS_NPROBE=5
export BATCH_SIZE=500
export CACHE_SIZE=100
```

### CPU Optimization

```bash
# Optimize for multi-core systems
export THREADS=8
export FAISS_USE_THREADS=1
```

### Model Size vs. Quality

Adjust model size based on your catalog:

| Catalog Size | Recommended Factors | Memory Usage |
|--------------|---------------------|--------------|
| <10K items | 50-100 | Low |
| 10K-50K items | 100-200 | Medium |
| >50K items | 200-300 | High |

```bash
# Configure model size
recsys-lite train als \
  --db data/recsys.db \
  --output model_artifacts/als \
  --params '{"factors": 150}'
```

### Faiss Index Configuration

Configure Faiss for faster similarity search:

```bash
# Configure Faiss index for better performance
export FAISS_NLIST=100  # Number of clusters (increase for larger catalogs)
export FAISS_NPROBE=10  # Number of clusters to search (tradeoff between speed and accuracy)
```

## Monitoring

### Health and Status

```bash
# Check API health
curl "http://localhost:8000/health"

# Get system metrics
curl "http://localhost:8000/metrics"
```

Example metrics response:
```json
{
  "uptime_seconds": 3600,
  "request_count": 1500,
  "recommendation_count": 15000,
  "error_count": 5,
  "recommendations_per_second": 4.16,
  "cache_hit_ratio": 0.85
}
```

### Log Inspection

```bash
# View API logs
tail -f logs/api.log

# View worker logs
tail -f logs/worker.log
```

### Prometheus Integration

RecSys-Lite metrics can be scraped by Prometheus:

```yaml
scrape_configs:
  - job_name: 'recsys-lite'
    scrape_interval: 15s
    static_configs:
      - targets: ['your-server:8000']
    metrics_path: '/metrics'
```

### Common Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `uptime_seconds` | API uptime | N/A |
| `request_count` | Total requests | N/A |
| `recommendation_count` | Total recommendations served | N/A |
| `error_count` | Failed requests | 0 |
| `recommendations_per_second` | Throughput | Depends on workload |
| `cache_hit_ratio` | Cache efficiency | >0.8 |

## Backup and Recovery

### Database Backup

```bash
# Backup DuckDB database
mkdir -p backups/$(date +%Y%m%d)
cp data/recsys.db backups/$(date +%Y%m%d)/recsys_$(date +%Y%m%d).db
```

### Model Artifacts Backup

```bash
# Backup model artifacts
mkdir -p backups/$(date +%Y%m%d)
tar -czf backups/$(date +%Y%m%d)/models_$(date +%Y%m%d).tar.gz model_artifacts/
```

### Automated Backup Script

```bash
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="backups/$DATE"
mkdir -p $BACKUP_DIR

# Backup database
cp data/recsys.db $BACKUP_DIR/recsys_$DATE.db

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz model_artifacts/

# Clean up old backups (keep 30 days)
find backups -type d -mtime +30 -exec rm -rf {} \;
EOF

chmod +x backup.sh
```

Add to crontab for daily execution:
```bash
# Run every day at 1 AM
0 1 * * * /path/to/backup.sh >> /path/to/backup.log 2>&1
```

### Recovery Procedure

```bash
# Choose backup date
BACKUP_DATE=20230315

# Restore database
cp backups/$BACKUP_DATE/recsys_$BACKUP_DATE.db data/recsys.db

# Restore models
mkdir -p model_artifacts_backup  # Backup current models just in case
mv model_artifacts model_artifacts_backup
mkdir -p model_artifacts
tar -xzf backups/$BACKUP_DATE/models_$BACKUP_DATE.tar.gz -C .

# Restart services
# (Restart API and worker processes)
```

## GDPR Compliance

RecSys-Lite includes tools for GDPR compliance.

### Data Inventory

User data is stored in:
1. DuckDB database (raw events)
2. Model embeddings (transformed data)
3. Faiss index (item vectors)

### User Data Export

Export all data associated with a user using the GDPR module (`src/recsys_lite/cli/gdpr.py`):

```bash
# Export a user's data
recsys-lite gdpr export-user \
  --user-id USER_ID \
  --db data/recsys.db \
  --output exports/user_USER_ID.json
```

The export contains:
- Raw event data
- Items the user interacted with
- Timestamps and quantities

Example export format:
```json
{
  "user_id": "user123",
  "events": [
    {
      "item_id": "item456",
      "timestamp": 1626533119,
      "quantity": 1
    },
    {
      "item_id": "item789",
      "timestamp": 1626533257,
      "quantity": 2
    }
  ],
  "items": [
    {
      "item_id": "item456",
      "category": "Electronics",
      "brand": "TechBrand",
      "price": 99.99,
      "img_url": "https://example.com/images/item456.jpg"
    },
    {
      "item_id": "item789",
      "category": "Clothing",
      "brand": "FashionCo",
      "price": 49.99,
      "img_url": "https://example.com/images/item789.jpg"
    }
  ],
  "export_timestamp": 1689245631,
  "export_date": "2023-07-13T12:00:31Z"
}
```

### User Data Deletion

Delete a user's data from the system using the GDPR module:

```bash
# Delete a user's data
recsys-lite gdpr delete-user \
  --user-id USER_ID \
  --db data/recsys.db
```

The deletion process:
1. Removes raw events from DuckDB
2. Marks the user for removal in models
3. Update worker applies changes to embeddings

### Data Retention Policy

Configure data retention period:

```bash
# Set 6-month retention policy (180 days)
recsys-lite configure \
  --retention-days 180
```

The system will automatically:
1. Identify events older than the retention period
2. Remove them from the database
3. Update models during the next retraining cycle

### Right to be Forgotten Workflow

1. Receive deletion request
2. Execute deletion command
3. Provide confirmation with timestamp
4. Document deletion in compliance log
5. Rotate backups to ensure complete removal

### Audit Trail

All GDPR-related operations are logged in `logs/gdpr.log`:

```bash
# View GDPR audit log
tail -f logs/gdpr.log
```

Log format:
```
2023-07-13 12:00:31 - DATA_EXPORT - user_id=user123 - success
2023-07-13 12:15:42 - DATA_DELETION - user_id=user456 - success
```

## Troubleshooting

### API Not Responding

1. Check container status:
   ```bash
   docker ps | grep recsys-lite
   ```

2. Check logs:
   ```bash
   tail -f logs/api.log
   ```

3. Verify database connection:
   ```bash
   duckdb data/recsys.db "SELECT COUNT(*) FROM events;"
   ```

4. Check model artifacts:
   ```bash
   ls -la model_artifacts/als/
   ```

5. Restart service:
   ```bash
   # Kill and restart the API process
   kill $(pgrep -f "recsys-lite serve")
   recsys-lite serve --model-dir model_artifacts/als --port 8000
   ```

### Poor Recommendation Quality

1. Check data quality:
   ```bash
   # Check event counts
   duckdb data/recsys.db "SELECT COUNT(*) FROM events;"
   
   # Check user and item counts
   duckdb data/recsys.db "SELECT COUNT(DISTINCT user_id) FROM events; SELECT COUNT(DISTINCT item_id) FROM events;"
   ```

2. Check model metrics:
   ```bash
   recsys-lite evaluate \
     --model-dir model_artifacts/als \
     --db data/recsys.db
   ```

3. Common issues and solutions:
   - **Cold start problem**: Use LightFM with item features
   - **Too generic recommendations**: Increase model factors
   - **Poor similar items**: Try item2vec or increase training data
   - **Missing recent behavior**: Check update worker logs

4. Try different model types:
   ```bash
   # Try BPR model instead of ALS
   recsys-lite train bpr \
     --db data/recsys.db \
     --output model_artifacts/bpr
   ```

### Update Worker Issues

1. Check worker status:
   ```bash
   ps aux | grep "recsys-lite worker"
   ```

2. Check worker logs:
   ```bash
   tail -f logs/worker.log
   ```

3. Common issues:
   - **Worker not updating**: Check incremental data path
   - **Memory errors**: Reduce batch size
   - **Slow updates**: Check event volume, adjust interval

4. Restart worker:
   ```bash
   kill $(pgrep -f "recsys-lite worker")
   recsys-lite worker --model-dir model_artifacts/als --db data/recsys.db --interval 60
   ```

### Database Corruption

1. Check database integrity:
   ```bash
   duckdb data/recsys.db "SELECT 1;"
   ```

2. Try to repair database:
   ```bash
   duckdb_recovery data/recsys.db
   ```

3. Restore from backup if necessary:
   ```bash
   cp backups/20230315/recsys_20230315.db data/recsys.db
   ```

### Out of Memory Errors

If you encounter out-of-memory errors:

1. Reduce model size:
   ```bash
   # Train with smaller factors
   recsys-lite train als \
     --db data/recsys.db \
     --output model_artifacts/als \
     --params '{"factors": 50}'
   ```

2. Reduce batch size:
   ```bash
   export BATCH_SIZE=200
   ```

3. Implement data sampling:
   ```bash
   # Train on a sample of data
   recsys-lite train als \
     --db data/recsys.db \
     --output model_artifacts/als \
     --sample-rate 0.5
   ```

## Maintenance Schedule

Follow this recommended maintenance schedule:

### Daily

- Check API health (`curl http://localhost:8000/health`)
- Review error logs (`tail -f logs/api.log | grep ERROR`)
- Verify update worker status
- Back up database and models

### Weekly

- Run full model retraining
- Check recommendation metrics
- Clean up old incremental files
- Review performance metrics

### Monthly

- Run hyperparameter optimization
- Perform disk cleanup
- Test recovery procedures
- Review GDPR compliance logs

### Quarterly

- Update to latest RecSys-Lite version
- Audit GDPR compliance
- Review and update retention policies
- Benchmark system performance
