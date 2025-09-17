# Vector Service

The Vector Service is a core component of RecSys-Lite that provides a unified interface for retrieving vector representations of users and items from various recommendation models.

## Features

- Unified interface for retrieving user and item vectors
- Support for different model types and implementations
- Deterministic error propagation with optional legacy fallback
- Robust error handling
- Consistent logging

## Architecture

The Vector Service is designed to work with different types of recommendation models:

1. **Models implementing the VectorProvider protocol**: These models provide a standardized interface for retrieving vectors through `get_user_vectors` and `get_item_vectors` methods.

2. **Models with factor matrices**: These models store user and item factors as matrices and provide access through `get_user_factors` and `get_item_factors` methods.

3. **Optional fallback**: In modern configurations the service raises a `VectorRetrievalError` when vectors are unavailable; an opt-in `allow_random_fallback=True` flag preserves the legacy random-vector behaviour when strictly required.

## Usage

### Basic Usage

```python
from recsys_lite.api.services import VectorService, EntityType

# Create a vector service
vector_service = VectorService()

# Get a user vector
user_vector = vector_service.get_vector(
    model=model,
    entity_type=EntityType.USER,
    entity_idx=user_idx
)

# Get an item vector
item_vector = vector_service.get_vector(
    model=model,
    entity_type=EntityType.ITEM,
    entity_idx=item_idx,
    entity_id=item_id  # Optional, for error reporting
)
```

### Convenience Methods

```python
# Get a user vector using the convenience method
user_vector = vector_service.get_user_vector(
    model=model,
    user_idx=user_idx
)

# Get an item vector using the convenience method
item_vector = vector_service.get_item_vector(
    model=model,
    item_idx=item_idx,
    item_id=item_id  # Optional, for error reporting
)
```

### Error Handling

```python
from recsys_lite.api.errors import VectorRetrievalError

try:
    vector = vector_service.get_vector(
        model=model,
        entity_type=EntityType.USER,
        entity_idx=user_idx
    )
except VectorRetrievalError as e:
    # Handle vector retrieval error
    logger.error(f"Failed to retrieve vector: {e}")
    # Fallback strategy or re-raise
```

## Implementation Details

### Vector Retrieval Process

1. **Try standardized interface**: First, the service tries to use the `get_user_vectors` or `get_item_vectors` methods if available.

2. **Try factor matrices**: If the standardized interface is not available or fails, the service tries to access the factor matrices directly.

3. **Signal failure (default)**: If both strategies return no data, the service raises a `VectorRetrievalError`. When `allow_random_fallback=True` is supplied, it instead generates a random vector for backwards compatibility.

### Error Handling

The service uses a robust error handling approach:

- Specific errors during vector retrieval are caught and logged
- If all retrieval methods fail, a `VectorRetrievalError` is raised with detailed information
- The error includes the entity type, entity ID, and reason for the failure

### Logging

The service includes comprehensive logging:

- Debug logs for successful vector retrievals
- Warning logs for fallbacks to alternative methods
- Error logs for complete failures
- All logs include context information like entity type and ID
