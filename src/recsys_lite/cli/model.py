"""Model training commands for RecSys-Lite CLI."""

import json
from pathlib import Path
from typing import Optional

import duckdb
import typer
from scipy.sparse import csr_matrix

from recsys_lite.cli import app, logger
from recsys_lite.cli.types import ModelType
from recsys_lite.indexing import FaissIndexBuilder
from recsys_lite.models import ModelRegistry
from recsys_lite.utils import IdMapper


@app.command()
def train(
    model_type: ModelType = typer.Argument(..., help="Type of model to train"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output directory"),
    test_size: float = typer.Option(0.2, help="Test set size for evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
    params_file: Optional[Path] = typer.Option(None, help="JSON file with model parameters"),
) -> None:
    """Train a recommendation model."""
    if output is None:
        output = Path(f"model_artifacts/{model_type.value}")
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data from {db}")
    conn = duckdb.connect(str(db))
    try:
        user_item_df = conn.execute(
            """
            SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
            FROM events
            GROUP BY user_id, item_id
            """
        ).fetchdf()
        item_df = conn.execute("SELECT * FROM items").fetchdf()
    finally:
        conn.close()

    logger.info("Creating user and item mappings")
    unique_users = user_item_df["user_id"].unique()
    unique_items = user_item_df["item_id"].unique()
    user_mapper = IdMapper.from_iterable(unique_users)
    item_mapper = IdMapper.from_iterable(unique_items)
    user_mapping = user_mapper.to_dict()
    item_mapping = item_mapper.to_dict()
    item_data = {row["item_id"]: row.to_dict() for _, row in item_df.iterrows()}
    if model_type == ModelType.TEXT_EMBEDDING:
        user_item_matrix = None
    else:
        rows, cols, data = [], [], []
        for _, row in user_item_df.iterrows():
            rows.append(user_mapper.to_index(row["user_id"]))
            cols.append(item_mapper.to_index(row["item_id"]))
            data.append(row["interaction"])
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_mapping), len(item_mapping)))
    params = {}
    if params_file and params_file.exists():
        with open(params_file, "r") as f:
            params = json.load(f)
    logger.info(f"Initializing {model_type.value} model")
    model_class = ModelRegistry.get_model_class(model_type.value)
    # Prepare default params per model
    if model_type == ModelType.ALS:
        model_params = {
            "factors": params.get("factors", 128),
            "regularization": params.get("regularization", 0.01),
            "alpha": params.get("alpha", 1.0),
            "iterations": params.get("iterations", 15),
        }
    elif model_type == ModelType.BPR:
        model_params = {
            "factors": params.get("factors", 100),
            "learning_rate": params.get("learning_rate", 0.05),
            "regularization": params.get("regularization", 0.01),
            "iterations": params.get("iterations", 100),
        }
    elif model_type == ModelType.ITEM2VEC:
        model_params = {
            "vector_size": params.get("vector_size", 100),
            "window": params.get("window", 5),
            "min_count": params.get("min_count", 5),
            "sg": params.get("sg", 1),
            "epochs": params.get("epochs", 5),
        }
    elif model_type == ModelType.LIGHTFM:
        model_params = {
            "no_components": params.get("no_components", 64),
            "learning_rate": params.get("learning_rate", 0.05),
            "loss": params.get("loss", "warp"),
            "epochs": params.get("epochs", 50),
        }
    elif model_type == ModelType.GRU4REC:
        model_params = {
            "n_items": len(item_mapping),
            "hidden_size": params.get("hidden_size", 100),
            "n_layers": params.get("n_layers", 1),
            "dropout": params.get("dropout", 0.1),
            "batch_size": params.get("batch_size", 64),
            "learning_rate": params.get("learning_rate", 0.001),
            "n_epochs": params.get("n_epochs", 10),
        }
    elif model_type == ModelType.EASE:
        model_params = {"lambda_": params.get("lambda_", 0.5)}
        if "topk" in params and params["topk"] is not None:
            model_params["topk"] = int(params["topk"])
    elif model_type == ModelType.TEXT_EMBEDDING:
        model_params = {
            "model_name": params.get("model_name", "all-MiniLM-L6-v2"),
            "item_text_fields": params.get("item_text_fields", ["title", "category", "brand", "description"]),
            "field_weights": params.get(
                "field_weights", {"title": 2.0, "category": 1.0, "brand": 1.0, "description": 3.0}
            ),
            "normalize_vectors": params.get("normalize_vectors", True),
            "batch_size": params.get("batch_size", 64),
            "max_length": params.get("max_length", 512),
        }
    else:
        typer.echo(f"Unknown model type: {model_type}")
        raise typer.Exit(code=1)
    if model_type == ModelType.HYBRID:
        typer.echo("Error: Hybrid models should be created using the train-hybrid command")
        raise typer.Exit(code=1)
    model = model_class(**model_params)
    logger.info(f"Training {model_type.value} model")
    if model_type == ModelType.TEXT_EMBEDDING:
        model.fit(user_item_matrix=user_item_matrix, item_data=item_data, output_dir=output)
    else:
        model.fit(user_item_matrix)
    logger.info(f"Saving model and artifacts to {output}")
    model.save_model(str(output))
    with open(output / "user_mapping.json", "w") as f:
        json.dump(user_mapping, f)
    with open(output / "item_mapping.json", "w") as f:
        json.dump(item_mapping, f)
    logger.info("Building Faiss index")
    # Get item vectors
    if hasattr(model, "get_item_vectors"):
        item_ids = list(item_mapping.values())
        item_vectors = model.get_item_vectors(item_ids)
    elif hasattr(model, "get_item_factors"):
        item_vectors = model.get_item_factors()
    else:
        import numpy as _np

        size = getattr(model, "factors", 100)
        item_vectors = _np.random.random((len(item_mapping), size)).astype(_np.float32)
    index_builder = FaissIndexBuilder(vectors=item_vectors, ids=list(range(len(item_mapping))))
    index_builder.save(str(output / "faiss_index"))
    logger.info(f"{model_type.value} model training complete")


@app.command()
def train_hybrid(
    models_dir: list[Path] = typer.Argument(..., help="List of model directories to combine"),
    output: Path = typer.Option(None, help="Output directory"),
    weights: Optional[list[float]] = typer.Option(None, help="Model weights"),
    dynamic: bool = typer.Option(True, help="Use dynamic weighting based on user history"),
    cold_start_threshold: int = typer.Option(5, help="Threshold for cold-start users"),
    cold_start_strategy: str = typer.Option("content_boost", help="Cold-start strategy"),
) -> None:
    """Create a hybrid model combining multiple recommenders."""
    if output is None:
        output = Path("model_artifacts/hybrid")
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {len(models_dir)} component models")
    from recsys_lite.models import ModelRegistry

    models = []
    types = []
    for model_dir in models_dir:
        model_files = list(model_dir.glob("*_model.pkl"))
        if not model_files:
            logger.warning(f"No model found in {model_dir}")
            continue
        mtype = model_files[0].stem.split("_")[0]
        types.append(mtype)
        try:
            m = ModelRegistry.load_model(mtype, str(model_dir))
            models.append(m)
            logger.info(f"Loaded {mtype} from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {e}")
    if not models:
        typer.echo("No models could be loaded")
        raise typer.Exit(code=1)
    if weights and len(weights) == len(models):
        ws = weights
    else:
        ws = None
    from recsys_lite.models import HybridModel

    hybrid = HybridModel(
        models=models,
        weights=ws,
        dynamic_weighting=dynamic,
        cold_start_threshold=cold_start_threshold,
        cold_start_strategy=cold_start_strategy,
    )
    logger.info(f"Saving hybrid model to {output}")
    hybrid.save_model(str(output))
    # Copy mappings
    for f in ["user_mapping.json", "item_mapping.json"]:
        src = models_dir[0] / f
        if src.exists():
            import shutil

            shutil.copy(src, output / f)
    logger.info("Building Faiss index for hybrid model")
    import json as _json

    item_map = _json.loads((output / "item_mapping.json").read_text())
    if hasattr(hybrid, "get_item_vectors"):
        item_ids = list(item_map.values())
        vectors = hybrid.get_item_vectors(item_ids)
    elif hasattr(hybrid, "get_item_factors"):
        vectors = hybrid.get_item_factors()
    else:
        import numpy as _np

        sz = getattr(hybrid, "factors", 100)
        vectors = _np.random.random((len(item_map), sz)).astype(_np.float32)
    FaissIndexBuilder(vectors=vectors, ids=list(range(len(item_map)))).save(str(output / "faiss_index"))
    logger.info(f"Hybrid model ({'+'.join(types)}) creation complete")
