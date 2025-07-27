"""Hyperparameter optimization commands for RecSys-Lite CLI."""

import json
import os
from pathlib import Path

import duckdb
import typer
from scipy.sparse import csr_matrix

try:
    # Lazy import to avoid requiring sklearn for CLI module load
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None  # type: ignore

from recsys_lite.cli import app, logger
from recsys_lite.cli.model import train
from recsys_lite.cli.types import MetricType, ModelType
from recsys_lite.models import ModelRegistry
from recsys_lite.optimization.optimizer import OptunaOptimizer


@app.command()
def optimize(
    model_type: ModelType = typer.Argument(..., help="Type of model to optimize"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output directory"),
    metric: MetricType = typer.Option(MetricType.NDCG_20, help="Evaluation metric"),
    trials: int = typer.Option(20, help="Number of optimization trials"),
    test_size: float = typer.Option(0.2, help="Test set size for evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Optimize hyperparameters for a recommendation model."""
    if output is None:
        output = Path(f"model_artifacts/{model_type.value}")
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data from {db}")
    conn = duckdb.connect(str(db))
    try:
        df = conn.execute(
            """
            SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
            FROM events
            GROUP BY user_id, item_id
            """
        ).fetchdf()
    finally:
        conn.close()

    users = df["user_id"].unique()
    items = df["item_id"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: j for j, it in enumerate(items)}
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        rows.append(user_map[row["user_id"]])
        cols.append(item_map[row["item_id"]])
        data.append(row["interaction"])
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))
    # Parameter space per model
    if model_type == ModelType.ALS:
        param_space = {
            "factors": [32, 64, 128, 256],
            "regularization": [0.001, 0.01, 0.1, 1.0],
            "alpha": [0.1, 1.0, 10.0, 40.0],
            "iterations": [10, 15, 20],
        }
    elif model_type == ModelType.BPR:
        param_space = {
            "factors": [32, 64, 100, 128, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "regularization": [0.001, 0.01, 0.1],
            "iterations": [50, 100, 200],
        }
    elif model_type == ModelType.ITEM2VEC:
        param_space = {
            "vector_size": [32, 64, 100, 128, 200],
            "window": [3, 5, 10],
            "min_count": [1, 3, 5],
            "sg": [0, 1],
            "epochs": [3, 5, 10],
        }
    elif model_type == ModelType.LIGHTFM:
        param_space = {
            "no_components": [32, 64, 128],
            "learning_rate": [0.01, 0.05, 0.1],
            "loss": ["warp", "bpr", "logistic"],
            "epochs": [10, 30, 50],
        }
    elif model_type == ModelType.GRU4REC:
        param_space = {
            "hidden_size": [50, 100, 200],
            "n_layers": [1, 2],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "batch_size": [32, 64, 128],
            "learning_rate": [0.0001, 0.001, 0.01],
            "n_epochs": [5, 10, 15],
        }
    elif model_type == ModelType.EASE:
        param_space = {"lambda_": [0.1, 0.5, 1.0, 5.0, 10.0]}
    elif model_type == ModelType.TEXT_EMBEDDING:
        param_space = {
            "model_name": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            "batch_size": [32, 64, 128],
            "field_weights": [
                {"title": 3.0, "category": 1.0, "brand": 1.0, "description": 2.0},
                {"title": 2.0, "category": 1.0, "brand": 1.0, "description": 3.0},
                {"title": 2.0, "category": 1.5, "brand": 1.5, "description": 2.0},
            ],
        }
    else:
        typer.echo(f"Optimization not supported for {model_type}")
        raise typer.Exit(code=1)
    model_class = ModelRegistry.get_model_class(model_type.value)
    optimizer = OptunaOptimizer(model_class=model_class, metric=metric.value, n_trials=trials, seed=seed)
    if train_test_split is None:
        typer.echo("Error: sklearn is required for hyperparameter optimization")
        raise typer.Exit(code=1)
    train_data, valid_data = train_test_split(user_item_matrix, test_size=test_size, random_state=seed)
    logger.info(f"Running optimization with {trials} trials")
    best_params = optimizer.optimize(train_data=train_data, valid_data=valid_data, param_space=param_space)
    params_file = output / "best_params.json"
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best parameters saved to {params_file}")
    # Retrain with best params
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as t:
        json.dump(best_params, t)
        temp_path = Path(t.name)
    train(model_type=model_type, db=db, output=output, test_size=test_size, seed=seed, params_file=temp_path)
    os.unlink(str(temp_path))
    logger.info("Optimization complete")
