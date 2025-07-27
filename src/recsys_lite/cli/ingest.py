"""Ingestion commands for RecSys-Lite CLI."""

from pathlib import Path

import typer

from recsys_lite.cli import app
from recsys_lite.cli.types import QueueType
from recsys_lite.ingest import ingest_data, queue_ingest, stream_events


@app.command()
def ingest(
    events: Path = typer.Argument(..., help="Path to events parquet file"),
    items: Path = typer.Argument(..., help="Path to items CSV file"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
) -> None:
    """Ingest data into DuckDB database."""
    ingest_data(events, items, db)
    typer.echo(f"Data ingested successfully into {db}")


@app.command(name="stream-ingest")
def stream_ingest(
    events_dir: Path = typer.Argument(..., help="Directory containing incremental parquet files"),
    db: Path = typer.Option("recsys.db", help="DuckDB database to append to"),
    poll_interval: int = typer.Option(5, help="Polling interval in seconds"),
) -> None:
    """Run a simple *file based* streaming ingest loop."""
    typer.echo(f"Starting streaming ingest – watching '{events_dir}' every {poll_interval}s …")
    try:
        stream_events(events_dir, db, poll_interval=poll_interval)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="queue-ingest")
def queue_ingest_command(
    queue_type: QueueType = typer.Argument(QueueType.RABBITMQ, help="Type of message queue"),
    db: Path = typer.Option("recsys.db", help="DuckDB database to append to"),
    batch_size: int = typer.Option(100, help="Number of messages to process in a batch"),
    poll_interval: int = typer.Option(5, help="Polling interval in seconds"),
) -> None:
    """Run a *message queue* based streaming ingest process."""
    # NOTE: Specific queue configuration flags (e.g. rabbitmq_host) are not supported in this command.
    queue_config = {}
    typer.echo(f"Starting {queue_type.value} queue ingest – batch size: {batch_size}, interval: {poll_interval}s")
    queue_ingest(
        queue_config=queue_config,
        db_path=db,
        queue_type=queue_type.value,
        batch_size=batch_size,
        poll_interval=poll_interval,
    )
