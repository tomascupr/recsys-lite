"""GDPR compliance commands for RecSys-Lite CLI."""

import json
import time
from pathlib import Path

import duckdb
import typer

from recsys_lite.cli import app

# GDPR subgroup
gdpr_app = typer.Typer()
app.add_typer(gdpr_app, name="gdpr")


@gdpr_app.command(name="export-user")
def export_user(
    user_id: str = typer.Argument(..., help="User ID to export data for"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output JSON file path"),
) -> None:
    """Export all data for a specific user (GDPR compliance)."""
    if output is None:
        output = Path(f"{user_id}.json")
    conn = duckdb.connect(str(db))
    try:
        events_df = conn.execute("SELECT * FROM events WHERE user_id = ?", [user_id]).fetchdf()
        if len(events_df):
            item_ids = events_df["item_id"].tolist()
            placeholders = ", ".join(["?"] * len(item_ids))
            item_df = conn.execute(f"SELECT * FROM items WHERE item_id IN ({placeholders})", item_ids).fetchdf()
        else:
            item_df = conn.execute("SELECT * FROM items WHERE 1=0").fetchdf()
        export_data = {
            "user_id": user_id,
            "export_timestamp": int(time.time()),
            "events": events_df.to_dict(orient="records"),
            "items": item_df.to_dict(orient="records"),
        }
    finally:
        conn.close()
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)
    typer.echo(f"User data exported to {output}")


@gdpr_app.command(name="delete-user")
def delete_user(
    user_id: str = typer.Argument(..., help="User ID to delete data for"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion without prompting"),
) -> None:
    """Delete all data for a specific user (GDPR compliance)."""
    conn = duckdb.connect(str(db))
    try:
        result = conn.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", [user_id]).fetchone()
        count = result[0] if result else 0
        if count == 0:
            typer.echo(f"No data found for user {user_id}")
            return
        if not confirm:
            typer.confirm(f"Delete {count} events for {user_id}?", abort=True)
        conn.execute("DELETE FROM events WHERE user_id = ?", [user_id])
        conn.execute("CREATE TABLE IF NOT EXISTS deleted_users (user_id VARCHAR, deletion_timestamp BIGINT)")
        conn.execute("INSERT INTO deleted_users VALUES (?, ?)", [user_id, int(time.time())])
        typer.echo(f"Deleted {count} events for user {user_id}")
    finally:
        conn.close()
