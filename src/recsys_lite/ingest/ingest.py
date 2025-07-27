"""Data ingestion functionality for RecSys-Lite."""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import pyarrow as pa

# Configure logging
logger = logging.getLogger(__name__)


def ingest_data(events_path: Path, items_path: Path, db_path: Path) -> None:
    """Ingest data into DuckDB database.

    Args:
        events_path: Path to events parquet file
        items_path: Path to items CSV file
        db_path: Path to DuckDB database
    """
    conn = duckdb.connect(str(db_path))
    try:
        # Create events table
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS events AS
            SELECT * FROM read_parquet('{events_path}')
            """
        )

        # Create items table
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS items AS
            SELECT * FROM read_csv('{items_path}')
            """
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Streaming ingest helpers
# ---------------------------------------------------------------------------


def stream_events(
    events_dir: Path,
    db_path: Path,
    poll_interval: int = 5,
) -> None:
    """Continuously ingest parquet files dropped into a directory.

    The function performs a very simple *file‑based* streaming ingestion.  Any
    ``*.parquet`` file that appears in *events_dir* is **appended** to the
    ``events`` table inside the DuckDB database.  Already processed files are
    tracked in‑memory for the lifetime of the process, so the same file will
    not be imported twice.

    The implementation purposefully avoids external dependencies (e.g.
    ``watchdog``) – it just polls the directory every *poll_interval* seconds
    which is usually sufficient for low‑volume, near‑real‑time pipelines.

    Args:
        events_dir: Directory to watch for new parquet files.
        db_path:   Path to the DuckDB database containing an ``events`` table.
        poll_interval: Number of seconds to wait between directory scans.
    """

    events_dir = events_dir.expanduser().resolve()
    processed: set[str] = set()

    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory '{events_dir}' does not exist")

    logger.info(f"[stream-ingest] Watching '{events_dir}' for parquet files. Press Ctrl+C to stop.")
    print(f"[stream-ingest] Watching '{events_dir}' for parquet files. Press Ctrl+C to stop.")

    try:
        while True:
            # Discover parquet files that have not been processed yet
            for parquet_file in sorted(events_dir.glob("*.parquet")):
                if parquet_file.name in processed:
                    continue

                try:
                    _append_parquet_to_events(parquet_file, db_path)
                    processed.add(parquet_file.name)
                    logger.info(f"[stream-ingest] Ingested {parquet_file.name}")
                    print(f"[stream-ingest] Ingested {parquet_file.name}")
                except Exception as exc:  # pragma: no cover – guard rail only
                    # We do not want the outer loop to die because of one bad file
                    logger.error(f"[stream-ingest] Failed to ingest {parquet_file.name}: {exc}")
                    print(f"[stream-ingest] Failed to ingest {parquet_file.name}: {exc}")

            # Sleep before the next scan
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[stream-ingest] Stopped – goodbye!")


def _append_parquet_to_events(parquet_file: Path, db_path: Path) -> None:
    """Helper that appends the content of *parquet_file* into ``events`` table."""

    conn = duckdb.connect(str(db_path))
    try:
        # Ensure the events table exists – if not, create it on‑the‑fly.
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS events AS
            SELECT * FROM read_parquet('{parquet_file}') WHERE 0=1
            """
        )

        # Append the actual data
        conn.execute(f"INSERT INTO events SELECT * FROM read_parquet('{parquet_file}')")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Message Queue Integration
# ---------------------------------------------------------------------------


class MessageQueueConsumer(ABC):
    """Abstract base class for message queue consumers."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the message queue."""
        pass

    @abstractmethod
    def consume(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Consume messages from the queue.

        Args:
            batch_size: Number of messages to consume at once

        Returns:
            List of consumed messages as dictionaries
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the message queue."""
        pass


class RabbitMQConsumer(MessageQueueConsumer):
    """RabbitMQ implementation of the message queue consumer."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        queue: str = "events",
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        prefetch_count: int = 100,
    ) -> None:
        """Initialize RabbitMQ consumer.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            queue: Queue name to consume from
            username: RabbitMQ username
            password: RabbitMQ password
            virtual_host: RabbitMQ virtual host
            prefetch_count: Number of messages to prefetch
        """
        try:
            import importlib.util

            if importlib.util.find_spec("pika") is None:
                raise ImportError("pika package not found")
        except ImportError as err:
            raise ImportError(
                "RabbitMQ support requires pika package. Install it with: pip install recsys-lite[mq]"
            ) from err

        self.host = host
        self.port = port
        self.queue = queue
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.prefetch_count = prefetch_count

        self.connection: Optional[Any] = None
        self.channel: Optional[Any] = None

    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            import pika

            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare the queue (create if doesn't exist)
            self.channel.queue_declare(queue=self.queue, durable=True)

            # Set prefetch count
            self.channel.basic_qos(prefetch_count=self.prefetch_count)

            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}, queue: {self.queue}")
        except ImportError as err:
            raise ImportError(
                "RabbitMQ support requires pika package. Install it with: pip install recsys-lite[mq]"
            ) from err
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def consume(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Consume messages from RabbitMQ queue.

        Args:
            batch_size: Maximum number of messages to consume

        Returns:
            List of consumed messages as dictionaries
        """

        if self.channel is None:
            self.connect()

        # Ensure channel is initialized
        assert self.channel is not None, "Channel must be initialized before consuming"

        messages: List[Dict[str, Any]] = []

        for _ in range(batch_size):
            method_frame: Any
            header_frame: Any
            body: bytes

            method_frame, header_frame, body = self.channel.basic_get(queue=self.queue, auto_ack=False)

            if method_frame:
                try:
                    message = json.loads(body.decode("utf-8"))
                    messages.append(message)

                    # Acknowledge the message
                    self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                except Exception as e:
                    # Reject the message in case of processing error
                    self.channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                    logger.error(f"Error processing message: {e}")
            else:
                # No more messages in the queue
                break

        return messages

    def close(self) -> None:
        """Close the connection to RabbitMQ."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.channel = None


class KafkaConsumer(MessageQueueConsumer):
    """Kafka implementation of the message queue consumer."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "events",
        group_id: str = "recsys-lite",
        auto_offset_reset: str = "latest",
    ) -> None:
        """Initialize Kafka consumer.

        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            topic: Topic to consume from
            group_id: Consumer group ID
            auto_offset_reset: Offset reset policy ('earliest' or 'latest')
        """
        try:
            import importlib.util

            if importlib.util.find_spec("kafka") is None:
                raise ImportError("kafka-python package not found")
        except ImportError as err:
            raise ImportError(
                "Kafka support requires kafka-python package. Install it with: pip install recsys-lite[mq]"
            ) from err

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset

        self.consumer: Optional[Any] = None

    def connect(self) -> None:
        """Establish connection to Kafka."""
        try:
            from kafka import KafkaConsumer as KConsumer

            self.consumer = KConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                enable_auto_commit=False,
            )

            logger.info(f"Connected to Kafka at {self.bootstrap_servers}, topic: {self.topic}, group: {self.group_id}")
        except ImportError as err:
            raise ImportError(
                "Kafka support requires kafka-python package. Install it with: pip install recsys-lite[mq]"
            ) from err
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def consume(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Consume messages from Kafka topic.

        Args:
            batch_size: Maximum number of messages to consume

        Returns:
            List of consumed messages as dictionaries
        """
        if self.consumer is None:
            self.connect()

        assert self.consumer is not None

        messages: List[Dict[str, Any]] = []

        # Poll for messages with a timeout of 1 second
        message_pack = self.consumer.poll(timeout_ms=1000, max_records=batch_size)

        for topic_partition, records in message_pack.items():
            for record in records:
                messages.append(record.value)
                self.consumer.commit({topic_partition: record.offset + 1})

        return messages

    def close(self) -> None:
        """Close the connection to Kafka."""
        if self.consumer is not None:
            self.consumer.close()
            self.consumer = None


def create_consumer(queue_type: str, config: Dict[str, Any]) -> MessageQueueConsumer:
    """Factory function to create the appropriate message queue consumer.

    Args:
        queue_type: Type of message queue ('rabbitmq' or 'kafka')
        config: Configuration parameters for the selected queue type

    Returns:
        Initialized message queue consumer
    """
    if queue_type.lower() == "rabbitmq":
        return RabbitMQConsumer(**config)
    elif queue_type.lower() == "kafka":
        return KafkaConsumer(**config)
    else:
        raise ValueError(f"Unsupported queue type: {queue_type}")


def process_event_messages(
    messages: List[Dict[str, Any]],
    db_path: Path,
    batch_size: int = 100,
) -> None:
    """Process event messages from a message queue and insert into DuckDB.

    Args:
        messages: List of event messages to process
        db_path: Path to DuckDB database
        batch_size: Size of batches for processing
    """
    if not messages:
        return

    # Convert messages to DataFrame
    df = pd.DataFrame(messages)

    # Ensure required columns exist
    required_columns = ["user_id", "item_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns in messages: {missing_columns}")
        return

    # Insert into DuckDB in batches
    total_messages = len(df)
    total_batches = (total_messages + batch_size - 1) // batch_size

    conn = duckdb.connect(str(db_path))
    try:
        # Ensure events table exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                user_id VARCHAR,
                item_id VARCHAR,
                qty INTEGER DEFAULT 1,
                timestamp TIMESTAMP
            )
            """
        )

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_messages)
            batch_df = df.iloc[start_idx:end_idx]

            # Ensure qty column exists, default to 1
            if "qty" not in batch_df.columns:
                batch_df["qty"] = 1

            # Ensure timestamp column exists, default to current time
            if "timestamp" not in batch_df.columns:
                batch_df["timestamp"] = pd.Timestamp.now()

            # Convert to Arrow Table for efficient insertion
            batch_table = pa.Table.from_pandas(batch_df)

            # Insert into DuckDB
            conn.execute(
                """
                INSERT INTO events 
                SELECT * FROM batch_table
                """,
                {"batch_table": batch_table},
            )

        logger.info(f"Inserted {total_messages} events into DuckDB")
    except Exception as e:
        logger.error(f"Error inserting events into DuckDB: {e}")
        raise
    finally:
        conn.close()


def queue_ingest(
    queue_config: Dict[str, Any],
    db_path: Path,
    queue_type: str = "rabbitmq",
    batch_size: int = 100,
    poll_interval: int = 5,
) -> None:
    """Continuously consume events from a message queue and ingest them.

    Args:
        queue_config: Configuration for the message queue connection
        db_path: Path to DuckDB database
        queue_type: Type of message queue ('rabbitmq' or 'kafka')
        batch_size: Number of messages to process in a batch
        poll_interval: Seconds to wait between polling if no messages
    """
    try:
        # Create consumer based on queue type
        consumer = create_consumer(queue_type, queue_config)
        consumer.connect()

        print(f"[queue-ingest] Connected to {queue_type} queue. Processing messages. Press Ctrl+C to stop.")
        logger.info(f"[queue-ingest] Connected to {queue_type} queue. Processing messages.")

        while True:
            try:
                # Consume batch of messages
                messages = consumer.consume(batch_size=batch_size)

                if messages:
                    # Process and insert messages
                    process_event_messages(messages, db_path, batch_size)
                    logger.info(f"[queue-ingest] Processed {len(messages)} messages")
                    print(f"[queue-ingest] Processed {len(messages)} messages")
                else:
                    # No messages, wait before polling again
                    time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"[queue-ingest] Error processing messages: {e}")
                print(f"[queue-ingest] Error processing messages: {e}")
                # Sleep before retry to avoid hammering the queue
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[queue-ingest] Stopped – goodbye!")
    finally:
        # Ensure consumer is closed
        if "consumer" in locals():
            consumer.close()  # No need for cast, consumer is already of MessageQueueConsumer type
