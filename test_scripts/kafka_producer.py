#!/usr/bin/env python
"""
Sample Kafka producer for testing the queue-ingest functionality.

This script sends sample event messages to a Kafka topic for testing
the RecSys-Lite queue-based ingest functionality.

Usage:
    python kafka_producer.py [--servers SERVERS] [--topic TOPIC] [--count COUNT]

Options:
    --servers SERVERS   Kafka bootstrap servers [default: localhost:9092]
    --topic TOPIC       Topic name [default: events]
    --count COUNT       Number of messages to send [default: 10]
"""

import argparse
import json
import random
import time
from datetime import datetime
from typing import Any, Dict

try:
    from kafka import KafkaProducer
except ImportError as err:
    raise ImportError(
        "Kafka producer requires kafka-python package. Install it with: pip install kafka-python"
    ) from err


def send_event_messages(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "events",
    message_count: int = 10,
) -> None:
    """Send sample event messages to Kafka.

    Args:
        bootstrap_servers: Comma-separated list of Kafka broker addresses
        topic: Topic name
        message_count: Number of messages to send
    """
    # Create producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )

    # Sample user and item IDs
    user_ids = [f"user_{i}" for i in range(1, 101)]
    item_ids = [f"item_{i}" for i in range(1, 501)]

    # Send messages
    for i in range(message_count):
        # Create a sample event message
        event: Dict[str, Any] = {
            "user_id": random.choice(user_ids),
            "item_id": random.choice(item_ids),
            "qty": random.randint(1, 5),
            "timestamp": datetime.now().isoformat(),
            "session_id": f"session_{random.randint(1000, 9999)}",
            "event_type": "view" if random.random() < 0.7 else "purchase",
        }

        # Send message to Kafka
        producer.send(topic, event)

        print(f"Sent message {i + 1}/{message_count}: {event['user_id']} -> {event['item_id']}")
        time.sleep(0.1)  # Small delay between messages

    # Flush and close producer
    producer.flush()
    producer.close()
    print(f"Successfully sent {message_count} messages to {topic} topic")


def main() -> None:
    """Parse command line arguments and send messages."""
    parser = argparse.ArgumentParser(description="Send sample events to Kafka")
    parser.add_argument("--servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="events", help="Topic name")
    parser.add_argument("--count", type=int, default=10, help="Number of messages to send")

    args = parser.parse_args()

    try:
        send_event_messages(
            bootstrap_servers=args.servers,
            topic=args.topic,
            message_count=args.count,
        )
    except Exception as e:
        print(f"Error sending messages: {e}")
        raise


if __name__ == "__main__":
    main()
