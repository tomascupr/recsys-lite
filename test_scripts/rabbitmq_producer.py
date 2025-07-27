#!/usr/bin/env python
"""
Sample RabbitMQ producer for testing the queue-ingest functionality.

This script sends sample event messages to a RabbitMQ queue for testing
the RecSys-Lite queue-based ingest functionality.

Usage:
    python rabbitmq_producer.py [--host HOST] [--port PORT] [--queue QUEUE] [--count COUNT]

Options:
    --host HOST     RabbitMQ host [default: localhost]
    --port PORT     RabbitMQ port [default: 5672]
    --queue QUEUE   Queue name [default: events]
    --count COUNT   Number of messages to send [default: 10]
"""

import argparse
import json
import random
import time
from datetime import datetime
from typing import Any, Dict

try:
    import pika
except ImportError as err:
    raise ImportError("RabbitMQ producer requires pika package. Install it with: pip install pika") from err


def send_event_messages(
    host: str = "localhost",
    port: int = 5672,
    queue: str = "events",
    username: str = "guest",
    password: str = "guest",
    message_count: int = 10,
) -> None:
    """Send sample event messages to RabbitMQ.

    Args:
        host: RabbitMQ host
        port: RabbitMQ port
        queue: Queue name
        username: RabbitMQ username
        password: RabbitMQ password
        message_count: Number of messages to send
    """
    # Create connection parameters
    credentials = pika.PlainCredentials(username, password)
    parameters = pika.ConnectionParameters(
        host=host,
        port=port,
        credentials=credentials,
    )

    # Establish connection
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Declare the queue
    channel.queue_declare(queue=queue, durable=True)

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

        # Convert to JSON and publish
        message_body = json.dumps(event)
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ),
        )

        print(f"Sent message {i + 1}/{message_count}: {event['user_id']} -> {event['item_id']}")
        time.sleep(0.1)  # Small delay between messages

    # Close connection
    connection.close()
    print(f"Successfully sent {message_count} messages to {queue} queue")


def main() -> None:
    """Parse command line arguments and send messages."""
    parser = argparse.ArgumentParser(description="Send sample events to RabbitMQ")
    parser.add_argument("--host", default="localhost", help="RabbitMQ host")
    parser.add_argument("--port", type=int, default=5672, help="RabbitMQ port")
    parser.add_argument("--queue", default="events", help="Queue name")
    parser.add_argument("--count", type=int, default=10, help="Number of messages to send")

    args = parser.parse_args()

    try:
        send_event_messages(
            host=args.host,
            port=args.port,
            queue=args.queue,
            message_count=args.count,
        )
    except Exception as e:
        print(f"Error sending messages: {e}")
        raise


if __name__ == "__main__":
    main()
