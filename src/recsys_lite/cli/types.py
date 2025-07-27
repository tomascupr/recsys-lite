"""Shared CLI types for RecSys-Lite."""

from enum import Enum


class ModelType(str, Enum):
    """Available model types."""

    ALS = "als"
    BPR = "bpr"
    ITEM2VEC = "item2vec"
    LIGHTFM = "lightfm"
    GRU4REC = "gru4rec"
    EASE = "ease"
    TEXT_EMBEDDING = "text_embedding"
    HYBRID = "hybrid"


class MetricType(str, Enum):
    """Available evaluation metrics."""

    HR_10 = "hr@10"
    HR_20 = "hr@20"
    NDCG_10 = "ndcg@10"
    NDCG_20 = "ndcg@20"


class QueueType(str, Enum):
    """Available message queue types."""

    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
