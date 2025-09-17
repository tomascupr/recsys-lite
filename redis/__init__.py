"""Lightweight Redis stub for test environments without redis-py."""

from __future__ import annotations

from typing import Any


class Redis:  # pragma: no cover - behaviour exercised via mocks
    """Minimal Redis client stub for environments without redis-py."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    def ping(self) -> bool:  # noqa: D401
        return False


def from_url(*args: Any, **kwargs: Any) -> Redis:  # noqa: D401
    return Redis(*args, **kwargs)


__all__ = ["Redis", "from_url"]
