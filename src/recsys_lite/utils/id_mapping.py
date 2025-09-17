"""Utility helpers for mapping external IDs to contiguous indices."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Union


ExternalId = Union[int, str]


class IdMapper:
    """Bidirectional mapping between external identifiers and integer indices."""

    def __init__(self, mapping: Optional[Mapping[ExternalId, int]] = None) -> None:
        self._forward: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

        if mapping:
            for key, value in mapping.items():
                self._add_pair(str(key), int(value))

    @classmethod
    def from_iterable(cls, ids: Iterable[ExternalId]) -> "IdMapper":
        """Create a mapper from an iterable of identifiers."""

        mapping = {str(identifier): index for index, identifier in enumerate(ids)}
        return cls(mapping)

    def _add_pair(self, key: str, index: int) -> None:
        self._forward[key] = index
        self._reverse[index] = key

    def to_index(self, external_id: ExternalId) -> int:
        """Map an external identifier to its index."""

        key = str(external_id)
        if key not in self._forward:
            raise KeyError(f"Unknown identifier: {external_id!r}")
        return self._forward[key]

    def to_external(self, index: int) -> str:
        """Map an internal index back to the external identifier."""

        if index not in self._reverse:
            raise KeyError(f"Unknown index: {index}")
        return self._reverse[index]

    def get_or_add(self, external_id: ExternalId) -> int:
        """Return the index for an identifier, creating one if necessary."""

        key = str(external_id)
        existing = self._forward.get(key)
        if existing is not None:
            return existing

        index = len(self._forward)
        self._add_pair(key, index)
        return index

    def extend(self, ids: Sequence[ExternalId]) -> None:
        """Append a sequence of identifiers to the mapping."""

        for identifier in ids:
            self.get_or_add(identifier)

    def to_dict(self) -> MutableMapping[str, int]:
        """Return a dictionary representation suitable for JSON serialization."""

        return dict(self._forward)

    def to_reverse_dict(self) -> MutableMapping[int, str]:
        """Return the reverse mapping dictionary."""

        return dict(self._reverse)

    def __contains__(self, external_id: object) -> bool:
        return str(external_id) in self._forward

    def __len__(self) -> int:
        return len(self._forward)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"IdMapper(size={len(self)})"
