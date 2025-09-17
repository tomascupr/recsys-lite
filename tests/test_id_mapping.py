"""Tests for IdMapper utility."""

import pytest

from recsys_lite.utils import IdMapper


def test_id_mapper_round_trip():
    mapper = IdMapper.from_iterable(["user-1", "user-2", "user-3"])

    assert len(mapper) == 3
    assert mapper.to_index("user-2") == 1
    assert mapper.to_external(1) == "user-2"

    with pytest.raises(KeyError):
        mapper.to_index("missing")


def test_id_mapper_get_or_add():
    mapper = IdMapper()

    idx_a = mapper.get_or_add("a")
    idx_b = mapper.get_or_add("b")
    idx_a_repeat = mapper.get_or_add("a")

    assert idx_a == 0
    assert idx_b == 1
    assert idx_a_repeat == idx_a
    assert mapper.to_external(idx_b) == "b"


def test_id_mapper_persistence():
    mapper = IdMapper.from_iterable(["x", "y"])
    mapper.extend(["z"])

    mapping = mapper.to_dict()
    reverse = mapper.to_reverse_dict()

    restored = IdMapper(mapping)

    assert len(restored) == 3
    assert restored.to_external(2) == "z"
    assert reverse[1] == "y"
