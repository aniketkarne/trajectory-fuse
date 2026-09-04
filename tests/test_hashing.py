"""Tests for canonical hashing."""

from trajectory_fuse.hashing import canonical_hash


class TestCanonicalHash:
    def test_dict_key_order_irrelevant(self):
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_nested_dict_order_irrelevant(self):
        a = {"outer": {"x": 1, "y": [1, 2, 3]}, "k": "v"}
        b = {"k": "v", "outer": {"y": [1, 2, 3], "x": 1}}
        assert canonical_hash(a) == canonical_hash(b)

    def test_list_vs_tuple_equal(self):
        assert canonical_hash([1, 2, 3]) == canonical_hash((1, 2, 3))

    def test_set_order_irrelevant(self):
        assert canonical_hash({1, 2, 3}) == canonical_hash({3, 1, 2})

    def test_distinct_payloads_differ(self):
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
        assert canonical_hash({"a": 1}) != canonical_hash({"b": 1})

    def test_string_stable(self):
        assert canonical_hash("hello") == canonical_hash("hello")

    def test_deterministic_length(self):
        h = canonical_hash({"k": "v"})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_unicode_safe(self):
        assert canonical_hash({"name": "café"}) == canonical_hash({"name": "café"})

    def test_rejects_unsupported_type(self):
        import pytest

        with pytest.raises(TypeError):
            canonical_hash(object())

    def test_none_handled(self):
        assert canonical_hash(None) == canonical_hash(None)
        assert canonical_hash(None) != canonical_hash({})