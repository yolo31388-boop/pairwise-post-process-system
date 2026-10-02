"""Pytest helpers shared by the post-process test suite.

The test classes are plain classes (not ``unittest.TestCase`` subclasses)
but use unittest-style assertion helpers, so provide them here without
modifying the test files.
"""


def _assert_is_not_none(self, obj, msg=None):
    if obj is None:
        raise AssertionError(msg or "expected a non-None value")


def pytest_collection_modifyitems(items):
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is not None and not hasattr(cls, "assertIsNotNone"):
            cls.assertIsNotNone = _assert_is_not_none
