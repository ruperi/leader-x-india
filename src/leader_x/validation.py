"""Collection-level validation for imported or generated domain records."""

from collections.abc import Iterable

from pydantic import BaseModel


def ensure_unique[T: BaseModel](values: Iterable[T], key: str) -> list[T]:
    """Return values or raise when the selected identity field is duplicated."""
    result = list(values)
    seen: set[object] = set()
    for item in result:
        identity = getattr(item, key)
        if identity in seen:
            raise ValueError(f"duplicate {key}: {identity}")
        seen.add(identity)
    return result
