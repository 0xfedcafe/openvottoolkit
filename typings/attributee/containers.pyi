"""Stub for ``attributee.containers``.

Container constructors are typed to return their coerced runtime container —
a list, a tuple, or a mapping — rather than the descriptor instance.
"""

from typing import Any, Dict, List as _PyList, Mapping, Optional, Tuple as _PyTuple, TypeVar

from attributee import Attribute, CoerceContext

_T = TypeVar("_T")


class ReadonlySequence: ...


class CoerceSequence(ReadonlySequence): ...


class ReadonlyMapping: ...


class CoerceMapping(ReadonlyMapping): ...


class Tuple(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> tuple: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> tuple: ...


class List(Attribute):
    # ``List(item_attr)`` creates a field whose instance access yields a list. The
    # element type matches the wrapped attribute's coerced type, but pyright
    # cannot inspect that, so we use ``list`` (the runtime type) here.
    #
    # The ``item`` argument is typed ``Any`` because callers sometimes pass an
    # already-coerced descriptor instance (e.g. ``List(Object(resolver))``) which
    # pyright reads as the descriptor's *return* type rather than its class.
    def __new__(  # type: ignore[misc]
        cls,
        item: Any = ...,
        **kwargs: Any,
    ) -> list: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> list: ...


class Map(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        item: Any = ...,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Dict[str, Any]: ...
