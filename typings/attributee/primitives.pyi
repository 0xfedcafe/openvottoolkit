"""Stub for ``attributee.primitives``.

Each primitive constructor is typed to return its coerced runtime type:

- ``Boolean(...)``   -> ``bool``       (see ``Boolean.coerce``)
- ``Integer(...)``   -> ``int``        (see ``Number.coerce`` with ``int``)
- ``Float(...)``     -> ``float``      (see ``Number.coerce`` with ``float``)
- ``String(...)``    -> ``str``        (see ``String.coerce``)
- ``Number(...)``    -> ``float``      (default conversion is ``_parse_number``)

The descriptor classes still exist at runtime; we just lie about what their
constructors return so that ``foo = Boolean(default=True)`` is typed ``bool``
on the class — which matches what ``self.foo`` evaluates to after
``AttributeeMeta`` strips the descriptor and ``Attributee.__init__`` writes
the coerced value into the instance dict.
"""

from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Union, overload

from attributee import Attribute, CoerceContext

_T = TypeVar("_T")


class Primitive(Attribute):
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Any: ...


class Number(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        conversion: Callable[[Any], Any] = ...,
        val_min: Optional[float] = ...,
        val_max: Optional[float] = ...,
        default: float = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> float: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> float: ...
    @property
    def min(self) -> Optional[float]: ...
    @property
    def max(self) -> Optional[float]: ...


class Integer(Number):
    def __new__(  # type: ignore[misc,override]
        cls,
        val_min: Optional[int] = ...,
        val_max: Optional[int] = ...,
        default: int = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> int: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> int: ...


class Float(Number):
    def __new__(  # type: ignore[misc,override]
        cls,
        val_min: Optional[float] = ...,
        val_max: Optional[float] = ...,
        default: Optional[float] = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> float: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> float: ...


class Boolean(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        default: bool = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> bool: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> bool: ...


class String(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        transformer: Optional[Callable[..., str]] = ...,
        default: Optional[str] = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> str: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Optional[str]: ...
    @property
    def transformer(self) -> Optional[Callable[..., str]]: ...


class URL(String): ...
class Pattern(String): ...


class Enumeration(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        options: Union[Type[Any], Mapping[Any, Any]],
        default: Any = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> Any: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Any: ...
    @property
    def options(self) -> Mapping[Any, Any]: ...
