"""Stub for the ``attributee`` package.

Constructors are typed to return the *coerced* runtime type rather than the
attribute descriptor class. This matches the behaviour of ``AttributeeMeta``
which pops every ``Attribute``-typed class field at class-creation time, so
that instance access (``self.bounded`` etc.) returns the value produced by
``Attribute.coerce`` rather than the descriptor instance itself.

If you ever need the raw descriptor for introspection, use ``cast()`` or fall
back to the runtime types via ``# type: ignore``.
"""

from typing import Any, Callable as _PyCallable, Dict, List as _PyList, Mapping, Optional, Tuple as _PyTuple, Type, TypeVar, overload

from .primitives import Boolean as Boolean
from .primitives import Enumeration as Enumeration
from .primitives import Float as Float
from .primitives import Integer as Integer
from .primitives import Number as Number
from .primitives import Pattern as Pattern
from .primitives import Primitive as Primitive
from .primitives import String as String
from .primitives import URL as URL

# Re-exports from submodules so ``from attributee import Object, List, ...`` is
# recognised by the type checker (these are re-exported at the bottom of the
# real ``attributee/__init__.py``).
from .object import Callable as Callable
from .object import Date as Date
from .object import Datetime as Datetime
from .object import Object as Object
from .containers import List as List
from .containers import Map as Map
from .containers import Tuple as Tuple

_T = TypeVar("_T")


class AttributeException(Exception):
    def __init__(self, *args: object) -> None: ...


class AttributeParseException(AttributeException):
    def __init__(self, cause: BaseException, key: Any) -> None: ...


class Singleton(type): ...


class CoerceContext:
    def __init__(self, parent: Optional["Attributee"] = ..., key: Optional[Any] = ...) -> None: ...
    # ``parent`` is typed as ``Any`` so downstream attribute access on the
    # owning ``Attributee`` (e.g. ``context.parent.directory``) does not need
    # a stub for every concrete subclass.
    @property
    def parent(self) -> Any: ...
    @property
    def key(self) -> Optional[Any]: ...


class Undefined(metaclass=Singleton): ...


def is_undefined(a: Any) -> bool: ...
def is_instance_or_subclass(val: Any, class_: Any) -> bool: ...


class Attribute:
    def __init__(self, default: Any = ..., description: str = ..., readonly: bool = ...) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Any: ...
    def dump(self, value: Any) -> Any: ...
    @property
    def description(self) -> str: ...
    @property
    def readonly(self) -> bool: ...
    @property
    def required(self) -> bool: ...


# Note: the real ``attributee`` package defines ``class Any(Attribute)``. We omit
# it from the stub because (a) no vot code imports it and (b) the name would
# shadow ``typing.Any`` within this stub.


class _NestedFactory:
    """Pyright sees ``Nested`` as a callable factory; the white-lie return type
    matches the *coerced* runtime value (an instance of the wrapped Attributee
    class), so ``self.realtime.grace`` resolves correctly.

    The real attributee ``Nested`` is a class — runtime ``isinstance`` checks
    against ``Nested`` still work because at runtime there is no stub.
    """
    def __call__(
        self,
        acls: Type[_T],
        override: Optional[Mapping[str, Any]] = ...,
        create: bool = ...,
        default: Any = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> _T: ...


Nested: _NestedFactory


class AttributeeMeta(type): ...


class Collector(Attribute):
    def filter(self, object: "Attributee", **kwargs: Any) -> dict[str, Any]: ...


class _IncludeFactory:
    """See :class:`_NestedFactory`. ``Include(SomeAttributee)`` is typed as the
    wrapped class instance type."""
    def __call__(
        self,
        acls: Type[_T],
        override: Optional[Mapping[str, Any]] = ...,
        create: bool = ...,
        default: Any = ...,
        description: str = ...,
        readonly: bool = ...,
    ) -> _T: ...


Include: _IncludeFactory


class Attributee(metaclass=AttributeeMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def dump(self) -> dict[str, Any]: ...
    @classmethod
    def list_attributes(cls) -> list[tuple[str, Attribute]]: ...
    @classmethod
    def attributes(cls) -> dict[str, Attribute]: ...


class Unclaimed(Collector): ...
