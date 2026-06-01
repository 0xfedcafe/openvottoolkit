"""Stub for ``attributee.object``.

Constructors are typed to return their coerced runtime value (the object built
by the resolver) rather than the descriptor instance, matching how
``AttributeeMeta`` strips the field at class creation time and how the
field's ``coerce`` method materialises the value at instance access time.
"""

from datetime import date, datetime
from typing import Any, Callable as PyCallable, Optional, Type, TypeVar

from attributee import Attribute, Attributee, CoerceContext

_T = TypeVar("_T")

ObjectResolver = PyCallable[..., Any]


def import_class(classpath: str) -> Type[Any]: ...
def class_fullname(o: Any) -> str: ...
def class_string(kls: Any) -> str: ...


def default_object_resolver(typename: str, context: Any, **kwargs: Any) -> Attributee: ...


class Object(Attribute):
    # ``Object(resolver)`` produces a class attribute whose instance access yields
    # the object created by ``resolver`` (an ``Attributee`` instance or any custom
    # class returned by a user-supplied resolver). Without static knowledge of
    # the resolver's return type, ``Any`` is the safest white-lie return type.
    def __new__(  # type: ignore[misc]
        cls,
        resolver: ObjectResolver = ...,
        **kwargs: Any,
    ) -> Any: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> Any: ...


class Callable(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> PyCallable[..., Any]: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> PyCallable[..., Any]: ...


class Date(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> date: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> date: ...


class Datetime(Attribute):
    def __new__(  # type: ignore[misc]
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> datetime: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def coerce(self, value: Any, context: Optional[CoerceContext] = ...) -> datetime: ...
