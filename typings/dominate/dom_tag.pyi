"""Stub for ``dominate.dom_tag``.

The real ``dom_tag.__new__`` returns either a wrapped function (when the tag is
used as a decorator) or a tag instance, so pyright infers a union that loses the
context-manager protocol — ``with table() as element:`` then fails to type check.
This stub types construction as always returning an instance, which is the only
form vot uses.
"""

from types import TracebackType
from typing import Any


class dom_tag:
    attributes: dict[str, Any]
    children: list[Any]
    parent: "dom_tag | None"
    is_single: bool
    is_pretty: bool
    is_inline: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> "dom_tag": ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...
    def add(self, *args: Any) -> Any: ...
    def set_attribute(self, key: int | str, value: Any) -> None: ...
    def get(self, attr: Any = ..., direct: bool = ..., **kwargs: Any) -> list[Any]: ...
    def render(self, indent: str = ..., pretty: bool = ..., xhtml: bool = ...) -> str: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
