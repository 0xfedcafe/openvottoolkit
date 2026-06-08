"""Stub for ``dominate.util``. Only the helpers imported by vot are declared."""

from dominate.dom_tag import dom_tag


class text(dom_tag):
    def __init__(self, _text: str, escape: bool = ...) -> None: ...


def raw(s: str) -> text: ...
