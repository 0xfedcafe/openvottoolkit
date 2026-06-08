"""Stub for the ``dominate`` package. Covers only the API used by vot."""

from typing import Any

from dominate import tags as tags
from dominate import util as util
from dominate.dom_tag import dom_tag


class document(dom_tag):
    title: str
    head: dom_tag
    body: dom_tag
    def __init__(self, title: str = ..., doctype: str = ..., *args: Any, **kwargs: Any) -> None: ...
