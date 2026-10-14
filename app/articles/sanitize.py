"""Article HTML comes from a rich text editor in the browser, so it is cleaned again here before it is stored."""

from __future__ import annotations

import html
import re

import nh3

ALLOWED_TAGS = {
    "p", "br", "hr", "h1", "h2", "h3", "h4", "strong", "b", "em", "i", "u", "s", "code", "pre", "blockquote",
    "ul", "ol", "li", "a",
}
ALLOWED_ATTRIBUTES = {"a": {"href"}}
ALLOWED_SCHEMES = {"http", "https", "mailto"}
_TAG = re.compile(r"<[^>]+>")


def clean_html(content: str) -> str:
    return nh3.clean(
        content, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, url_schemes=ALLOWED_SCHEMES,
        link_rel="noopener noreferrer nofollow",
    )


def plain_text(clean: str) -> str:
    """The text of already cleaned HTML, words of neighbouring blocks kept apart."""
    return " ".join(html.unescape(_TAG.sub(" ", clean)).split())


def excerpt(clean: str, length: int = 200) -> str:
    text = plain_text(clean)
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"
