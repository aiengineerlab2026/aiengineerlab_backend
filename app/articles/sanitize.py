"""Article HTML comes from a rich text editor in the browser, so it is cleaned again here before it is stored."""

from __future__ import annotations

import html
import re

import nh3

ALLOWED_TAGS = {
    "p", "br", "hr", "h1", "h2", "h3", "h4", "strong", "b", "em", "i", "u", "s", "code", "pre", "blockquote",
    "ul", "ol", "li", "a", "img",
}
ALLOWED_ATTRIBUTES = {"a": {"href"}, "img": {"src", "alt", "title", "width", "height"}}
ALLOWED_SCHEMES = {"http", "https", "mailto", "data"}  # "data" is only let through for <img src>, see _attribute_filter
_TAG = re.compile(r"<[^>]+>")
_DATA_IMAGE = re.compile(r"^data:image/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+$")  # no SVG: it can carry scripts


def _attribute_filter(tag: str, attr: str, value: str) -> str | None:
    """Pasted images are stored inside the article as data URIs; links and everything else keep to the web schemes."""
    v = value.strip()
    if tag == "img" and attr == "src":
        return v if _DATA_IMAGE.match(v) or v.lower().startswith("https://") else None
    if tag == "img" and attr in ("width", "height"):  # the size set by resizing the picture in the editor
        return v if re.fullmatch(r"\d{1,5}(?:\.\d+)?", v) else None
    if tag == "a" and attr == "href":
        return v if v.lower().startswith(("http://", "https://", "mailto:")) else None
    return value


def clean_html(content: str) -> str:
    return nh3.clean(
        content, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, url_schemes=ALLOWED_SCHEMES,
        link_rel="noopener noreferrer nofollow", attribute_filter=_attribute_filter,
    )


def plain_text(clean: str) -> str:
    """The text of already cleaned HTML, words of neighbouring blocks kept apart."""
    return " ".join(html.unescape(_TAG.sub(" ", clean)).split())


def excerpt(clean: str, length: int = 200) -> str:
    text = plain_text(clean)
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"
