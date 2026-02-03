"""
Caption cleaning utilities.
"""

from __future__ import annotations

import html
import re


HTML_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    text = html.unescape(text or "")
    text = HTML_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()
