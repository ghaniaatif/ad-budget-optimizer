"""Comment cleaning used before TF-IDF scoring (links, mentions, HTML, timestamps, hashtags, emojis, stretched words)."""
import html
import re

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?:@|\bu/)\w+")
HTML_RE = re.compile(r"<[^>]+>")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
HASHTAG_RE = re.compile(r"#\w+")
REPEAT_RE = re.compile(r"(.)\1{3,}")                            # sooooo -> sooo
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    text = REPEAT_RE.sub(r"\1\1\1", text)
    return SPACE_RE.sub(" ", text).strip()
