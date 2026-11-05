"""Turn a folder of Markdown documents into heading-aware, citable chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*(?:\{#[^}]*\})?\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_url: str
    category: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def indexed_text(self) -> str:
        """Text used for search. The breadcrumb gives a chunk context it would otherwise lose."""
        crumb = f"{self.title} > {self.section}" if self.section else self.title
        return f"{crumb}\n{self.text}"


def clean_markup(text: str) -> str:
    """Remove Hugo shortcodes, HTML and link syntax while keeping the words people read."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # {{< glossary_tooltip text="Pod" term_id="pod" >}} becomes "Pod".
    text = re.sub(r'\{\{[<%]\s*glossary_tooltip\s+[^>%]*?text="([^"]*)"[^>%]*?[>%]\}\}', r"\1", text)
    text = re.sub(r'\{\{[<%]\s*feature-state\s+[^>%]*?[>%]\}\}', " ", text)
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_front_matter(raw: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[match.end():]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on Markdown headings, ignoring # lines inside code fences."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, clean_markup(heading.group(2)).strip()))
            sections.append((" > ".join(name for _, name in stack), []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "\n".join(lines).strip()]


def split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            pieces.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return pieces


def window_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Pack sentences into windows of at most max_words, repeating some for overlap."""
    sentences = split_sentences(text)
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if words > max_words:
            # A huge "sentence" is usually a table or code block, so cut it by words.
            if current:
                windows.append(" ".join(current))
                current, count = [], 0
            tokens = sentence.split()
            step = max(1, max_words - overlap_words)
            windows.extend(" ".join(tokens[i:i + max_words]) for i in range(0, len(tokens), step))
            continue
        if count + words > max_words and current:
            windows.append(" ".join(current))
            kept: list[str] = []
            kept_count = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if kept_count + previous_words > overlap_words:
                    break
                kept.insert(0, previous)
                kept_count += previous_words
            current, count = kept, kept_count
        current.append(sentence)
        count += words
    if current:
        windows.append(" ".join(current))
    return windows


def doc_url(doc_id: str, base_url: str) -> str:
    slug = re.sub(r"(/_index|/index)?\.md$", "", doc_id)
    return f"{base_url.rstrip('/')}/{slug}/"


def load_chunks(
    root: str | Path,
    base_url: str = "https://kubernetes.io/docs/concepts",
    max_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 8,
) -> list[Chunk]:
    root = Path(root)
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        doc_id = path.relative_to(root).as_posix()
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title = str(meta.get("title") or path.stem.replace("-", " ").title())
        category = doc_id.split("/")[0] if "/" in doc_id else "general"
        index = 0
        for section, section_text in split_sections(body):
            for window in window_words(clean_markup(section_text), max_words, overlap_words):
                if len(window.split()) < min_words:
                    continue
                chunks.append(Chunk(
                    id=f"{doc_id}#{index}", doc_id=doc_id, title=title, section=section,
                    text=window, source_url=doc_url(doc_id, base_url), category=category,
                ))
                index += 1
    return chunks
