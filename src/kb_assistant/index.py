"""Hybrid search index: BM25 for exact terms, embeddings for meaning, merged with RRF."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its me my no not of on "
    "or our so that the their them then there these they this to us was we were what when where "
    "which who why will with would you your should could have has had".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. CamelCase is also split, so NetworkPolicy matches network policy."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens = re.findall(r"[a-z0-9]+", spaced.lower())
    whole = re.findall(r"[a-z0-9]+", text.lower())
    combined = tokens + [w for w in whole if w not in tokens]
    return [_stem(t) for t in combined if t not in STOPWORDS]


def _stem(token: str) -> str:
    """Conservative plural stripping. A full stemmer hurts technical terms such as ingress."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


def _load_tokenizer(path: Path, max_length: int):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    return tokenizer


def _session(path: Path):
    import onnxruntime

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/download_models.py")
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _inputs(encodings: list) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }


class MiniLmEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings (384 dimensions, quantized ONNX), run locally on CPU."""

    def __init__(self, model_dir: str | Path = "data/models/all-MiniLM-L6-v2"):
        model_dir = Path(model_dir)
        self._session = _session(model_dir / "model_quantized.onnx")
        self._tokenizer = _load_tokenizer(model_dir / "tokenizer.json", 256)

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        inputs = _inputs(encodings)
        tokens = self._session.run(None, inputs)[0]
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (tokens * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)


class MsMarcoReranker:
    """ms-marco-MiniLM-L-6-v2 cross-encoder. It reads the query and passage together, so it's
    slower than embeddings but much better at judging whether a passage answers the question."""

    def __init__(self, model_dir: str | Path = "data/models/ms-marco-MiniLM-L-6-v2", batch: int = 16):
        model_dir = Path(model_dir)
        self._session = _session(model_dir / "model.onnx")
        self._tokenizer = _load_tokenizer(model_dir / "tokenizer.json", 512)
        self._batch = batch

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        scores = []
        for start in range(0, len(texts), self._batch):
            part = texts[start:start + self._batch]
            encodings = self._tokenizer.encode_batch([(query, text) for text in part])
            scores.append(self._session.run(None, _inputs(encodings))[0][:, 0])
        return np.concatenate(scores) if scores else np.zeros(0)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None,
                 reranker: Reranker | None = None):
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker
        self._tokens = [tokenize(c.indexed_text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens)
        self.vectors = vectors if vectors is not None else self._embed_all(embedder, chunks)
        self._categories = np.array([c.category for c in chunks])

    @staticmethod
    def _embed_all(embedder: Embedder, chunks: list[Chunk], batch: int = 32) -> np.ndarray:
        parts = [
            embedder.embed([c.indexed_text for c in chunks[i:i + batch]])
            for i in range(0, len(chunks), batch)
        ]
        return np.vstack(parts)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder, reranker: Reranker | None = None) -> "HybridIndex":
        directory = Path(directory)
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text())]
        return cls(chunks, embedder, np.load(directory / "vectors.npy"), reranker)

    def _mask(self, categories: Iterable[str] | None) -> np.ndarray | None:
        if categories is None:
            return None
        return np.isin(self._categories, list(categories))

    def rank_bm25(self, query: str, mask: np.ndarray | None = None, limit: int = 50) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def rank_dense(self, query: str, mask: np.ndarray | None = None, limit: int = 50) -> list[tuple[int, float]]:
        scores = self.vectors @ self.embedder.embed([query])[0]
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]

    def search(
        self, query: str, k: int = 5, mode: str = "rerank",
        categories: Iterable[str] | None = None, rrf_k: int = 60, candidates: int = 30,
    ) -> list[Hit]:
        """``categories`` is a hard filter applied before ranking, so a caller can never
        retrieve a chunk outside the collections it is allowed to read."""
        mask = self._mask(categories)
        if mode == "bm25":
            ranked = self.rank_bm25(query, mask, limit=k)
            return [Hit(self.chunks[i], s, bm25_rank=r) for r, (i, s) in enumerate(ranked)]
        if mode == "dense":
            ranked = self.rank_dense(query, mask, limit=k)
            return [Hit(self.chunks[i], s, dense_rank=r) for r, (i, s) in enumerate(ranked)]
        if mode not in {"hybrid", "rerank"}:
            raise ValueError(f"Unknown mode {mode!r}.")
        if mode == "rerank":
            if self.reranker is None:
                raise ValueError("Rerank mode needs a reranker.")
            pool = self.search(query, k=candidates, mode="hybrid", categories=categories, rrf_k=rrf_k)
            if not pool:
                return []
            scores = self.reranker.score(query, [hit.chunk.indexed_text for hit in pool])
            order = np.argsort(-scores, kind="stable")[:k]
            return [Hit(pool[i].chunk, float(scores[i]), pool[i].bm25_rank, pool[i].dense_rank) for i in order]

        bm25_ranks = {i: r for r, (i, _) in enumerate(self.rank_bm25(query, mask))}
        dense_ranks = {i: r for r, (i, _) in enumerate(self.rank_dense(query, mask))}
        scores: dict[int, float] = {}
        for ranks in (bm25_ranks, dense_ranks):
            for i, rank in ranks.items():
                scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [Hit(self.chunks[i], scores[i], bm25_ranks.get(i), dense_ranks.get(i)) for i in top]

    def term_coverage(self, query: str, hits: list[Hit]) -> float:
        """IDF-weighted share of query terms found in the hits. Rare terms count most, so a
        question about something the corpus never mentions scores low."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not hits:
            return 0.0
        found: set[str] = set()
        for hit in hits:
            found.update(tokenize(hit.chunk.indexed_text))
        max_idf = max(self._bm25.idf.values())
        weights = {t: self._bm25.idf.get(t, max_idf) for t in terms}
        total = sum(max(w, 1e-6) for w in weights.values())
        return sum(max(weights[t], 1e-6) for t in terms if t in found) / total
