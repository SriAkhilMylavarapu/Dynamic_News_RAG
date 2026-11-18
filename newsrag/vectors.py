"""Embeddings and the Chroma vector index."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """ONNX sentence embeddings via fastembed (no PyTorch, quick on CPU)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds the model's query instruction (e.g. for BGE models).
        return next(iter(self.model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Not semantically smart; used for tests and for running offline when the
    embedding model can't be downloaded.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embedder(model_name: str) -> Embedder:
    if model_name == "hash":
        return HashEmbedder()
    return FastEmbedder(model_name)


def _where(
    since: datetime | None = None,
    categories: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    exclude_source: str | None = None,
) -> dict | None:
    clauses: list[dict] = []
    if since is not None:
        clauses.append({"published_ts": {"$gte": since.timestamp()}})
    if categories:
        clauses.append({"category": {"$in": list(categories)}})
    if sources:
        clauses.append({"source_lc": {"$in": [s.lower() for s in sources]}})
    if exclude_source:
        clauses.append({"source_lc": {"$ne": exclude_source.lower()}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorIndex:
    """Two Chroma collections: one vector per chunk (for retrieval) and one per
    article lead (title + opening, for grouping articles into stories)."""

    def __init__(self, path: Path | str | None, model_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = ChromaSettings(anonymized_telemetry=False)
        if path is None:
            self.client = chromadb.EphemeralClient(settings=settings)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path), settings=settings)
        # Vectors from different models are incompatible, so name collections by model.
        suffix = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()[:40]
        cfg = {"hnsw": {"space": "cosine"}}
        self.chunks = self.client.get_or_create_collection(f"chunks-{suffix}", embedding_function=None, configuration=cfg)
        self.leads = self.client.get_or_create_collection(f"leads-{suffix}", embedding_function=None, configuration=cfg)

    def add_chunks(self, ids: Sequence[str], embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict]) -> None:
        if ids:
            self.chunks.upsert(ids=list(ids), embeddings=list(embeddings), metadatas=list(metadatas))

    def add_lead(self, article_id: str, embedding: Sequence[float], metadata: dict) -> None:
        self.leads.upsert(ids=[article_id], embeddings=[list(embedding)], metadatas=[metadata])

    def search_chunks(
        self,
        embedding: Sequence[float],
        limit: int,
        since: datetime | None = None,
        categories: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Returns (chunk_id, cosine similarity)."""
        if self.chunks.count() == 0:
            return []
        res = self.chunks.query(
            query_embeddings=[list(embedding)],
            n_results=min(limit, self.chunks.count()),
            where=_where(since, categories, sources),
            include=["distances"],
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def cosine_for(self, chunk_ids: Sequence[str], embedding: Sequence[float]) -> dict[str, float]:
        """Cosine similarity between the query and specific chunks (e.g. ones only BM25 found)."""
        if not chunk_ids:
            return {}
        got = self.chunks.get(ids=list(chunk_ids), include=["embeddings"])
        if not got["ids"]:
            return {}
        q = np.asarray(embedding, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        mat = np.asarray(got["embeddings"], dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
        return dict(zip(got["ids"], (mat @ q).tolist()))

    def nearest_lead(
        self, embedding: Sequence[float], since: datetime, exclude_source: str | None = None
    ) -> tuple[str, float, dict] | None:
        """Most similar recent article lead, optionally from a different outlet."""
        if self.leads.count() == 0:
            return None
        res = self.leads.query(
            query_embeddings=[list(embedding)],
            n_results=1,
            where=_where(since=since, exclude_source=exclude_source),
            include=["distances", "metadatas"],
        )
        if not res["ids"][0]:
            return None
        return res["ids"][0][0], 1.0 - res["distances"][0][0], res["metadatas"][0][0]

    def delete(self, article_ids: Sequence[str], chunk_ids: Sequence[str]) -> None:
        for i in range(0, len(chunk_ids), 1000):
            self.chunks.delete(ids=list(chunk_ids[i : i + 1000]))
        for i in range(0, len(article_ids), 1000):
            self.leads.delete(ids=list(article_ids[i : i + 1000]))

    def reset(self) -> None:
        for col in (self.chunks, self.leads):
            ids = col.get(include=[])["ids"]
            for i in range(0, len(ids), 1000):
                col.delete(ids=ids[i : i + 1000])
