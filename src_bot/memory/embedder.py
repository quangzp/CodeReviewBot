"""
Custom sentence-transformers embedder for Graphiti.

Implements graphiti_core's EmbedderClient interface using a local
sentence-transformers model — completely free, no API calls needed.
"""
from __future__ import annotations

from typing import Iterable

from graphiti_core.embedder.client import EmbedderClient


class SentenceTransformerEmbedder(EmbedderClient):
    """
    Drop-in embedder for graphiti-core using local sentence-transformers.
    Model is loaded once and reused across all calls.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            text = str(input_data)
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vecs = self._model.encode(input_data_list, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
