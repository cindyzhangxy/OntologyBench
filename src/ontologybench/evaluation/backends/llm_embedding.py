import os
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI


class LLMEmbeddingBackend:
    """
    OpenAI embedding backend for OntologyBench.
    Uses frozen LLM embeddings for semantic retrieval, with disk caching.
    """

    def __init__(self, model_name="text-embedding-3-large", batch_size=64,
                 cache_dir=".cache/openai_embeddings"):
        self.model_name = model_name
        self.batch_size = batch_size

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str) -> str:
        h = hashlib.sha1()
        h.update(self.model_name.encode("utf-8"))
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _embed_texts(self, texts):
        embeddings = [None] * len(texts)
        to_compute = []
        to_compute_idx = []

        # 1. Load from cache when possible
        for i, text in enumerate(texts):
            key = self._hash_text(text)
            path = self.cache_dir / f"{key}.npy"

            if path.exists():
                embeddings[i] = np.load(path)
            else:
                to_compute.append(text)
                to_compute_idx.append(i)

        # 2. Compute missing embeddings in batches
        if to_compute:
            for i in tqdm(
                range(0, len(to_compute), self.batch_size),
                desc=f"Embedding ({self.model_name})"
            ):
                batch = to_compute[i : i + self.batch_size]
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                for emb, idx in zip(
                    resp.data,
                    to_compute_idx[i : i + self.batch_size]
                ):
                    vec = np.asarray(emb.embedding, dtype=np.float32)
                    embeddings[idx] = vec

                    key = self._hash_text(texts[idx])
                    np.save(self.cache_dir / f"{key}.npy", vec)

        return np.vstack(embeddings)

    def run(self, queries, docs, top_k=10):
        """
        Args:
            queries: list of {"query_id", "query"}
            docs: list of {"doc_id", "doc"}
            top_k: number of docs to return per query
        Returns:
            run dict {qid: {docid: score}}
        """

        qids = [q["query_id"] for q in queries]
        query_texts = [q["query"] for q in queries]

        dids = [d["doc_id"] for d in docs]
        doc_texts = [d["doc"] for d in docs]

        Q = self._embed_texts(query_texts)
        D = self._embed_texts(doc_texts)

        # cosine normalization
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        D /= np.linalg.norm(D, axis=1, keepdims=True)

        scores = Q @ D.T

        run = {}
        for i, qid in enumerate(qids):
            top_idx = np.argsort(scores[i])[::-1][:top_k]
            run[qid] = [dids[j] for j in top_idx]

        return run
