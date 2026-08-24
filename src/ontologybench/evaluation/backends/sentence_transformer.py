# backends/sentence_transformer.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerBackend:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def run(self, queries, docs, top_k=10):
        doc_emb = self.model.encode([d["doc"] for d in docs], normalize_embeddings=True)
        doc_ids = [d["doc_id"] for d in docs]

        run = {}
        for q in queries:
            q_emb = self.model.encode(q["query"], normalize_embeddings=True)
            scores = doc_emb @ q_emb
            top = np.argsort(scores)[::-1][:top_k]
            run[q["query_id"]] = [doc_ids[i] for i in top]
        return run
