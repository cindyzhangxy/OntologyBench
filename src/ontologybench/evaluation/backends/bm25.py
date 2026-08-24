# backends/bm25.py
from rank_bm25 import BM25Okapi

class BM25Backend:
    def run(self, queries, docs, top_k=10):
        corpus = [d["doc"].lower().split() for d in docs]
        doc_ids = [d["doc_id"] for d in docs]
        bm25 = BM25Okapi(corpus)

        run = {}
        for q in queries:
            scores = bm25.get_scores(q["query"].lower().split())
            top = sorted(range(len(scores)),
                         key=lambda i: scores[i],
                         reverse=True)[:top_k]
            run[q["query_id"]] = [doc_ids[i] for i in top]
        return run
