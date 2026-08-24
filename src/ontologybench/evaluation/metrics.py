import math


def _validate_k(k):
    if k <= 0:
        raise ValueError(f"k must be positive; received {k}")


def deduplicate_retrieved(retrieved):
    seen = set()
    unique = []
    for doc in retrieved:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique


def dcg_at_k(relevance, k):
    _validate_k(k)
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevance[:k])
    )


def ndcg_at_k(retrieved, gold, k):
    _validate_k(k)
    gold = set(gold)
    if not gold:
        raise ValueError("Cannot compute nDCG for an empty gold set")

    retrieved = deduplicate_retrieved(retrieved)
    relevance = [1 if doc in gold else 0 for doc in retrieved[:k]]
    ideal = [1] * min(len(gold), k)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(relevance, k) / idcg


def hit_at_k(retrieved, gold, k):
    """Return binary Hit@k after order-preserving retrieval deduplication."""
    _validate_k(k)
    gold = set(gold)
    if not gold:
        raise ValueError("Cannot compute Hit@k for an empty gold set")

    retrieved = deduplicate_retrieved(retrieved)
    return 1.0 if any(doc in gold for doc in retrieved[:k]) else 0.0


def evaluate_run(run, qrels, k=10):
    _validate_k(k)
    if not qrels:
        raise ValueError("qrels must not be empty")

    unknown_qids = set(run) - set(qrels)
    if unknown_qids:
        unknown = ", ".join(sorted(str(qid) for qid in unknown_qids))
        raise ValueError(f"Run contains query IDs absent from qrels: {unknown}")

    mrr, hit, ndcg = 0.0, 0.0, 0.0

    for qid, relevant_docs in qrels.items():
        retrieved = deduplicate_retrieved(run.get(qid, []))
        gold = set(relevant_docs)
        if not gold:
            raise ValueError(f"Qrels query {qid!r} has an empty gold set")

        # MRR
        for rank, doc in enumerate(retrieved[:k], start=1):
            if doc in gold:
                mrr += 1.0 / rank
                break

        # Hit@k (binary per query)
        hit += hit_at_k(retrieved, gold, k)

        # nDCG@k
        ndcg += ndcg_at_k(retrieved, gold, k)

    n = len(qrels)
    return {
        f"MRR@{k}": mrr / n,
        f"Hit@{k}": hit / n,
        f"nDCG@{k}": ndcg / n,
    }
