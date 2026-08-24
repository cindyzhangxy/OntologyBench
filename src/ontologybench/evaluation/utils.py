import json
from pathlib import Path
from collections import defaultdict


# ---------------------------
# JSONL loader
# ---------------------------

def load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(line) for line in f]


# ---------------------------
# Qrels loader
# ---------------------------

def load_qrels(path: Path):
    """
    Load qrels.jsonl into:
        { query_id: [doc_id, doc_id, ...] }
    """
    qrels_raw = load_jsonl(path)
    qrels = defaultdict(list)

    for r in qrels_raw:
        qid = r["query_id"]
        for doc_id in r.get("positive_doc_ids", []):
            qrels[qid].append(doc_id)

    return qrels


# ---------------------------
# Task loader
# ---------------------------

def load_task(task_dir: Path):
    """
    Load a task directory and return:
        queries, docs, qrels
    """
    queries = load_jsonl(task_dir / "queries.jsonl")
    docs = load_jsonl(task_dir / "docs.jsonl")
    qrels = load_qrels(task_dir / "qrels.jsonl")

    return queries, docs, qrels


# ---------------------------
# Light text normalization (optional)
# ---------------------------

def normalize(text: str) -> str:
    """
    Minimal normalization:
    - lowercase
    - strip whitespace
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())
