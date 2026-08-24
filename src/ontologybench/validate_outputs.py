import json
import re
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path(__file__).resolve().parent / "data" / "tasks_global"

# ===========================
# Helpers
# ===========================

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(x) for x in f]


def normalize(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


# ===========================
# QC State
# ===========================

QC = {
    "tasks": {},
    "warnings": defaultdict(int),
    "examples": defaultdict(set),
}


def warn(kind, example=None):
    if example is None:
        QC["warnings"][kind] += 1
    else:
        QC["examples"][kind].add(example)


def reset_qc():
    QC["tasks"].clear()
    QC["warnings"].clear()
    QC["examples"].clear()


# ===========================
# Structural Checks
# ===========================

def check_required_fields(items, fields):
    for f in fields:
        for x in items:
            if not x.get(f):
                return False
    return True


def find_duplicate_text(items, field):
    seen = set()
    dups = set()
    for x in items:
        t = normalize(x.get(field, ""))
        if t in seen:
            dups.add(t)
        else:
            seen.add(t)
    return dups


def check_unique_ids(docs):
    ids = [d["doc_id"] for d in docs]
    return len(ids) == len(set(ids))


def check_qrels_consistency(queries, docs, qrels):
    q_ids = {q["query_id"] for q in queries}
    d_ids = {d["doc_id"] for d in docs}

    for r in qrels:
        if r["query_id"] not in q_ids:
            return False
        for d in r.get("positive_doc_ids", []):
            if d not in d_ids:
                return False
    return True


# ===========================
# Leakage Checks (EDGE-ONLY)
# ===========================

def check_train_eval_leakage(tier1, eval_tasks):
    """
    Allowed:
      - identical query text across splits
      - identical query IDs with different positives

    Forbidden:
      - reuse of identical (query_id, positive_doc_id) supervision edges
    """

    train_edges = set()

    for _, _, _, qrels in tier1:
        for r in qrels:
            for pos in r.get("positive_doc_ids", []):
                train_edges.add((r["query_id"], pos))

    for _, _, _, qrels in eval_tasks:
        for r in qrels:
            for pos in r.get("positive_doc_ids", []):
                if (r["query_id"], pos) in train_edges:
                    warn("edge_leakage")


def check_cross_eval_overlap(eval_tasks, max_examples=5):
    seen_queries = {}

    for name, queries, _, _ in eval_tasks:
        qs = {normalize(q["query"]) for q in queries}

        for other_qs in seen_queries.values():
            overlap = qs & other_qs
            if overlap:
                warn("cross_eval_query_overlap")
                for q in sorted(overlap)[:max_examples]:
                    warn("cross_eval_query_overlap", q)

        seen_queries[name] = qs


# ===========================
# Task Validator
# ===========================

def validate_task(task_dir, max_examples=5):
    name = task_dir.name

    docs = load_jsonl(task_dir / "docs.jsonl")
    queries = load_jsonl(task_dir / "queries.jsonl")
    qrels = load_jsonl(task_dir / "qrels.jsonl")

    QC["tasks"][name] = {
        "docs": len(docs),
        "queries": len(queries),
        "qrels": len(qrels),
    }

    if not check_required_fields(docs, ["doc_id", "doc"]):
        warn("missing_doc_fields")

    if not check_required_fields(queries, ["query_id", "query"]):
        warn("missing_query_fields")

    if not check_required_fields(qrels, ["query_id", "positive_doc_ids"]):
        warn("missing_qrels_fields")

    dup_q = find_duplicate_text(queries, "query")
    if dup_q:
        warn("duplicate_queries")
        for q in sorted(dup_q)[:max_examples]:
            warn("duplicate_queries", q)

    dup_d = find_duplicate_text(docs, "doc")
    if dup_d:
        warn("duplicate_docs")

    if not check_unique_ids(docs):
        warn("duplicate_doc_ids")

    if not check_qrels_consistency(queries, docs, qrels):
        warn("invalid_qrels")

    return name, queries, docs, qrels


# ===========================
# Main
# ===========================

def main():
    reset_qc()
    tasks = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())

    tier1 = []
    eval_tasks = []

    for t in tasks:
        name, q, d, r = validate_task(t)
        if name.startswith(("1A", "1B", "1C")):
            tier1.append((name, q, d, r))
        else:
            eval_tasks.append((name, q, d, r))

    check_train_eval_leakage(tier1, eval_tasks)
    check_cross_eval_overlap(eval_tasks)

    # ===========================
    # FINAL COMPACT SUMMARY
    # ===========================

    total_docs = sum(v["docs"] for v in QC["tasks"].values())
    total_queries = sum(v["queries"] for v in QC["tasks"].values())
    total_qrels = sum(v["qrels"] for v in QC["tasks"].values())

    print("\n==============================")
    print("OntologyBench QC Summary")
    print("==============================")
    print(f"Tasks validated : {len(QC['tasks'])}")
    print(f"Total documents : {total_docs}")
    print(f"Total queries   : {total_queries}")
    print(f"Total qrels    : {total_qrels}")
    print(f"Issue types    : {len(QC['warnings'])}")
    print(f"Total issues   : {sum(QC['warnings'].values())}")

    if QC["warnings"]:
        print("Issue breakdown:")
        for k, v in sorted(QC["warnings"].items()):
            print(f"  - {k}: {v}")

    for k in ("duplicate_queries", "cross_eval_query_overlap"):
        if QC["examples"].get(k):
            print(f"\nExamples for {k}:")
            for ex in sorted(QC["examples"][k]):
                print(f"  • {ex}")

    print(f"QC status      : {'PASSED' if not QC['warnings'] else 'COMPLETED WITH WARNINGS'}")
    print("==============================\n")


if __name__ == "__main__":
    main()
