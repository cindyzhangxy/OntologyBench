import pandas as pd
from pathlib import Path
import re
import numpy as np

from ontologybench.utils.io import save_jsonl
from ontologybench.utils.text import join_with_and

OUTDIR = Path("ontologybench/data/tasks/1A-R_gene_document_retrieval")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_summary(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[provided.*?\]", "", text, flags=re.IGNORECASE).strip()


def clean_gene_group(group):
    if group is None:
        return None
    if isinstance(group, (list, tuple, np.ndarray)):
        items = list(group)
    else:
        g = str(group).strip("[]{}()")
        items = [p.strip() for p in re.split(r"[;,]", g)]
    cleaned = [x for x in items if x]
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned if cleaned else None


def build_gene_document(row):
    loc = row["location"]
    group = clean_gene_group(row["gene_group"])
    summary = clean_summary(row["Summary"])

    doc = f"The gene is a protein-coding gene located on chromosome {loc}."
    if group:
        doc += f" It belongs to the following gene groups: {join_with_and(group)}."
    if summary:
        doc += f" Functionally, {summary}"

    return doc.strip()


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

def build_gene_document_retrieval(df: pd.DataFrame):
    """
    Task 1A-R (Tier 1 — Concept Grounding; training-only): Gene Document Retrieval

    docs.jsonl:
        {doc_id: ENTREZ_ID, doc: gene_document}

    queries.jsonl:
        {query_id: ENTREZ_ID, query: natural-language gene function question}

    qrels.jsonl:
        {query_id: ENTREZ_ID, positive_doc_ids: [ENTREZ_ID]}
    """

    required = [
        "entrez_id", "symbol", "location",
        "gene_group", "Summary"
    ]

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["entrez_id", "symbol", "location"])

    # -----------------------------------------------------
    # Build rendered documents
    # -----------------------------------------------------
    df = df.copy()
    df["gene_document"] = df.apply(build_gene_document, axis=1)

    # deterministic collapse
    df = df.sort_values("entrez_id")
    df = df.drop_duplicates(subset=["gene_document"], keep="first")
    df = df.drop_duplicates(subset=["entrez_id"], keep="first")

    if df.empty:
        raise ValueError("No gene documents generated.")

    docs = []
    queries = []
    qrels = []

    # -----------------------------------------------------
    # Build docs, queries, qrels
    # -----------------------------------------------------
    for _, row in df.iterrows():
        gid = str(row["entrez_id"]).strip()
        symbol = str(row["symbol"]).strip()
        doc = row["gene_document"]

        if not gid:
            continue
        if not symbol:
            continue
        if not isinstance(doc, str) or not doc.strip():
            continue

        query_text = f"{symbol}"

        # hard guarantee: non-empty query
        if not query_text.strip():
            continue

        docs.append({
            "doc_id": gid,
            "doc": doc
        })


        queries.append({
            "query_id": gid,
            "query": query_text
        })

        qrels.append({
            "query_id": gid,
            "positive_doc_ids": [gid]
        })

    # -----------------------------------------------------
    # Final guards
    # -----------------------------------------------------
    if not queries:
        raise ValueError("No non-empty queries generated.")

    if not docs:
        raise ValueError("No documents generated.")

    doc_ids = {d["doc_id"] for d in docs}
    for r in qrels:
        assert r["query_id"] in doc_ids
        assert r["positive_doc_ids"][0] in doc_ids

    # -----------------------------------------------------
    # 5. Save
    # -----------------------------------------------------
    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")

    print(
        f"[OK] Task 1A-R (Tier 1 — Concept Grounding) — "
        f"{len(queries)} queries, {len(docs)} documents."
    )
