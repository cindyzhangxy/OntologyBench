# ontologybench/tasks/d2gene_twohop.py

import pandas as pd
from pathlib import Path
from ontologybench.utils.io import save_jsonl

OUTDIR = Path("ontologybench/data/tasks/2B-R_disease_to_gene")


def build_disease_to_gene_twohop(df: pd.DataFrame):
    """
    Task 2B-R (Tier 2 — Relational Retrieval; evaluation-only):
    Disease → Gene Retrieval (implicit two-hop)

    Query:
        MONDO disease name + disease definition

    Documents:
        Gene summaries (one per Entrez ID)

    Qrels:
        MONDO_ID → list of GENE:<entrez_id>
    """

    required = [
        "MONDO_ID",
        "disease_name",
        "disorder_definition",
        "entrez_id",
        "symbol",
        "Summary",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # -----------------------------------------------------
    # 1. Filter required fields
    # -----------------------------------------------------
    subset = (
        df[required]
        .dropna(subset=["MONDO_ID", "entrez_id"])
        .copy()
    )

    # -----------------------------------------------------
    # 2. Build gene documents (dedup by Entrez ID)
    # -----------------------------------------------------
    gene_df = (
        subset[["entrez_id", "symbol", "Summary"]]
        .dropna(subset=["entrez_id"])
        .sort_values("entrez_id")
        .drop_duplicates(subset=["entrez_id"], keep="first")
    )

    docs = [
        {
            "doc_id": f"GENE:{row['entrez_id']}",
            "doc": f"{row['symbol']}. {row['Summary']}".strip(),
        }
        for _, row in gene_df.iterrows()
        if isinstance(row["symbol"], str) and row["symbol"].strip()
        and isinstance(row["Summary"], str) and row["Summary"].strip()
    ]

    if not docs:
        raise ValueError("No gene documents generated.")

    doc_ids = {d["doc_id"] for d in docs}

    # -----------------------------------------------------
    # 3. Build disease queries (one per MONDO_ID)
    # -----------------------------------------------------
    queries = []
    mondo_groups = subset.groupby("MONDO_ID")

    for mondo_id, g in mondo_groups:
        name = g["disease_name"].iloc[0]
        definition = g["disorder_definition"].iloc[0]

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(definition, str) or not definition.strip():
            continue

        query_text = f"{name.strip()}. {definition.strip()}"

        if not query_text.strip():
            continue

        queries.append({
            "query_id": mondo_id,
            "query": query_text
        })

    if not queries:
        raise ValueError("No non-empty disease queries generated.")

    query_ids = {q["query_id"] for q in queries}

    # -----------------------------------------------------
    # 4. Build qrels (filter to existing docs)
    # -----------------------------------------------------
    qrels = []

    for mondo_id, g in mondo_groups:
        if mondo_id not in query_ids:
            continue

        genes = sorted({
            f"GENE:{gid}"
            for gid in g["entrez_id"].tolist()
            if f"GENE:{gid}" in doc_ids
        })

        if not genes:
            continue

        qrels.append({
            "query_id": mondo_id,
            "positive_doc_ids": genes
        })

    if not qrels:
        raise ValueError("No valid qrels generated.")

    # -----------------------------------------------------
    # 5. Final integrity checks
    # -----------------------------------------------------
    for r in qrels:
        assert r["query_id"] in query_ids
        for d in r["positive_doc_ids"]:
            assert d in doc_ids

    # -----------------------------------------------------
    # 6. Save outputs
    # -----------------------------------------------------
    OUTDIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")

    print(
        f"[OK] Task 2B-R (Tier 2 — Relational Retrieval) built: "
        f"{len(queries)} queries, {len(docs)} gene documents."
    )
