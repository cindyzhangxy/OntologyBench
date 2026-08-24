# ontologybench/tasks/phe2gene.py

import pandas as pd
from pathlib import Path
from ontologybench.utils.io import save_jsonl

OUTDIR = Path("ontologybench/data/tasks/2A-R_phenotype_to_gene")


def build_phenotype_to_gene_twohop(df: pd.DataFrame):
    """
    Task 2A-R (Tier 2 — Relational Retrieval; evaluation-only):
    Phenotype → Gene Retrieval (implicit two-hop)

    Query:
        HPO name + HPO definition (one per HPO_ID)

    Documents:
        Gene summaries (one per Entrez ID)

    Qrels:
        HPO_ID → list of GENE:<entrez_id>
    """

    required = [
        "hpo_id",
        "hpo_name",
        "hpo_definitions",
        "entrez_id",
        "symbol",
        "Summary",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # -----------------------------------------------------
    # Base filtering
    # -----------------------------------------------------
    subset = (
        df[required]
        .dropna(subset=["hpo_id", "entrez_id"])
        .copy()
    )

    # -----------------------------------------------------
    # Build gene documents (dedup by Entrez ID)
    # -----------------------------------------------------
    gene_df = (
        subset[["entrez_id", "symbol", "Summary"]]
        .dropna(subset=["symbol", "Summary"])
        .sort_values("entrez_id")
        .drop_duplicates(subset=["entrez_id"], keep="first")
    )

    docs = [
        {
            "doc_id": f"GENE:{r['entrez_id']}",
            "doc": f"{r['symbol'].strip()}. {r['Summary'].strip()}",
        }
        for _, r in gene_df.iterrows()
        if r["symbol"].strip() and r["Summary"].strip()
    ]

    if not docs:
        raise ValueError("No valid gene documents generated.")

    doc_ids = {d["doc_id"] for d in docs}

    # -----------------------------------------------------
    # Build queries (one per HPO_ID)
    # -----------------------------------------------------
    queries = []
    hpo_groups = subset.groupby("hpo_id")

    for hpo_id, g in hpo_groups:
        name = g["hpo_name"].iloc[0]
        definition = g["hpo_definitions"].iloc[0]

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(definition, str) or not definition.strip():
            continue

        query_text = f"{name.strip()}. {definition.strip()}"

        if not query_text.strip():
            continue

        queries.append({
            "query_id": hpo_id,
            "query": query_text,
        })

    if not queries:
        raise ValueError("No non-empty phenotype queries generated.")

    query_ids = {q["query_id"] for q in queries}

    # -----------------------------------------------------
    # Build qrels (filtered to existing docs)
    # -----------------------------------------------------
    qrels = []

    for hpo_id, g in hpo_groups:
        if hpo_id not in query_ids:
            continue

        positives = sorted({
            f"GENE:{gid}"
            for gid in g["entrez_id"].tolist()
            if f"GENE:{gid}" in doc_ids
        })

        if not positives:
            continue

        qrels.append({
            "query_id": hpo_id,
            "positive_doc_ids": positives,
        })

    if not qrels:
        raise ValueError("No valid qrels generated.")

    # -----------------------------------------------------
    # Integrity checks
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
        f"[OK] Task 2A-R (Tier 2 — Relational Retrieval) built: "
        f"{len(queries)} queries, {len(docs)} gene documents."
    )
