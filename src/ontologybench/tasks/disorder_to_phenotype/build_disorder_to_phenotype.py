# ontologybench/tasks/disorder_to_phenotype/build_disorder_to_phenotype.py

import pandas as pd
from pathlib import Path
from ontologybench.utils.io import save_jsonl

OUTDIR = Path("ontologybench/data/tasks/1B-R_disorder_to_phenotype")


def build_disorder_to_phenotype(df: pd.DataFrame):
    """
    Task 1B-R (Tier 2 — Relational Retrieval): Disease → Phenotype Retrieval

    Query:
        MONDO disease name + disease definition

    Documents:
        HPO definitions (deduplicated at rendered-text level)

    Qrels:
        MONDO_ID → list of HPO_IDs
    """

    # ---------------------------------------------------------
    # Required fields and basic filtering
    # ---------------------------------------------------------
    required = [
        "MONDO_ID", "disease_name", "disorder_definition",
        "hpo_id", "hpo_definitions"
    ]

    df = df.dropna(subset=["MONDO_ID", "hpo_id"])
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    subset = df[required].copy()

    # ---------------------------------------------------------
    # Build phenotype documents (rendered-text dedup)
    # ---------------------------------------------------------
    docs_df = (
        subset[["hpo_id", "hpo_definitions"]]
        .dropna(subset=["hpo_definitions"])
        .sort_values("hpo_id")            # deterministic
        .drop_duplicates(subset=["hpo_definitions"], keep="first")
    )

    docs = [
        {
            "doc_id": r["hpo_id"],
            "doc": r["hpo_definitions"].strip(),
        }
        for _, r in docs_df.iterrows()
        if isinstance(r["hpo_definitions"], str) and r["hpo_definitions"].strip()
    ]

    kept_hpo_ids = {d["doc_id"] for d in docs}

    if not docs:
        raise ValueError("No phenotype documents generated.")

    # ---------------------------------------------------------
    # Build disease queries (one per MONDO_ID)
    # ---------------------------------------------------------
    queries = []
    mondo_to_query = {}

    for mondo_id, g in subset.groupby("MONDO_ID"):
        name = g["disease_name"].iloc[0]
        definition = g["disorder_definition"].iloc[0]

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(definition, str) or not definition.strip():
            continue

        query_text = f"{name.strip()}. {definition.strip()}"

        queries.append({
            "query_id": mondo_id,
            "query": query_text
        })
        mondo_to_query[mondo_id] = True

    if not queries:
        raise ValueError("No non-empty disease queries generated.")

    # ---------------------------------------------------------
    # Build qrels (filtering to kept documents)
    # ---------------------------------------------------------
    qrels = []

    for mondo_id, g in subset.groupby("MONDO_ID"):
        if mondo_id not in mondo_to_query:
            continue

        hpos = [
            h for h in g["hpo_id"].unique().tolist()
            if h in kept_hpo_ids
        ]

        if not hpos:
            continue

        qrels.append({
            "query_id": mondo_id,
            "positive_doc_ids": hpos
        })

    if not qrels:
        raise ValueError("No valid qrels generated.")

    # ---------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------
    query_ids = {q["query_id"] for q in queries}
    doc_ids = {d["doc_id"] for d in docs}

    for r in qrels:
        assert r["query_id"] in query_ids
        for d in r["positive_doc_ids"]:
            assert d in doc_ids

    # ---------------------------------------------------------
    # 6. Save outputs
    # ---------------------------------------------------------
    OUTDIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")

    print(
        f"[OK] Task 1B-R (Tier 2 — Relational Retrieval) built: "
        f"{len(queries)} queries, {len(docs)} docs."
    )
