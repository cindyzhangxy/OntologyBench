# ontologybench/tasks/phe2d.py


import pandas as pd
from pathlib import Path
from ontologybench.utils.io import save_jsonl
from ontologybench.utils.split import make_splits

OUTDIR = Path('ontologybench/data/tasks/1C-R_phenotype_to_disorder')


def build_phenotype_to_disorder(df: pd.DataFrame):
    """
    Task 1C-R (Tier 2 — Relational Retrieval): Phenotype → Disease Retrieval
    Each HPO term -> list of MONDO diseases associated with it.
    Query: HPO name + definition
    Docs: MONDO disease definition
    Positive_doc_ids: list of MONDO_IDs for that HPO
    """   


    # ---------------------------------------------------------
    # 1. Basic filtering
    # ---------------------------------------------------------
    subset = df[[
        "MONDO_ID", "disease_name", "disorder_definition",
        "hpo_id", "hpo_name", "hpo_definitions"
    ]].dropna(subset=["MONDO_ID", "hpo_id"])

    # ---------------------------------------------------------
    # 2. Build disease documents
    # ---------------------------------------------------------
    # One document per MONDO_ID
    docs = (
        subset
        .drop_duplicates(subset=["MONDO_ID"])
        .apply(
            lambda r: {
                "doc_id": r["MONDO_ID"],
                "doc": f"{r['disease_name']}. {r['disorder_definition']}",
            },
            axis=1
        )
        .tolist()
    )

    # ---------------------------------------------------------
    # 3. Build queries (one per HPO term)
    # ---------------------------------------------------------
    queries = (
        subset.groupby('hpo_id')
        .apply(lambda g: {
            "query_id": g["hpo_id"].iloc[0],
            "query": f"{g['hpo_name'].iloc[0]}. {g['hpo_definitions'].iloc[0]}",
            "positive_doc_ids": list(g["MONDO_ID"].unique())
        })
    )

    queries = list(queries)

    # ---------------------------------------------------------
    # 4. Build qrels (relevance mapping)
    # ---------------------------------------------------------
    # Each HPO → list of MONDO diseases    

    qrels = (
        subset.groupby('hpo_id')['MONDO_ID']
        .unique()
        .apply(list)
        .reset_index()
        .rename(columns={'hpo_id': 'query_id',"MONDO_ID": "positive_doc_ids"})
        .to_dict(orient="records")
    )
 
    # ---------------------------------------------------------
    # 6. Save datasets
    # ---------------------------------------------------------
    OUTDIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")


    print(f"[OK] Task 1C-R (Tier 2 — Relational Retrieval) built: "
          f"{len(queries)} queries, {len(docs)} docs.")
