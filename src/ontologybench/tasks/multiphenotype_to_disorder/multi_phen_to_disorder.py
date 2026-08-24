# ontologybench/tasks/tier3_multi_phen_to_disorder.py

import pandas as pd
import numpy as np
from pathlib import Path

from ontologybench.utils.io import save_jsonl
from ontologybench.utils.text import join_with_and

OUTDIR = Path("ontologybench/data/tasks/3A_multi_phenotype_to_disorder")


# ---------------------------------------------------------
# Helper: sample one subset of exactly k phenotypes
# ---------------------------------------------------------
def sample_fixed_k(phenotypes, k):
    ph = list(set(phenotypes))
    if len(ph) < k:
        return None
    np.random.shuffle(ph)
    return ph[:k]


# ---------------------------------------------------------
# Tier 3 — Compositional Retrieval Builder (Evaluation)
# ---------------------------------------------------------
def build_tier3_multi_phenotype(df: pd.DataFrame, k=3, min_match=3):
    """
    Task 3A (Tier 3 — Compositional Retrieval; evaluation-only):

    Query:
        An unordered set of k canonical HPO phenotype names
        (definitions explicitly excluded by design)

    Documents:
        Disorder name + disorder definition (one per MONDO_ID)

    Qrels:
        Query → list of MONDO_IDs with ≥ min_match overlapping phenotypes
    """

    required = [
        "MONDO_ID",
        "disease_name",
        "disorder_definition",
        "hpo_id",
        "hpo_name",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # -----------------------------------------------------
    # 1. Clean base dataframe
    # -----------------------------------------------------
    subset = (
        df[required]
        .dropna(subset=["MONDO_ID", "hpo_id", "hpo_name"])
        .copy()
    )

    # -----------------------------------------------------
    # 2. Build disease documents (name + definition)
    # -----------------------------------------------------
    disease_df = (
        subset[["MONDO_ID", "disease_name", "disorder_definition"]]
        .dropna(subset=["disease_name", "disorder_definition"])
        .sort_values("MONDO_ID")
        .drop_duplicates(subset=["MONDO_ID"], keep="first")
    )

    docs = []
    for _, r in disease_df.iterrows():
        name = r["disease_name"].strip()
        definition = r["disorder_definition"].strip()

        if not name or not definition:
            continue

        docs.append({
            "doc_id": r["MONDO_ID"],
            "doc": f"{name}. {definition}",
        })

    if not docs:
        raise ValueError("No valid disease documents generated.")

    doc_ids = {d["doc_id"] for d in docs}

    # -----------------------------------------------------
    # 3. Build phenotype ↔ disease indices
    # -----------------------------------------------------
    mondo_to_hpos = (
        subset.groupby("MONDO_ID")["hpo_id"]
        .unique()
        .apply(list)
        .to_dict()
    )

    hpo_to_mondos = (
        subset.groupby("hpo_id")["MONDO_ID"]
        .unique()
        .apply(set)
        .to_dict()
    )

    hpo_info = subset.groupby("hpo_id").first()

    # -----------------------------------------------------
    # 4. Build queries + qrels (deduplicated)
    # -----------------------------------------------------
    queries = []
    qrels = []
    seen_subsets = set()
    qid_counter = 0

    for hpo_list in mondo_to_hpos.values():

        sub = sample_fixed_k(hpo_list, k=k)
        if sub is None:
            continue

        subset_key = tuple(sorted(sub))
        if subset_key in seen_subsets:
            continue
        seen_subsets.add(subset_key)

        # compute disease overlap
        overlap_counts = {}
        for h in sub:
            for d in hpo_to_mondos.get(h, []):
                overlap_counts[d] = overlap_counts.get(d, 0) + 1

        positives = [
            d for d, c in overlap_counts.items()
            if c >= min_match and d in doc_ids
        ]
        if not positives:
            continue

        # build query text: phenotype NAMES ONLY
        names = []
        for h in sub:
            name = hpo_info.loc[h]["hpo_name"]
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

        if not names:
            continue

        query_text = join_with_and(names)

        qid = f"Q{qid_counter}"
        qid_counter += 1

        queries.append({
            "query_id": qid,
            "query": query_text,
            "subset_hpos": sub,
        })

        qrels.append({
            "query_id": qid,
            "positive_doc_ids": positives,
        })

    if not queries:
        raise ValueError("No Tier-3 queries generated.")

    # -----------------------------------------------------
    # 5. Integrity checks
    # -----------------------------------------------------
    query_ids = {q["query_id"] for q in queries}
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
        f"[OK] Task 3A (Tier 3 — Compositional Retrieval) built: {len(queries)} queries, "
        f"{len(docs)} disease docs (k={k}, min_match={min_match})"
    )
