import re
import json
import pandas as pd
from pathlib import Path

from ontologybench.utils.text import mask_exact_alias
from ontologybench.utils.io import save_jsonl

OUTDIR = Path("ontologybench/data/tasks/1A-R_hpo_definition_retrieval")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def dedupe_aliases_case_insensitive(lst):
    seen = set()
    out = []
    for a in lst:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def remove_parenthetical(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

def build_hpo_definition_retrieval(df: pd.DataFrame):
    """
    Task 1A-R (Tier 1 — Concept Grounding; training-only): HPO Definition Retrieval

    Query:
        HPO alias (non-empty)

    Document:
        Masked HPO definition (rendered-text deduplicated)

    Qrels:
        (HPO_ID::alias) → HPO_ID
    """

    docs = []
    queries = []
    qrels = []

    required = ["hpo_id", "hpo_alias", "hpo_definitions"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # -----------------------------------------------------
    # 1. Build raw docs / queries
    # -----------------------------------------------------
    for _, row in df[required].iterrows():
        hid = row["hpo_id"]
        raw_aliases = row["hpo_alias"]
        definition = row["hpo_definitions"]

        if not isinstance(hid, str) or not hid.strip():
            continue
        if not isinstance(raw_aliases, list):
            continue
        if not isinstance(definition, str) or not definition.strip():
            continue

        aliases = [
            a.strip() for a in raw_aliases
            if isinstance(a, str) and a.strip()
        ]
        aliases = dedupe_aliases_case_insensitive(aliases)

        if not aliases:
            continue

        # mask all aliases
        masked_def = definition
        for alias in aliases:
            masked_def, _ = mask_exact_alias(
                alias, masked_def, replacement="[MASK]"
            )

        masked_def = remove_parenthetical(masked_def)

        if not masked_def.strip():
            continue

        docs.append({
            "doc_id": hid,
            "doc": masked_def
        })

        for alias in aliases:
            qid = f"{hid}::{alias}"

            queries.append({
                "query_id": qid,
                "query": alias
            })

            qrels.append({
                "query_id": qid,
                "positive_doc_ids": [hid]
            })

    # -----------------------------------------------------
    # 2. Deduplicate documents (rendered-text level)
    # -----------------------------------------------------
    docs_df = pd.DataFrame(docs)
    if docs_df.empty:
        raise ValueError("No HPO documents generated.")

    docs_df = docs_df.sort_values("doc_id")
    docs_df = docs_df.drop_duplicates(subset=["doc"], keep="first")

    kept_doc_ids = set(docs_df["doc_id"])

    # -----------------------------------------------------
    # 3. Filter queries and qrels + enforce non-empty queries
    # -----------------------------------------------------
    queries = [
        q for q in queries
        if q["query_id"].split("::")[0] in kept_doc_ids
        and isinstance(q["query"], str)
        and q["query"].strip()
    ]

    if not queries:
        raise ValueError(
            "No non-empty queries remain after document deduplication."
        )

    qrels = [
        {
            "query_id": r["query_id"],
            "positive_doc_ids": [r["query_id"].split("::")[0]]
        }
        for r in qrels
        if r["query_id"].split("::")[0] in kept_doc_ids
    ]

    docs = docs_df.to_dict("records")

    # -----------------------------------------------------
    # 4. Final integrity check
    # -----------------------------------------------------
    query_ids = {q["query_id"] for q in queries}
    doc_ids = {d["doc_id"] for d in docs}

    for r in qrels:
        assert r["query_id"] in query_ids
        assert r["positive_doc_ids"][0] in doc_ids

    # -----------------------------------------------------
    # 5. Save
    # -----------------------------------------------------
    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")

    print(
        f"[OK] HPO Definition Retrieval — "
        f"{len(queries)} queries, {len(docs)} documents."
    )
