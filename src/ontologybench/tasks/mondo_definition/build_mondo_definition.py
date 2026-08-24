import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ontologybench.utils.text import mask_exact_alias
from ontologybench.utils.io import save_jsonl

OUTDIR = Path("ontologybench/data/tasks/1A-R_mondo_definition_retrieval")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def dedupe_case_insensitive(xs):
    seen = {}
    for x in xs:
        key = x.lower()
        if key not in seen:
            seen[key] = x  # keep first-seen surface form
    return sorted(seen.values())


def normalize_alias_list(x):
    if isinstance(x, list):
        items = x
    elif isinstance(x, np.ndarray):
        items = x.tolist()
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("{") and s.endswith("}"):
            return [p.strip() for p in s.strip("{}").split(",") if p.strip()]
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if str(a).strip()]
        except Exception:
            pass
        return [s]
    else:
        return []

    return [a.strip() for a in items if isinstance(a, str) and a.strip()]


def remove_parenthetical(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

def build_mondo_definition_retrieval(df: pd.DataFrame):
    """
    Task 1A-R (Tier 1 — Concept Grounding; training-only): MONDO Definition Retrieval

    Query:
        MONDO alias (non-empty)

    Document:
        Masked MONDO definition (rendered-text deduplicated)

    Qrels:
        (MONDO_ID::alias) → MONDO_ID
    """

    required = ["MONDO_ID", "disorder_alias", "disorder_definition"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    docs = []
    queries = []
    qrels = []

    # -----------------------------------------------------
    # Aggregate aliases per MONDO_ID
    # -----------------------------------------------------
    mondo_df = (
        df[required]
        .dropna(subset=["MONDO_ID"])
        .groupby("MONDO_ID")
        .agg({
            "disorder_alias": lambda xs: sum(
                (normalize_alias_list(a) for a in xs), []
            ),
            "disorder_definition": "first",
        })
        .reset_index()
    )

    mondo_df["disorder_alias"] = mondo_df["disorder_alias"].apply(
        lambda xs: dedupe_case_insensitive(xs) if isinstance(xs, list) else []
    )

    # -----------------------------------------------------
    # Build raw docs / queries / qrels
    # -----------------------------------------------------
    for _, row in mondo_df.iterrows():
        mid = row["MONDO_ID"]
        aliases = row["disorder_alias"]
        definition = row["disorder_definition"]

        if not isinstance(mid, str) or not mid.strip():
            continue
        if not isinstance(definition, str) or not definition.strip():
            continue
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
            "doc_id": mid,
            "doc": masked_def
        })

        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                continue

            qid = f"{mid}::{alias}"

            queries.append({
                "query_id": qid,
                "query": alias
            })

            qrels.append({
                "query_id": qid,
                "positive_doc_ids": [mid]
            })

    if not docs:
        raise ValueError("No MONDO documents generated.")

    # -----------------------------------------------------
    # Deduplicate documents (rendered-text level)
    # -----------------------------------------------------
    docs_df = pd.DataFrame(docs)
    docs_df = docs_df.sort_values("doc_id")
    docs_df = docs_df.drop_duplicates(subset=["doc"], keep="first")

    kept_doc_ids = set(docs_df["doc_id"])

    # -----------------------------------------------------
    # 4. Filter queries / qrels + enforce non-empty queries
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
    # Integrity checks
    # -----------------------------------------------------
    query_ids = {q["query_id"] for q in queries}
    doc_ids = {d["doc_id"] for d in docs}

    for r in qrels:
        assert r["query_id"] in query_ids
        assert r["positive_doc_ids"][0] in doc_ids

    # -----------------------------------------------------
    # 6. Save outputs
    # -----------------------------------------------------
    save_jsonl(docs, OUTDIR / "docs.jsonl")
    save_jsonl(queries, OUTDIR / "queries.jsonl")
    save_jsonl(qrels, OUTDIR / "qrels.jsonl")

    print(
        f"[OK] MONDO Definition Retrieval — "
        f"{len(queries)} queries, {len(docs)} documents."
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    DATA_ROOT = Path("./data/output")
    jsonl_path = DATA_ROOT / "master_df.jsonl"

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    build_mondo_definition_retrieval(df)
