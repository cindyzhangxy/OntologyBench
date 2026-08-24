#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
09_export_master_jsonl.py

Purpose:
    Export the final cleaned ontology–phenotype–gene dataset into:
        - JSONL  (master_df.jsonl)   <-- authoritative, numpy-clean
        - Dataset statistics (stats.json)

Inputs:
    - data/intermediate/master_cleaned.pkl

Outputs:
    - data/output/master_df.jsonl
    - data/output/stats.json
"""

import os
import json
import pandas as pd
import numpy as np

INTER_DIR = "../../data/intermediate"
OUT_DIR = "../../data/output"

os.makedirs(OUT_DIR, exist_ok=True)

MASTER_CLEAN_PATH = os.path.join(INTER_DIR, "master_cleaned.pkl")


# ==========================================================
#        SAFE JSON CONVERSION
# ==========================================================

def ensure_json_safe(obj):
    """
    Recursively convert Python/numpy/pandas objects into JSON-safe types.
    """

    # None, bool, int, float, str
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array → list
    if isinstance(obj, np.ndarray):
        return [ensure_json_safe(v) for v in obj.tolist()]

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [ensure_json_safe(v) for v in obj]

    # dictionary
    if isinstance(obj, dict):
        return {k: ensure_json_safe(v) for k, v in obj.items()}

    # pandas NA
    if pd.isna(obj):
        return None

    # fallback
    return str(obj)


# ==========================================================
#        STATISTICS
# ==========================================================

def compute_stats(df: pd.DataFrame) -> dict:
    stats = {
        "num_rows": len(df),
        "num_diseases": df["MONDO_ID"].nunique() if "MONDO_ID" in df else 0,
        "num_phenotypes": df["hpo_id"].nunique() if "hpo_id" in df else 0,
        "num_genes": df["entrez_id"].nunique() if "entrez_id" in df else 0
    }
    return stats


# ==========================================================
#        MAIN LOGIC
# ==========================================================

def main():

    print("Loading cleaned master dataset...")
    df_clean = pd.read_pickle(MASTER_CLEAN_PATH)
    print(f"Loaded dataframe: {df_clean.shape}")

    # ------------------------------------------------------
    # Debug: check for numpy arrays in list-valued columns
    # ------------------------------------------------------
    list_cols = ["hpo_alias", "disorder_alias", "gene_alias", "gene_group"]
    alert_count = 0

    for col in list_cols:
        if col not in df_clean.columns:
            continue
        for idx, val in df_clean[col].items():
            if isinstance(val, np.ndarray):
                alert_count += 1

    print(f"[DEBUG] NumPy arrays detected in list columns: {alert_count}")

    # ------------------------------------------------------
    # Export JSONL (authoritative format)
    # ------------------------------------------------------
    jsonl_path = os.path.join(OUT_DIR, "master_df.jsonl")
    print(f"Writing JSONL to {jsonl_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_clean.iterrows():
            row_dict = row.to_dict()
            safe_dict = {k: ensure_json_safe(v) for k, v in row_dict.items()}
            f.write(json.dumps(safe_dict, ensure_ascii=False) + "\n")

    print(f"Saved JSONL: {jsonl_path}")

    # ------------------------------------------------------
    # Export statistics
    # ------------------------------------------------------
    stats = compute_stats(df_clean)
    stats_path = os.path.join(OUT_DIR, "stats.json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  Saved stats: {stats_path}")
    print(f" Stats content: {stats}")

    print("Completed: 09_export_master_jsonl.py")


if __name__ == "__main__":
    main()
