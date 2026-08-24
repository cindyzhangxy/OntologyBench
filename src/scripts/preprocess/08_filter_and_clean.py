#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_filter_and_clean.py (NUMPY-PROOF, OPTION A)

Purpose:
    Clean unified master table and **guarantee all alias fields remain
    pure Python lists**, never numpy arrays.

Key Fixes:
    ✓ Normalize list fields BEFORE and AFTER every pandas operation
    ✓ Prevent numpy ndarray from surviving dedupe or boolean filtering
    ✓ Enforce Option A: drop rows with empty alias lists
"""

import os
import re
import numpy as np
import pandas as pd

INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

MASTER_PATH = os.path.join(INTER_DIR, "master_table.pkl")


# ==========================================================
# Helpers — Clean summary text
# ==========================================================
def clean_summary(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[provided.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================================
# Helpers — Normalize aliases into pure Python lists
# ==========================================================
def normalize_list(x):
    """Convert ndarray → list, tuple → list, scalar → [scalar], None → []"""
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        return [x]
    return [x]


def normalize_columns(df, cols):
    """Apply list normalization to specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_list)
    return df


def count_numpy(df, cols):
    """Count how many cells contain NumPy arrays."""
    count = 0
    for col in cols:
        if col not in df.columns:
            continue
        count += sum(isinstance(v, np.ndarray) for v in df[col])
    return count


# ==========================================================
# Main cleaning logic
# ==========================================================
def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    list_cols = ["hpo_alias", "disorder_alias", "gene_alias", "gene_group"]

    print("[DEBUG] BEFORE NORMALIZATION — ndarray count:",
          count_numpy(df, list_cols))

    # ------------------------------------------------------
    # STEP 0 — Normalize BEFORE filtering
    # ------------------------------------------------------
    df = normalize_columns(df, list_cols)

    print("[DEBUG] AFTER INITIAL NORMALIZATION — ndarray count:",
          count_numpy(df, list_cols))

    # ------------------------------------------------------
    # STEP 1 — protein-coding only
    # ------------------------------------------------------
    if "locus_group" in df.columns:
        df = df[df["locus_group"] == "protein-coding gene"]
        df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 2 — drop missing essential IDs
    # ------------------------------------------------------
    essential_cols = ["MONDO_ID", "hpo_id", "entrez_id", "symbol"]
    for col in essential_cols:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
            df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 3 — Option A: drop empty alias rows
    # ------------------------------------------------------
    if "hpo_alias" in df.columns:
        df = df[df["hpo_alias"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 4 — drop missing definitions
    # ------------------------------------------------------
    if "hpo_definitions" in df.columns:
        df = df[df["hpo_definitions"].notna() & (df["hpo_definitions"].str.strip() != "")]
        df = normalize_columns(df, list_cols)

    if "disorder_definition" in df.columns:
        df = df[df["disorder_definition"].notna() &
                (df["disorder_definition"].str.strip() != "")]
        df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 5 — Clean gene summary
    # ------------------------------------------------------
    if "Summary" in df.columns:
        df["Summary"] = df["Summary"].fillna("").astype(str).apply(clean_summary)
        df = df[df["Summary"].str.strip() != ""]
        df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 6 — Ensure MONDO exists
    # ------------------------------------------------------
    if "MONDO_ID" in df.columns:
        df = df[df["MONDO_ID"].notna() & (df["MONDO_ID"].str.strip() != "")]
        df = normalize_columns(df, list_cols)

    # ------------------------------------------------------
    # STEP 7 — SAFE DEDUPLICATION
    # ------------------------------------------------------
    key_cols = [c for c in ["MONDO_ID", "hpo_id", "entrez_id", "symbol"]
                if c in df.columns]

    if key_cols:
        idx = df.drop_duplicates(subset=key_cols).index
        df = df.loc[idx].reset_index(drop=True)
        df = normalize_columns(df, list_cols)

    print("[DEBUG] AFTER ALL CLEANING — ndarray count:",
          count_numpy(df, list_cols))

    return df


# ==========================================================
# Entry point
# ==========================================================
def main():

    print("Loading master table...")
    df = pd.read_pickle(MASTER_PATH)
    print(f"Input shape: {df.shape}")

    print("Filtering & cleaning (NUMPY-PROOF)...")
    df_clean = filter_and_clean(df)
    print(f"Output shape: {df_clean.shape}")

    out_path = os.path.join(INTER_DIR, "master_cleaned.pkl")
    df_clean.to_pickle(out_path)

    print(f"Saved cleaned master table to {out_path}")


if __name__ == "__main__":
    main()
