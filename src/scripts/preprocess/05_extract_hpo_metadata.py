#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_extract_hpo_metadata.py

Purpose:
    Attach HPO ontology metadata (preferred label, aliases, definitions)
    to the HPO-disease mapping produced earlier.

Inputs:
    - data/intermediate/hpo_metadata.pkl        (from 02)
    - data/intermediate/mondo_annotations.pkl   (from 04)

Output:
    - data/intermediate/hpo_annotations.pkl
"""

import os
import pandas as pd

INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

HPO_META_PATH = os.path.join(INTER_DIR, "hpo_metadata.pkl")
MONDO_ANN_PATH = os.path.join(INTER_DIR, "mondo_annotations.pkl")

# ==========================================================
#             BUILD HPO LOOKUP TABLES
# ==========================================================

def build_hpo_lookup(hpo_df: pd.DataFrame):
    """
    Construct lookup dictionaries:
        HPO_ID → aliases
        HPO_ID → definition
        HPO_ID → preferred label
    """
    alias_map = {}
    def_map = {}
    label_map = {}

    for _, row in hpo_df.iterrows():
        hid = row["HPO_ID"]

        # aliases is a plain list already extracted in Step 02
        aliases = row["aliases"] if isinstance(row["aliases"], list) else []

        alias_map[hid] = aliases
        def_map[hid] = row.get("definition", None)
        label_map[hid] = row.get("label", "")

    return alias_map, def_map, label_map


# ==========================================================
#                MAIN ATTACH FUNCTION
# ==========================================================

def attach_hpo_metadata(df, alias_map, def_map, label_map):
    """
    Add:
        - hpo_alias
        - hpo_definitions
        - hpo_label (canonical)
    """

    df = df.copy()

    df["hpo_alias"] = df["hpo_id"].map(alias_map)
    df["hpo_definitions"] = df["hpo_id"].map(def_map)
    df["hpo_label"] = df["hpo_id"].map(label_map)

    return df


# ==========================================================
#                     MAIN ENTRYPOINT
# ==========================================================

def main():

    print("Loading HPO metadata...")
    hpo_meta = pd.read_pickle(HPO_META_PATH)

    print("Loading MONDO-annotated disease/phenotype table...")
    df_mondo = pd.read_pickle(MONDO_ANN_PATH)

    print("Building HPO lookup tables...")
    alias_map, def_map, label_map = build_hpo_lookup(hpo_meta)

    print("Attaching HPO ontology metadata to annotation rows...")
    df_out = attach_hpo_metadata(df_mondo, alias_map, def_map, label_map)

    out_path = os.path.join(INTER_DIR, "hpo_annotations.pkl")
    df_out.to_pickle(out_path)

    print(f"Saved enriched HPO annotations: {df_out.shape} to {out_path}")
    print("Completed: 05_extract_hpo_metadata.py")


if __name__ == "__main__":
    main()
