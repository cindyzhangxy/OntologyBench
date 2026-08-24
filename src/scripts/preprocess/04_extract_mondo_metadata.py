#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_extract_mondo_metadata.py

Purpose:
    Attach MONDO ontology metadata (canonical name, aliases, definitions)
    to the cross-linked HPO annotation table produced in step 03.

Inputs:
    - data/intermediate/mondo_metadata.pkl        (from 02)
    - data/intermediate/hpo_with_mondo.pkl        (from 03)

Output:
    - data/intermediate/mondo_annotations.pkl
"""

import os
import pandas as pd

INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

MONDO_META_PATH = os.path.join(INTER_DIR, "mondo_metadata.pkl")
HPO_WITH_MONDO_PATH = os.path.join(INTER_DIR, "hpo_with_mondo.pkl")


# ==========================================================
#              BUILD LOOKUP TABLES
# ==========================================================

def build_mondo_lookup(df_mondo):
    """
    Build two lookup dictionaries:
        1. MONDO_ID → aliases (canonical-first)
        2. MONDO_ID → definition
    """
    alias_map = {}
    def_map = {}

    for _, row in df_mondo.iterrows():
        mondo_id = row["MONDO_ID"]
        canonical = row["canonical_name"]
        synonyms = row["synonyms"] or []

        # Canonical-first alias list, deduplicated
        alias_list = []
        if isinstance(canonical, str) and canonical.strip():
            alias_list.append(canonical)

        for s in synonyms:
            if isinstance(s, str) and s not in alias_list:
                alias_list.append(s)

        alias_map[mondo_id] = alias_list
        def_map[mondo_id] = row.get("definition", None)

    return alias_map, def_map


# ==========================================================
#                MAIN PROCESSING FUNCTION
# ==========================================================

def attach_mondo_metadata(df_hpo, alias_map, def_map):
    """
    For each row in df_hpo (HPO-level annotation), attach:

        - disorder_alias (list of strings)
        - disorder_definition (string or None)
    """

    df = df_hpo.copy()

    # Normalize MONDO ID style in case ':' → '_' was used earlier
    df["MONDO_KEY"] = df["MONDO_ID"].str.replace(":", "_", regex=False)

    df["disorder_alias"] = df["MONDO_ID"].apply(lambda x: alias_map.get(x, []))
    df["disorder_definition"] = df["MONDO_ID"].apply(lambda x: def_map.get(x, None))

    # Drop temporary key
    df = df.drop(columns=["MONDO_KEY"], errors="ignore")

    return df


# ==========================================================
#                               MAIN
# ==========================================================

def main():

    print("Loading MONDO metadata...")
    df_mondo = pd.read_pickle(MONDO_META_PATH)

    print("Loading cross-linked HPO annotations...")
    df_hpo = pd.read_pickle(HPO_WITH_MONDO_PATH)

    print("Building MONDO alias + definition lookup tables...")
    alias_map, def_map = build_mondo_lookup(df_mondo)

    print("Attaching MONDO metadata to HPO annotation rows...")
    df_out = attach_mondo_metadata(df_hpo, alias_map, def_map)

    out_path = os.path.join(INTER_DIR, "mondo_annotations.pkl")
    df_out.to_pickle(out_path)

    print(f"Saved enriched MONDO annotations: {df_out.shape} to {out_path}")
    print("Completed: 04_extract_mondo_metadata.py")


if __name__ == "__main__":
    main()
