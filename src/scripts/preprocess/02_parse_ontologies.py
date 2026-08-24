#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_parse_ontologies.py

Extracts clean ontology-level metadata from MONDO and HPO.
Outputs:
    data/intermediate/hpo_metadata.pkl
    data/intermediate/mondo_metadata.pkl
"""

import os
import json
import pandas as pd
from collections import defaultdict


# ==========================================================
#               PATH CONFIGURATION
# ==========================================================

RAW_DIR = "../../data/raw"
OUT_DIR = "../../data/intermediate"

os.makedirs(OUT_DIR, exist_ok=True)

MONDO_PATH = os.path.join(RAW_DIR, "mondo.json")
HPO_PATH = os.path.join(RAW_DIR, "hp.json")


# ==========================================================
#                 HELPER: Clean MONDO ID
# ==========================================================

def normalize_mondo_id(node_id: str) -> str:
    """
    Convert IDs like:
        http://purl.obolibrary.org/obo/MONDO_0005148
    →  MONDO:0005148
    """
    if not isinstance(node_id, str):
        return None

    if "MONDO_" in node_id:
        tail = node_id.split("/")[-1]       # MONDO_0005148
        return tail.replace("_", ":")       # MONDO:0005148

    if node_id.startswith("MONDO:"):
        return node_id

    return None


# ==========================================================
#                 PARSE MONDO ONTOLOGY
# ==========================================================

def parse_mondo(mondo_json: dict) -> pd.DataFrame:
    """
    Extract:
        - MONDO_ID
        - canonical disease name
        - synonyms list
        - definition
        - OMIM IDs
        - ORPHA IDs
    """

    records = []

    nodes = mondo_json.get("graphs", [])[0].get("nodes", [])

    for node in nodes:
        node_id_raw = node.get("id", "")
        mondo_id = normalize_mondo_id(node_id_raw)
        if mondo_id is None:
            continue

        lbl = node.get("lbl", "")
        meta = node.get("meta", {})

        # ---- Extract synonyms ----
        syns = []

        # 1. .synonyms[]
        for s in meta.get("synonyms", []):
            if isinstance(s, dict) and "val" in s:
                syns.append(s["val"])

        # 2. basicPropertyValues[]
        for s in meta.get("basicPropertyValues", []):
            pred = s.get("pred", "")
            if pred.endswith((
                "hasExactSynonym",
                "hasRelatedSynonym",
                "hasBroadSynonym",
                "hasNarrowSynonym"
            )):
                syns.append(s.get("val", ""))

        # ---- Extract definition ----
        definition = None
        if "definition" in meta and isinstance(meta["definition"], dict):
            definition = meta["definition"].get("val")

        # ---- Extract OMIM / ORPHA xrefs ----
        omim_ids = []
        orpha_ids = []

        for x in meta.get("xrefs", []):
            val = x.get("val", "")
            if val.startswith("OMIM"):
                omim_ids.append(val)
            elif val.startswith("Orphanet") or val.startswith("ORPHA"):
                # unify notation
                if "Orphanet:" in val:
                    val = val.replace("Orphanet:", "ORPHA:")
                orpha_ids.append(val)

        records.append({
            "MONDO_ID": mondo_id,
            "canonical_name": lbl,
            "synonyms": list(dict.fromkeys([lbl] + syns)),  # canonical first
            "definition": definition,
            "omim_ids": omim_ids,
            "orpha_ids": orpha_ids
        })

    return pd.DataFrame(records)


# ==========================================================
#                 PARSE HPO ONTOLOGY
# ==========================================================

def normalize_hpo_id(node_id: str) -> str:
    """
    Convert IDs like:
        HP_0000118 → HP:0000118
    """
    if not isinstance(node_id, str):
        return None

    if "HP_" in node_id:
        tail = node_id.split("/")[-1]
        return tail.replace("_", ":")

    if node_id.startswith("HP:"):
        return node_id

    return None


def parse_hpo(hp_json: dict) -> pd.DataFrame:
    """
    Extract:
        - HPO_ID
        - preferred label
        - synonyms
        - definition
    """

    records = []

    nodes = hp_json.get("graphs", [])[0].get("nodes", [])

    for node in nodes:
        node_id = normalize_hpo_id(node.get("id", ""))
        if node_id is None:
            continue

        label = node.get("lbl", "")
        meta = node.get("meta", {})

        # ---- definition ----
        definition = None
        if "definition" in meta and isinstance(meta["definition"], dict):
            definition = meta["definition"].get("val")

        # ---- synonyms ----
        syns = []
        for s in meta.get("synonyms", []):
            if isinstance(s, dict) and "val" in s:
                syns.append(s["val"])

        # canonical label always included last
        alias_list = syns + [label]

        records.append({
            "HPO_ID": node_id,
            "label": label,
            "aliases": alias_list,
            "definition": definition
        })

    return pd.DataFrame(records)


# ==========================================================
#                       MAIN
# ==========================================================

def main():

    # ------------------------------
    # MONDO
    # ------------------------------
    print("Loading MONDO ontology...")
    with open(MONDO_PATH, "r", encoding="utf-8") as f:
        mondo_json = json.load(f)

    mondo_df = parse_mondo(mondo_json)
    mondo_df.to_pickle(os.path.join(OUT_DIR, "mondo_metadata.pkl"))
    print(f"Saved MONDO metadata: {mondo_df.shape}")

    # ------------------------------
    # HPO
    # ------------------------------
    print("Loading HPO ontology...")
    with open(HPO_PATH, "r", encoding="utf-8") as f:
        hpo_json = json.load(f)

    hpo_df = parse_hpo(hpo_json)
    hpo_df.to_pickle(os.path.join(OUT_DIR, "hpo_metadata.pkl"))
    print(f"Saved HPO metadata: {hpo_df.shape}")

    print("Completed: 02_parse_ontologies.py")


if __name__ == "__main__":
    main()
