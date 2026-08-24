#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_map_cross_ontology_ids.py

Goal:
    Connect HPO annotations (phenotype.hpoa) with MONDO IDs via OMIM/ORPHA
    using MONDO's cross-references.

Inputs:
    - data/intermediate/mondo_metadata.pkl
    - data/raw/phenotype.hpoa
    - data/raw/phenotype_to_genes.txt
    - data/raw/maxo-annotations.tsv

Outputs:
    data/intermediate/hpo_with_mondo.pkl
"""

import os
import json
import pandas as pd


RAW_DIR = "../../data/raw"
INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

# -----------------------------
# Paths
# -----------------------------
MONDO_META = os.path.join(INTER_DIR, "mondo_metadata.pkl")
PHENO_HPOA = os.path.join(RAW_DIR, "phenotype.hpoa")
PHENO_TO_GENES = os.path.join(RAW_DIR, "phenotype_to_genes.txt")
MAXO_PATH = os.path.join(RAW_DIR, "maxo-annotations.tsv")


# ==========================================================
#       LOAD INPUT TABLES
# ==========================================================

def load_inputs():
    print("Loading MONDO metadata...")
    mondo_df = pd.read_pickle(MONDO_META)

    print("Loading HPO disease annotations (phenotype.hpoa)...")
    df_hpoa = pd.read_csv(
        PHENO_HPOA, sep="\t", comment="#", dtype=str
    )[['database_id', 'disease_name', 'hpo_id']]

    print("Loading phenotype_to_genes...")
    phen2gene = pd.read_csv(PHENO_TO_GENES, sep="\t", dtype=str)

    print("Loading MAXO annotations...")
    tx_df = pd.read_csv(MAXO_PATH, sep="\t", dtype=str)[
        ['disease_id', 'disease_name', 'maxo_name', 'hpo_id']
    ]

    return mondo_df, df_hpoa, phen2gene, tx_df


# ==========================================================
#       BUILD CROSS-ONTOLOGY MAPS
# ==========================================================

def build_mondo_lookup(mondo_df: pd.DataFrame):
    """
    Construct dicts that map:
        OMIM → MONDO
        ORPHA → MONDO
    """

    mondo_flat = {}

    for _, row in mondo_df.iterrows():
        mondo_id = row["MONDO_ID"]

        # OMIM IDs
        for om in row["omim_ids"]:
            # normalize patterns (e.g., OMIMPS → OMIM)
            norm = om.replace("OMIMPS:", "OMIM:").replace("OMIMPS", "OMIM")
            if norm.startswith("OMIM"):
                mondo_flat[norm] = mondo_id

        # ORPHA IDs
        for orp in row["orpha_ids"]:
            norm = orp.replace("Orphanet:", "ORPHA:").replace("ORPHA::", "ORPHA:")
            if norm.startswith("ORPHA"):
                mondo_flat[norm] = mondo_id

    return mondo_flat


def build_hpo_name_and_gene_maps(phen2gene_df: pd.DataFrame):
    """
    Build:
        hpo_id → hpo_name
        hpo_id → entrez gene id
    """
    hpo_map = dict(zip(phen2gene_df['hpo_id'], phen2gene_df['hpo_name']))
    gene_map = dict(zip(phen2gene_df['hpo_id'], phen2gene_df['ncbi_gene_id']))

    return hpo_map, gene_map


def build_maxo_map(tx_df: pd.DataFrame):
    """
    Build:
        (disease_id, hpo_id) → maxo_name
    """
    return {
        (row['disease_id'], row['hpo_id']): row['maxo_name']
        for _, row in tx_df.iterrows()
    }


# ==========================================================
#               MAIN PROCESSING LOGIC
# ==========================================================

def map_all(df_hpoa, mondo_flat, hpo_map, gene_map, maxo_map):
    """
    Add:
        - HPO name
        - Entrez ID
        - MONDO ID mapped from OMIM/ORPHA
        - Treatment annotation (MAXO)
    """

    # ---------------------------
    # Attach HPO name
    # ---------------------------
    df_hpoa["hpo_name"] = df_hpoa["hpo_id"].map(hpo_map)

    # ---------------------------
    # Attach gene ID
    # ---------------------------
    df_hpoa["entrez_id"] = df_hpoa["hpo_id"].map(gene_map)

    # Ensure gene IDs are consistent strings
    df_hpoa["entrez_id"] = df_hpoa["entrez_id"].astype(str)

    # ---------------------------
    # Map database_id (OMIM/ORPHA) → MONDO
    # ---------------------------
    df_hpoa["MONDO_ID"] = df_hpoa["database_id"].map(mondo_flat)

    # ---------------------------
    # MAXO treatment, if present
    # ---------------------------
    df_hpoa["treatment"] = df_hpoa.apply(
        lambda row: maxo_map.get((row["MONDO_ID"], row["hpo_id"])),
        axis=1
    )

    return df_hpoa


# ==========================================================
#                           MAIN
# ==========================================================

def main():
    mondo_df, df_hpoa, phen2gene_df, tx_df = load_inputs()

    print("Building MONDO lookup...")
    mondo_flat = build_mondo_lookup(mondo_df)

    print("Building HPO name/gene lookup...")
    hpo_map, gene_map = build_hpo_name_and_gene_maps(phen2gene_df)

    print("Building MAXO treatment map...")
    maxo_map = build_maxo_map(tx_df)

    print("Mapping all cross-ontology annotations...")
    df_out = map_all(df_hpoa, mondo_flat, hpo_map, gene_map, maxo_map)

    out_path = os.path.join(INTER_DIR, "hpo_with_mondo.pkl")
    df_out.to_pickle(out_path)
    print(f"Saved cross-linked dataset: {df_out.shape} to {out_path}")

    print("Completed: 03_map_cross_ontology_ids.py")


if __name__ == "__main__":
    main()
