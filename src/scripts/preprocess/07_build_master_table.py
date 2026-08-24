#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_build_master_table.py

Purpose:
    Build the unified ontology–phenotype–gene master table that combines:
        - MONDO disease metadata
        - HPO phenotype metadata
        - Gene annotations (HGNC + NCBI summaries + mapping from phenotype_to_genes)
        - MAXO treatment metadata (already attached)

This script does *not* perform filtering or cleaning.
That is done in Step 08.

Inputs:
    - data/intermediate/gene_annotations.pkl     (output of Step 06)

Output:
    - data/intermediate/master_table.pkl
"""

import os
import pandas as pd

INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

GENE_ANN_PATH = os.path.join(INTER_DIR, "gene_annotations.pkl")


# ==========================================================
#                SELECT FINAL COLUMN SCHEMA
# ==========================================================

def select_final_columns(df):
    """
    Keep only the columns needed for benchmark construction.
    
    This structure matches the dataset you manually constructed
    during prototyping and ensures downstream reproducibility.
    """

    cols = [
        # MONDO-level disease metadata
        "MONDO_ID",
        "disease_name",
        "disorder_alias",
        "disorder_definition",

        # HPO-level phenotype metadata
        "hpo_id",             # Note: in previous steps the column might be 'hpo_id'
        "hpo_name",
        "hpo_label",
        "hpo_alias",
        "hpo_definitions",

        # Gene-level metadata
        "entrez_id",
        "symbol",
        "gene_name",
        "gene_alias",
        "gene_group",
        "location",
        "locus_group",
        "Summary",            # NCBI gene summary (unprocessed)

        # Treatment metadata (from MAXO)
        "treatment"
    ]

    # Some intermediate files use 'hpo_id' or 'HPO_ID'. Normalize:
    df = df.rename(columns={"HPO_ID": "hpo_id"})

    # Keep only existing columns
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()


# ==========================================================
#                           MAIN
# ==========================================================

def main():

    print("Loading gene-enriched annotation table...")
    df_gene = pd.read_pickle(GENE_ANN_PATH)

    print("Constructing unified master table...")
    df_master = select_final_columns(df_gene)

    out_path = os.path.join(INTER_DIR, "master_table.pkl")
    df_master.to_pickle(out_path)

    print(f"Saved master table: {df_master.shape} to {out_path}")
    print("Completed: 07_build_master_table.py")


if __name__ == "__main__":
    main()
