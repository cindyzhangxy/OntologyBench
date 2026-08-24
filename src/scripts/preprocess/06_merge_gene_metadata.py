#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_merge_gene_metadata.py

Purpose:
    Attach human gene metadata (HGNC + NCBI) to the ontology-enriched
    disease–phenotype table produced in step 05.

Inputs:
    - data/intermediate/hpo_annotations.pkl
    - data/intermediate/raw_loaded.pkl (contains HGNC + NCBI + phenotype_to_genes)

Output:
    - data/intermediate/gene_annotations.pkl
"""

import os
import pandas as pd

INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

RAW_PATH = os.path.join(INTER_DIR, "raw_loaded.pkl")
HPO_ANN_PATH = os.path.join(INTER_DIR, "hpo_annotations.pkl")


# ==========================================================
#             HELPER: BUILD HGNC LOOKUP TABLE
# ==========================================================

def build_hgnc_lookup(hgnc_json):
    """
    Build dictionary:
        entrez_id → {
            symbol,
            gene_name,
            alias,
            gene_group,
            location,
            locus_group
        }
    """
    docs = hgnc_json["response"]["docs"]

    lookup = {}
    for d in docs:
        entrez_id = d.get("entrez_id")
        if not entrez_id:
            continue

        lookup[str(entrez_id)] = {
            "symbol": d.get("symbol"),
            "gene_name": d.get("name"),
            "alias": (d.get("alias_symbol", []) + d.get("prev_symbol", [])),
            "gene_group": d.get("gene_group", []),
            "location": d.get("location"),
            "locus_group": d.get("locus_group"),
        }
    return lookup


# ==========================================================
#     HELPER: MERGE NCBI GENE SUMMARY WITH GENE TABLE
# ==========================================================

def attach_gene_summary(df: pd.DataFrame, ncbi_df: pd.DataFrame):
    """
    Adds NCBI Summary based on GeneID (entrez).
    """
    ncbi_df = ncbi_df.rename(columns={"GeneID": "entrez_id"})

    return df.merge(
        ncbi_df[["entrez_id", "Summary"]],
        on="entrez_id",
        how="left"
    )


# ==========================================================
#        MAIN MERGING LOGIC
# ==========================================================

def merge_gene_metadata(df_hpo, phen2gene, hgnc_lookup, ncbi_df):
    """
    Attach gene metadata to phenotype rows.
    """

    df = df_hpo.copy()

    # ------------------------------------------------------
    # 1. Attach HPO → gene mapping (entrez already added in step 03)
    # ------------------------------------------------------
    # Ensure all entrez_id values are strings
    df["entrez_id"] = df["entrez_id"].astype(str)

    # ------------------------------------------------------
    # 2. Attach HGNC fields:
    #    symbol, gene_name, alias, gene_group, location, locus_group
    # ------------------------------------------------------
    def extract(field, default=None):
        return df["entrez_id"].apply(
            lambda gid: hgnc_lookup.get(gid, {}).get(field, default)
        )

    df["symbol"] = extract("symbol")
    df["gene_name"] = extract("gene_name")
    df["gene_alias"] = extract("alias")
    df["gene_group"] = extract("gene_group")
    df["location"] = extract("location")
    df["locus_group"] = extract("locus_group")

    # ------------------------------------------------------
    # 3. Add NCBI summary text
    # ------------------------------------------------------
    df = attach_gene_summary(df, ncbi_df)

    return df


# ==========================================================
#                            MAIN
# ==========================================================

def main():

    print("Loading raw data...")
    raw = pd.read_pickle(RAW_PATH)

    phen2gene_df = raw["phen2gene"]
    hgnc_json = raw["hgnc"]
    ncbi_gene_summary = raw["ncbi_gene_summary"]

    print("Loading phenotype+MONDO+HPO data...")
    df_hpo = pd.read_pickle(HPO_ANN_PATH)

    print("Building HGNC lookup table...")
    hgnc_lookup = build_hgnc_lookup(hgnc_json)

    print("Merging gene metadata with ontology annotations...")
    df_out = merge_gene_metadata(df_hpo, phen2gene_df, hgnc_lookup, ncbi_gene_summary)

    out_path = os.path.join(INTER_DIR, "gene_annotations.pkl")
    df_out.to_pickle(out_path)

    print(f"Saved gene-enriched annotation table: {df_out.shape} to {out_path}")
    print("Completed: 06_merge_gene_metadata.py")


if __name__ == "__main__":
    main()
