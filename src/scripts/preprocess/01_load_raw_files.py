#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_load_raw_files.py

Purpose:
    Load all raw ontology/annotation files and store them in a single
    intermediate pickle for downstream preprocessing.

Inputs (all under data/raw/):
    - genes_to_disease.txt
    - genes_to_phenotype.txt
    - phenotype_to_genes.txt
    - maxo-annotations.tsv
    - phenotype.hpoa
    - mondo.json
    - hp.json
    - hgnc_complete_set.json
    - ncbi_gene_summary.tsv
    - human_gene_info.tsv

Output:
    data/intermediate/raw_loaded.pkl
"""

import os
import json
import pandas as pd

RAW_DIR = "../../data/raw"
INTER_DIR = "../../data/intermediate"
os.makedirs(INTER_DIR, exist_ok=True)

def check_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return path

def main():

    print("Loading raw files...")

    # -------------------------------
    # Tabular annotation files
    # -------------------------------
    gene2dz_df = pd.read_csv(
        check_exists(f"{RAW_DIR}/genes_to_disease.txt"),
        sep="\t",
        dtype=str
    )

    gene2pheno_df = pd.read_csv(
        check_exists(f"{RAW_DIR}/genes_to_phenotype.txt"),
        sep="\t",
        dtype=str
    )

    phen2gene_df = pd.read_csv(
        check_exists(f"{RAW_DIR}/phenotype_to_genes.txt"),
        sep="\t",
        dtype=str
    )

    tx_df = pd.read_csv(
        check_exists(f"{RAW_DIR}/maxo-annotations.tsv"),
        sep="\t",
        dtype=str
    )

    df_hpoa = pd.read_csv(
        check_exists(f"{RAW_DIR}/phenotype.hpoa"),
        sep="\t",
        comment="#",
        dtype=str
    )

    # -------------------------------
    # JSON ontologies
    # -------------------------------

    with open(check_exists(f"{RAW_DIR}/mondo.json"),  "r", encoding="utf-8",) as f:
        mondo = json.load(f)

    with open(check_exists(f"{RAW_DIR}/hp.json"), "r", encoding="utf-8") as f:
        hp = json.load(f)

    with open(check_exists(f"{RAW_DIR}/hgnc_complete_set.json"), "r", encoding="utf-8") as f:
        hgnc = json.load(f)

    # -------------------------------
    # NCBI gene summaries
    # -------------------------------
    ncbi_gene_summary = pd.read_csv(
        check_exists(f"{RAW_DIR}/ncbi_gene_summary.tsv"),
        sep="\t",
        dtype=str
    )

    # -------------------------------
    # Save aggregated raw data
    # -------------------------------
    raw_data = {
        "gene2dz": gene2dz_df,
        "gene2pheno": gene2pheno_df,
        "phen2gene": phen2gene_df,
        "tx": tx_df,
        "hpoa": df_hpoa,
        "mondo": mondo,
        "hp": hp,
        "hgnc": hgnc,
        "ncbi_gene_summary": ncbi_gene_summary
    }

    out_path = f"{INTER_DIR}/raw_loaded.pkl"
    pd.to_pickle(raw_data, out_path)

    print(f"Loaded raw data saved to: {out_path}")
    print("Completed: 01_load_raw_files.py")


if __name__ == "__main__":
    main()
