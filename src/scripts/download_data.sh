#!/usr/bin/env bash
set -euo pipefail

# Download raw ontology and gene metadata files.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPOSITORY_ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

echo "Downloading raw data into ${RAW_DIR} ..."

cd "${RAW_DIR}"

# -------------------------
# MONDO + MAXO
# -------------------------
echo "Downloading MONDO..."
wget -c https://purl.obolibrary.org/obo/mondo.json \
     -O mondo.json

echo "Downloading MAXO annotations..."
curl -L -o maxo-annotations.tsv \
  https://raw.githubusercontent.com/monarch-initiative/maxo-annotations/master/annotations/maxo-annotations.tsv



# -------------------------
# HPO ontology + annotations
# -------------------------
echo "Downloading HPO ontology..."
wget -c https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-10-22/hp.json \
     -O hp.json

echo "Downloading HPO annotations..."
curl -L -o phenotype_to_genes.txt \
    https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-10-22/phenotype_to_genes.txt
 

curl -L -o genes_to_phenotype.txt \
    https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-10-22/genes_to_phenotype.txt

curl -L -o genes_to_disease.txt \
    https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-10-22/genes_to_disease.txt

curl -L -o phenotype.hpoa \
    https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-10-22/phenotype.hpoa


# -------------------------
# HGNC gene metadata (CC0)
# -------------------------
echo "Downloading HGNC gene metadata..."
wget -c https://storage.googleapis.com/public-download-files/hgnc/json/json/hgnc_complete_set.json \
     -O hgnc_complete_set.json


# -------------------------
# NCBI gene summary + gene info
# -------------------------
echo "Downloading NCBI gene summary..."
wget -c https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz \
     -O gene_info.gz

gunzip -f gene_info.gz
mv gene_info ncbi_gene_summary.tsv

echo "Downloading Homo sapiens gene_info..."
wget -c https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz \
     -O Homo_sapiens.gene_info.gz

gunzip -f Homo_sapiens.gene_info.gz
mv Homo_sapiens.gene_info human_gene_info.tsv

echo "All downloads complete."
