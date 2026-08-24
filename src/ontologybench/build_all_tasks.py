# ontologybench/build_all_tasks.py

import pandas as pd
from pathlib import Path
import json

# from ontologybench.tasks.phenotype_to_disorder.build_phenotype_to_disorder import build_phenotype_to_disorder

from ontologybench.tasks.disorder_to_phenotype.build_disorder_to_phenotype import build_disorder_to_phenotype
from ontologybench.tasks.gene_document.build_gene_document import build_gene_document_retrieval
from ontologybench.tasks.hpo_definition.build_hpo_definition import build_hpo_definition_retrieval
from ontologybench.tasks.mondo_definition.build_mondo_definition import build_mondo_definition_retrieval
from ontologybench.tasks.phenotype_to_disorder.phe2d import build_phenotype_to_disorder as build_phe2d
from ontologybench.tasks.phenotype_to_gene.phe2gene import build_phenotype_to_gene_twohop as build_phe2g
from ontologybench.tasks.disorder_to_gene.d2gene import build_disease_to_gene_twohop as build_d2g
from ontologybench.tasks.multiphenotype_to_disorder.multi_phen_to_disorder import build_tier3_multi_phenotype as multiphe

DATA_ROOT = Path("./data/output")   # folder where master_df.parquet lives
# MASTER = DATA_ROOT / "master_df.parquet"
MASTER = DATA_ROOT / "master_df.jsonl"


def main():
    print("🔧 Loading master dataframe...")
    # df = pd.read_parquet(MASTER)
    rows = []
    with open(MASTER, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    print(f"Loaded master_df: {df.shape}")

    # 1. Gene document retrieval
    print("\n[1/8] gene document...")
    build_gene_document_retrieval(df)

    # 2. HPO definition retrieval
    print("\n[2/8] HPO definition...")
    build_hpo_definition_retrieval(df)

    # 3. MONDO definition retrieval
    print("\n[3/8] MONDO definition...")
    build_mondo_definition_retrieval(df)
   
    # 4. disorder → phenotype
    print("\n[4/8] Building disorder_to_phenotype...")
    build_disorder_to_phenotype(df)

    # 5. phenotype → disorder
    print("\n[5/8] Building phenotype_to_disorder...")
    build_phe2d(df)

    # 6. Phenotype → Gene (two-hop)
    print("\n[6/8] Building phenotype_to_gene (two-hop)...")
    build_phe2g(df)

    # 7. Disease → Gene (two-hop)
    print("\n[7/8] Building disease_to_gene (two-hop)...")
    build_d2g(df)

    # 8. HPO: phenotype to disease to gene to disease (multihop)
    print("\n[8/8] Building multi_phenotype_to_disorder reasoning")
    multiphe(df)

   

    
    print("\n=======================================")
    print("All OntologyBench tasks generated!")
    print("=======================================\n")


if __name__ == "__main__":
    main()


# Run from the repository root with: python -m ontologybench.build_all_tasks
