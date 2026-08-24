

"""
run_all.py

Master runner for the archival OntologyBench preprocessing pipeline.

This script executes:
    download_data.sh (optional)
    01_load_raw_files.py
    02_parse_ontologies.py
    03_map_cross_ontology_ids.py
    04_extract_mondo_metadata.py
    05_extract_hpo_metadata.py
    06_merge_gene_metadata.py
    07_build_master_table.py
    08_filter_and_clean.py
    09_export_master_jsonl.py

It ensures:
    - All raw files exist before running
    - Output directories are created
    - Each step executes in correct dependency order
"""

import os
import sys
import subprocess
from pathlib import Path
import datetime


# ==========================================================
# CONFIG
# ==========================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
INTER_DIR = REPOSITORY_ROOT / "data" / "intermediate"
OUT_DIR = REPOSITORY_ROOT / "data" / "output"

SCRIPTS = [
    "01_load_raw_files.py",
    "02_parse_ontologies.py",
    "03_map_cross_ontology_ids.py",
    "04_extract_mondo_metadata.py",
    "05_extract_hpo_metadata.py",
    "06_merge_gene_metadata.py",
    "07_build_master_table.py",
    "08_filter_and_clean.py",
    "09_export_master_jsonl.py",
]


REQUIRED_FILES = [
    "mondo.json",
    "maxo-annotations.tsv",
    "hp.json",
    "phenotype_to_genes.txt",
    "genes_to_phenotype.txt",
    "genes_to_disease.txt",
    "phenotype.hpoa",
    "hgnc_complete_set.json",
    "ncbi_gene_summary.tsv",
]


# ==========================================================
# UTILITIES
# ==========================================================

def check_raw_files():
    print("Checking raw files...")

    missing = []
    for fname in REQUIRED_FILES:
        if not (RAW_DIR / fname).exists():
            missing.append(fname)

    if missing:
        print("\nERROR: Missing required raw files:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun:    bash download_data.sh\n")
        sys.exit(1)

    print("✓ All raw files found.\n")


def run_script(script_name, step_idx, total_steps, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{step_idx:02d}_{script_name}_{timestamp}.log")

    print(f"\n[{step_idx}/{total_steps}] Starting: {script_name}")
    print(f"    → Log: {log_file}")
    print("--------------------------------------------------")

    script_path = Path(__file__).parent / script_name

    with open(log_file, "w", encoding="utf-8") as lf:

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent   # run in this folder
        )

        for line in process.stdout:
            print(line, end="")
            lf.write(line)

        process.wait()

    if process.returncode != 0:
        print(f"❌ FAILED: {script_name} (see log: {log_file})")
        sys.exit(process.returncode)

    print(f"✓ Finished: {script_name}")


# ==========================================================
# MAIN LOGIC
# ==========================================================

def main():

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTER_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    check_raw_files()

    total = len(SCRIPTS)
    for i, script in enumerate(SCRIPTS, start=1):
        run_script(script, i, total)

    print("\n==============================================")
    print("        Pipeline completed successfully!       ")
    print("==============================================")
    print(f"Final outputs located in: {OUT_DIR}")


if __name__ == "__main__":
    main()
