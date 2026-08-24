# Provenance and reproducibility

## Release target

Version 1.0.0 is an artifact-centered release of the exact eight-task paper split. Its reproducibility target is the checksummed JSONL under `src/ontologybench/data/tasks_global/`, not a byte-identical rebuild from mutable upstream services.

Run:

```bash
ontologybench-verify
ontologybench-validate
```

The split contains 8 tasks, 8,699 documents, and 36,285 query/qrel pairs. Validation intentionally reports two documented warning classes: duplicated query surface forms within tasks and a cross-evaluation query overlap. They are warnings, not checksum failures.

## Upstream evidence

- HPO ontology and annotations: release `v2025-10-22`.
- Mondo: the archived construction workspace records version `2025-11-04`.
- Gene metadata: HGNC complete-set JSON and NCBI Gene human metadata.

The archival preprocessing scripts are retained under `src/scripts/` for transparency. Upstream endpoints can change, so reconstruction may differ from the released split. Gene Ontology was not used by the released tasks; its unused download/patch path has therefore been removed.

## Paper results

`results/paper/` contains compact aggregate tables, not the roughly gigabyte-scale rankings and score traces from the research workspace. Each aggregate row identifies its source artifact, SHA-256 digest, recorded campaign commit when available, and provenance status. The main campaign was recorded from commit `b8f38238276b49564f5e312c2ffe8415b0751378` with a dirty worktree; this limitation is preserved explicitly.

Paper-reported values and values recoverable from saved artifacts are stored in separate columns whenever they differ. See `results/paper/README.md` before using the reranking table.

## Generative scoring

`src/ontologybench/generative/scoring.py` is the canonical Appendix F prompt implementation for this release. Its semantic template SHA-256 is `11b242f710f212f0e37e21db54c704c32f7153f01c0991f3a939151438fce08a`. The historical OpenAI score artifact reproduces the corrected paper aggregate, but the exact historical prompt serialization could not be verified; the manifest labels that uncertainty.

## Hugging Face training and test split

`src/scripts/create_splits.py` joins the full archival task artifacts into positive query-target pairs. Task-scoped target identifiers are deterministically assigned 80% to train and 20% to test with seed 42; normalized duplicate pairs are removed within each split. The resulting release contains 471,854 training rows and 125,744 test rows and reproduces every per-task paper count and average-positive statistic.

The generated `hf_dataset/` folder is a separate upload payload and is not committed to the software repository. It carries its own dataset card, deterministic assignment file, summary, and SHA-256 manifest.
