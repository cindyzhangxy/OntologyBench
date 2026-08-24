# Paper result artifacts

These are compact, auditable aggregates for the paper; they are not fresh benchmark runs.

- `retrieval_results.csv` contains nDCG, MRR, and Hit/Recall at 1, 5, and 10 for the 12 text retrievers in Table 2 plus the ontology-aware reference systems from Table 3.
- `reranking_results.csv` separates the values displayed in Table 4 (`paper_*`) from metrics recoverable from the located saved artifact (`saved_*`).
- `artifact_manifest.json` records relative archival-workspace identifiers, file sizes, SHA-256 digests, and provenance status. It does not expose local absolute paths.

## Important limitations

The recorded main campaign commit is `b8f38238276b49564f5e312c2ffe8415b0751378`, but the source worktree was dirty. The Qwen3-Embed first-stage row and Qwen3-4B-Instruct row do not have exact matching saved aggregates. The ModernColBERT† saved nDCG rounds to 0.184 rather than the reported 0.183. The Qwen3-4B-Thinking artifact contained incomplete/invalid records. The corrected GPT aggregate matches the displayed row, but its historical prompt serialization is unresolved.

The Appendix F prompt in `src/ontologybench/generative/scoring.py` is canonical for future scoring; it must not be retroactively claimed as the verified prompt for the historical GPT score file.

Ontology-aware reference systems have explicit access to ontology structure and are diagnostic comparators, not like-for-like text-only retrievers.
