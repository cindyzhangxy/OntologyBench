---
license: cc-by-4.0
language:
- en
task_categories:
- text-retrieval
pretty_name: OntologyBench
size_categories:
- 100K<n<1M
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: "data/train-*.jsonl.gz"
  - split: test
    path: "data/test-*.jsonl.gz"
---

# OntologyBench

OntologyBench is a tiered benchmark for structured biomedical knowledge retrieval: Tier 1 **Concept Grounding**, Tier 2 **Relational Retrieval**, and Tier 3 **Compositional Retrieval**. This dataset exposes the paper's model-independent positive training pairs and held-out test pairs as one Hugging Face `DatasetDict` with `train` and `test` splits.

## Dataset structure

Each row contains:

- `task_name`: canonical OntologyBench task name.
- `tier_id`: integer tier identifier (`1`, `2`, or `3`).
- `tier_name`: canonical human-readable tier label.
- `query_id`: source query identifier.
- `query`: biomedical query text.
- `target_id`: task-scoped target identifier used for split assignment.
- `raw_target_id`: source document identifier.
- `target`: positive target text.
- `split`: `train` or `test`.

All rows are positive query-target pairs; negatives are not materialized. Retrieval training code should construct in-batch or mined negatives as appropriate.

## Split construction

Task-scoped target identifiers are sorted, shuffled with seed 42, and assigned 80% to train and 20% to test. All positive pairs for a target stay in one split. Duplicate normalized query-target text pairs are removed within each split. The split is target-disjoint within each task but transductive across tasks because the same biomedical entity may occur in different task namespaces.

The generated release includes `summary.json`, deterministic target assignments in `assignments.jsonl.gz`, and `SHA256SUMS`. The paper-count validation must pass before release generation completes.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("cxyzhang/OntologyBench")
print(dataset)
print(dataset["train"].features)
```

## Source and limitations

The benchmark was derived from public biomedical resources including HPO, Mondo, HGNC, and NCBI Gene. The task artifacts are intended for biomedical retrieval research, not clinical diagnosis or patient-care decisions. Ontology-aware reference systems use a different information regime from text-only retrievers and should not be treated as directly comparable learned systems.

See the accompanying paper and software repository for full construction details, metrics, provenance limitations, and validation warnings.

## License

The released task data are available under CC BY 4.0. Upstream biomedical resources retain their own attribution and licensing requirements.

## Citation

```bibtex
@article{zhang2026ontologybench,
  title   = {OntologyBench: Can Dense Retrieval Satisfy Structured Biomedical Constraints?},
  author  = {Zhang, Xiao Yu Cindy and Wasserman, Wyeth W. and Zhu, Jian},
  year    = {2026}
}
```
