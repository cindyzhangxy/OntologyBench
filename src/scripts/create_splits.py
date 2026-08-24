"""Create the canonical model-independent OntologyBench v1 split.

The algorithm reproduces the paper/HPC partition: task-scoped target IDs are
sorted, shuffled with seed 42, and assigned 80% to train and 20% to test.
Because the namespace includes the task name, ontology entities may recur
across tasks; this is a transductive, task-scoped target-disjoint split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = REPOSITORY_ROOT / "src" / "ontologybench" / "data" / "tasks"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "hf_dataset"
DATASET_CARD = REPOSITORY_ROOT / "huggingface" / "README.md"

TRAIN_RATIO = 0.8
SEED = 42
SPLIT_TYPE = "task_scoped_target_disjoint"
MAX_SHARD_BYTES = 100 * 1024 * 1024

TASK_TIER_METADATA: dict[str, tuple[int, str]] = {
    "1A-R_gene_document_retrieval": (1, "Concept Grounding"),
    "1A-R_hpo_definition_retrieval": (1, "Concept Grounding"),
    "1A-R_mondo_definition_retrieval": (1, "Concept Grounding"),
    "1B-R_disorder_to_phenotype": (2, "Relational Retrieval"),
    "1C-R_phenotype_to_disorder": (2, "Relational Retrieval"),
    "2A-R_phenotype_to_gene": (2, "Relational Retrieval"),
    "2B-R_disease_to_gene": (2, "Relational Retrieval"),
    "3A-multi_phenotype_to_disorder": (3, "Compositional Retrieval"),
}

ALL_TASKS: tuple[str, ...] = tuple(TASK_TIER_METADATA)

SOURCE_TASK_DIRS = {
    "3A-multi_phenotype_to_disorder": "3A_multi_phenotype_to_disorder",
}

PAPER_EXPECTED_STATS: dict[str, dict[str, dict[str, float | int]]] = {
    "1A-R_gene_document_retrieval": {
        "train": {"queries": 1914, "avg_positives": 1.00},
        "test": {"queries": 431, "avg_positives": 1.00},
    },
    "1A-R_hpo_definition_retrieval": {
        "train": {"queries": 17763, "avg_positives": 1.00},
        "test": {"queries": 4581, "avg_positives": 1.00},
    },
    "1A-R_mondo_definition_retrieval": {
        "train": {"queries": 39644, "avg_positives": 1.00},
        "test": {"queries": 9205, "avg_positives": 1.00},
    },
    "1B-R_disorder_to_phenotype": {
        "train": {"queries": 6647, "avg_positives": 19.81},
        "test": {"queries": 6291, "avg_positives": 5.63},
    },
    "1C-R_phenotype_to_disorder": {
        "train": {"queries": 7988, "avg_positives": 16.62},
        "test": {"queries": 5229, "avg_positives": 6.57},
    },
    "2A-R_phenotype_to_gene": {
        "train": {"queries": 6960, "avg_positives": 1.00},
        "test": {"queries": 1507, "avg_positives": 1.00},
    },
    "2B-R_disease_to_gene": {
        "train": {"queries": 6623, "avg_positives": 17.58},
        "test": {"queries": 6504, "avg_positives": 5.31},
    },
    "3A-multi_phenotype_to_disorder": {
        "train": {"queries": 5642, "avg_positives": 4.39},
        "test": {"queries": 2537, "avg_positives": 2.26},
    },
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_jsonl_row(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


def write_gzip_jsonl(rows: Iterable[dict], path: Path) -> None:
    """Write deterministic gzip-compressed JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
            for row in rows:
                handle.write(encode_jsonl_row(row))


def write_split_shards(
    rows: Sequence[dict],
    output_dir: Path,
    split: str,
    max_uncompressed_bytes: int = MAX_SHARD_BYTES,
) -> list[Path]:
    """Write stable size-bounded JSONL gzip shards for one dataset split."""

    if max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be positive")

    boundaries: list[tuple[int, int]] = []
    start = 0
    current_bytes = 0
    for index, row in enumerate(rows):
        row_bytes = len(encode_jsonl_row(row))
        if index > start and current_bytes + row_bytes > max_uncompressed_bytes:
            boundaries.append((start, index))
            start = index
            current_bytes = 0
        current_bytes += row_bytes
    boundaries.append((start, len(rows)))

    shard_paths: list[Path] = []
    total = len(boundaries)
    for shard_index, (start, end) in enumerate(boundaries):
        path = output_dir / "data" / f"{split}-{shard_index:05d}-of-{total:05d}.jsonl.gz"
        write_gzip_jsonl(rows[start:end], path)
        shard_paths.append(path)
    return shard_paths


def write_checksum_manifest(output_dir: Path, artifacts: Iterable[Path]) -> None:
    lines: list[str] = []
    for path in sorted(artifacts, key=lambda item: item.relative_to(output_dir).as_posix()):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def build_records_from_task(task_dir: Path, task_name: str) -> list[dict]:
    """Join one retrieval task into generic positive relevance records."""

    try:
        tier_id, tier_name = TASK_TIER_METADATA[task_name]
    except KeyError as error:
        raise ValueError(f"Unknown canonical OntologyBench task: {task_name}") from error

    required = tuple(task_dir / name for name in ("docs.jsonl", "queries.jsonl", "qrels.jsonl"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task {task_name} is missing required files: {missing}")

    documents = {row["doc_id"]: row["doc"] for row in load_jsonl(task_dir / "docs.jsonl")}
    queries = {row["query_id"]: row["query"] for row in load_jsonl(task_dir / "queries.jsonl")}
    qrels: dict[str, list[str]] = defaultdict(list)
    for row in load_jsonl(task_dir / "qrels.jsonl"):
        qrels[row["query_id"]].extend(row["positive_doc_ids"])

    records: list[dict] = []
    for query_id, raw_target_ids in qrels.items():
        if query_id not in queries:
            raise ValueError(f"Task {task_name} qrels reference unknown query_id {query_id!r}")
        query = queries[query_id]
        for raw_target_id in raw_target_ids:
            if raw_target_id not in documents:
                raise ValueError(f"Task {task_name} qrels reference unknown doc_id {raw_target_id!r}")
            target = documents[raw_target_id]
            if normalize_text(query) == normalize_text(target):
                continue
            records.append(
                {
                    "task_name": task_name,
                    "tier_id": tier_id,
                    "tier_name": tier_name,
                    "query_id": query_id,
                    "query": query,
                    "target_id": f"{task_name}::{raw_target_id}",
                    "raw_target_id": raw_target_id,
                    "target": target,
                }
            )
    return records


def assign_task_scoped_targets(
    records: Sequence[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[list[dict], dict[str, str]]:
    """Return stable assignment rows and a target-to-split lookup."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")

    target_to_raw = {record["target_id"]: record["raw_target_id"] for record in records}
    shuffled_targets = sorted(target_to_raw)
    if not shuffled_targets:
        raise ValueError("Cannot split an empty collection of target identifiers")
    random.Random(seed).shuffle(shuffled_targets)

    split_index = int(train_ratio * len(shuffled_targets))
    train_targets = set(shuffled_targets[:split_index])
    assignment = {
        target_id: "train" if target_id in train_targets else "test"
        for target_id in target_to_raw
    }
    assignment_rows = [
        {
            "task_name": target_id.split("::", 1)[0],
            "target_id": target_id,
            "raw_target_id": target_to_raw[target_id],
            "split": assignment[target_id],
        }
        for target_id in sorted(assignment)
    ]
    return assignment_rows, assignment


def partition_records(
    records: Sequence[dict], assignment: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    test: list[dict] = []
    for record in records:
        output = dict(record)
        output["split"] = assignment[record["target_id"]]
        (train if output["split"] == "train" else test).append(output)
    return train, test


def deduplicate(records: Iterable[dict]) -> list[dict]:
    """Keep the first normalized query-target text pair, matching the HPC pipeline."""

    seen: set[tuple[str, str]] = set()
    clean: list[dict] = []
    for record in records:
        key = (normalize_text(record["query"]), normalize_text(record["target"]))
        if key in seen:
            continue
        seen.add(key)
        clean.append(record)
    return clean


def compute_task_stats(
    train_records: Sequence[dict],
    test_records: Sequence[dict],
    tasks: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for task_name in tasks:
        task_stats: dict[str, dict[str, float | int]] = {}
        for split_name, records in (("train", train_records), ("test", test_records)):
            relevant = [record for record in records if record["task_name"] == task_name]
            query_ids = {record["query_id"] for record in relevant}
            average = len(relevant) / len(query_ids) if query_ids else 0.0
            task_stats[split_name] = {
                "queries": len(query_ids),
                "avg_positives": round(average, 2),
            }
        stats[task_name] = task_stats
    return stats


def verify_paper_counts(observed: dict[str, dict[str, dict[str, float | int]]]) -> None:
    mismatches: list[str] = []
    for task_name, expected_task in PAPER_EXPECTED_STATS.items():
        if task_name not in observed:
            mismatches.append(f"{task_name}: missing")
            continue
        for split_name in ("train", "test"):
            expected = expected_task[split_name]
            actual = observed[task_name][split_name]
            if actual != expected:
                mismatches.append(
                    f"{task_name} {split_name}: expected {expected}, observed {actual}"
                )
    if mismatches:
        raise ValueError(
            "Split does not reproduce the paper statistics:\n  - " + "\n  - ".join(mismatches)
        )


def create_splits(
    tasks_root: Path,
    output_dir: Path,
    tasks: Sequence[str] = ALL_TASKS,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    verify_expected_counts: bool = False,
) -> dict:
    """Create assignment, train/test, and summary artifacts."""

    if not tasks:
        raise ValueError("At least one task must be selected")

    records: list[dict] = []
    for task_name in tasks:
        source_dir = SOURCE_TASK_DIRS.get(task_name, task_name)
        records.extend(build_records_from_task(tasks_root / source_dir, task_name))

    assignment_rows, assignment = assign_task_scoped_targets(records, train_ratio, seed)
    train_records, test_records = partition_records(records, assignment)
    train_records = deduplicate(train_records)
    test_records = deduplicate(test_records)
    task_stats = compute_task_stats(train_records, test_records, tasks)

    if verify_expected_counts:
        if tuple(tasks) != ALL_TASKS:
            raise ValueError("Paper-count verification requires the canonical eight tasks in order")
        verify_paper_counts(task_stats)

    summary = {
        "version": "v1",
        "split_type": SPLIT_TYPE,
        "transductive_across_tasks": True,
        "seed": seed,
        "train_ratio": train_ratio,
        "test_ratio": round(1.0 - train_ratio, 12),
        "assignments": len(assignment_rows),
        "train_rows": len(train_records),
        "test_rows": len(test_records),
        "task_stats": task_stats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if not DATASET_CARD.is_file():
        raise FileNotFoundError(f"Hugging Face dataset card template not found: {DATASET_CARD}")
    shutil.copyfile(DATASET_CARD, output_dir / "README.md")
    assignments_path = output_dir / "assignments.jsonl.gz"
    write_gzip_jsonl(assignment_rows, assignments_path)
    train_paths = write_split_shards(train_records, output_dir, "train")
    test_paths = write_split_shards(test_records, output_dir, "test")
    summary["data_files"] = {
        "train": [path.relative_to(output_dir).as_posix() for path in train_paths],
        "test": [path.relative_to(output_dir).as_posix() for path in test_paths],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksum_manifest(
        output_dir,
        [output_dir / "README.md", assignments_path, summary_path, *train_paths, *test_paths],
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the model-independent OntologyBench v1 Hugging Face split."
    )
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tasks", nargs="+", default=list(ALL_TASKS))
    parser.add_argument(
        "--skip-paper-count-validation",
        action="store_true",
        help="Allow custom data without enforcing the paper's expected counts.",
    )
    return parser.parse_args(argv)


def print_summary(summary: dict) -> None:
    print(f"Split type: {summary['split_type']}")
    print(f"Seed: {summary['seed']}")
    print(f"Train relevance pairs: {summary['train_rows']:,}")
    print(f"Test relevance pairs: {summary['test_rows']:,}")
    print()
    print(f"{'Task':42} {'Train Q':>9} {'AvgPos':>8} {'Test Q':>9} {'AvgPos':>8}")
    for task_name, task_stats in summary["task_stats"].items():
        train = task_stats["train"]
        test = task_stats["test"]
        print(
            f"{task_name:42} {train['queries']:>9,} {train['avg_positives']:>8.2f} "
            f"{test['queries']:>9,} {test['avg_positives']:>8.2f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = create_splits(
        tasks_root=args.tasks_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        train_ratio=args.train_ratio,
        seed=args.seed,
        verify_expected_counts=not args.skip_paper_count_validation,
    )
    print_summary(summary)
    print(f"\nHugging Face JSONL artifacts written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted safely")
