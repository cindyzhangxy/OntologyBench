import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "src" / "scripts" / "create_splits.py"


def load_create_splits_module():
    spec = importlib.util.spec_from_file_location("create_splits", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load split script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class CreateSplitTests(unittest.TestCase):
    def test_build_release_writes_model_independent_huggingface_splits(self):
        create_splits = load_create_splits_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks_root = root / "tasks"
            output_dir = root / "output"
            task_name = "1A-R_gene_document_retrieval"
            task_dir = tasks_root / task_name

            write_jsonl(
                task_dir / "docs.jsonl",
                [
                    {"doc_id": str(index), "doc": f"Positive {index}"}
                    for index in range(5)
                ],
            )
            write_jsonl(
                task_dir / "queries.jsonl",
                [
                    {"query_id": f"q{index}", "query": f"Anchor {index}"}
                    for index in range(5)
                ]
                + [{"query_id": "q-duplicate", "query": "  ANCHOR   0 "}],
            )
            write_jsonl(
                task_dir / "qrels.jsonl",
                [
                    {"query_id": f"q{index}", "positive_doc_ids": [str(index)]}
                    for index in range(5)
                ]
                + [{"query_id": "q-duplicate", "positive_doc_ids": ["0"]}],
            )

            summary = create_splits.create_splits(
                tasks_root=tasks_root,
                output_dir=output_dir,
                tasks=[task_name],
                train_ratio=0.8,
                seed=42,
            )

            assignments = read_gzip_jsonl(output_dir / "assignments.jsonl.gz")
            train_files = sorted((output_dir / "data").glob("train-*.jsonl.gz"))
            test_files = sorted((output_dir / "data").glob("test-*.jsonl.gz"))
            train_rows = [row for path in train_files for row in read_gzip_jsonl(path)]
            test_rows = [row for path in test_files for row in read_gzip_jsonl(path)]
            summary_file = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertTrue(train_files)
            self.assertTrue(test_files)
            self.assertTrue((output_dir / "README.md").is_file())
            self.assertTrue((output_dir / "assignments.jsonl.gz").is_file())
            self.assertTrue((output_dir / "SHA256SUMS").is_file())

            self.assertEqual(
                {row["target_id"] for row in train_rows},
                {
                    f"{task_name}::1",
                    f"{task_name}::2",
                    f"{task_name}::3",
                    f"{task_name}::4",
                },
            )
            self.assertEqual(
                {row["target_id"] for row in test_rows},
                {f"{task_name}::0"},
            )
            self.assertEqual(len(test_rows), 1)
            self.assertTrue(all(row["split"] == "train" for row in train_rows))
            self.assertTrue(all(row["split"] == "test" for row in test_rows))
            self.assertEqual(
                set(train_rows[0]),
                {
                    "task_name",
                    "tier_id",
                    "tier_name",
                    "query_id",
                    "query",
                    "target_id",
                    "raw_target_id",
                    "target",
                    "split",
                },
            )
            self.assertTrue(all(row["tier_id"] == 1 for row in train_rows + test_rows))
            self.assertTrue(
                all(row["tier_name"] == "Concept Grounding" for row in train_rows + test_rows)
            )
            self.assertEqual(len(assignments), 5)
            self.assertEqual(
                {row["target_id"]: row["split"] for row in assignments},
                {
                    f"{task_name}::0": "test",
                    f"{task_name}::1": "train",
                    f"{task_name}::2": "train",
                    f"{task_name}::3": "train",
                    f"{task_name}::4": "train",
                },
            )
            self.assertEqual(summary["train_rows"], 4)
            self.assertEqual(summary["test_rows"], 1)
            self.assertEqual(summary["test_ratio"], 0.2)
            self.assertEqual(
                summary["task_stats"][task_name],
                {
                    "train": {"queries": 4, "avg_positives": 1.0},
                    "test": {"queries": 1, "avg_positives": 1.0},
                },
            )
            self.assertEqual(summary_file, summary)

            checksums = (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            for line in checksums:
                expected, relative = line.split(maxsplit=1)
                artifact = output_dir / relative
                self.assertTrue(artifact.is_file())
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), expected)

    def test_generation_is_byte_deterministic_and_uses_canonical_tier3_name(self):
        create_splits = load_create_splits_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks_root = root / "tasks"
            source_name = "3A_multi_phenotype_to_disorder"
            public_name = "3A-multi_phenotype_to_disorder"
            task_dir = tasks_root / source_name
            write_jsonl(
                task_dir / "docs.jsonl",
                [{"doc_id": str(index), "doc": f"Disease {index}"} for index in range(5)],
            )
            write_jsonl(
                task_dir / "queries.jsonl",
                [{"query_id": f"q{index}", "query": f"Phenotypes {index}"} for index in range(5)],
            )
            write_jsonl(
                task_dir / "qrels.jsonl",
                [
                    {"query_id": f"q{index}", "positive_doc_ids": [str(index)]}
                    for index in range(5)
                ],
            )

            first = root / "first"
            second = root / "second"
            for output in (first, second):
                create_splits.create_splits(
                    tasks_root=tasks_root,
                    output_dir=output,
                    tasks=[public_name],
                    train_ratio=0.8,
                    seed=42,
                )

            first_files = {
                path.relative_to(first).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

            rows = [
                row
                for path in sorted((first / "data").glob("*.jsonl.gz"))
                for row in read_gzip_jsonl(path)
            ]
            self.assertEqual({row["task_name"] for row in rows}, {public_name})
            self.assertEqual({row["tier_id"] for row in rows}, {3})
            self.assertEqual({row["tier_name"] for row in rows}, {"Compositional Retrieval"})
            self.assertFalse(
                {row["target_id"] for row in rows if row["split"] == "train"}
                & {row["target_id"] for row in rows if row["split"] == "test"}
            )


if __name__ == "__main__":
    unittest.main()
