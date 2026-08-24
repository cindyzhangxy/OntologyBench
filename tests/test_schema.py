import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import ontologybench  # noqa: E402
from ontologybench import list_tasks  # noqa: E402


DATA_ROOT = Path(ontologybench.__file__).resolve().parent / "data" / "tasks_global"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SchemaTests(unittest.TestCase):
    def test_packaged_tasks_have_valid_retrieval_schema(self):
        for task_name in list_tasks():
            with self.subTest(task=task_name):
                task_dir = DATA_ROOT / task_name
                documents = read_jsonl(task_dir / "docs.jsonl")
                queries = read_jsonl(task_dir / "queries.jsonl")
                qrels = read_jsonl(task_dir / "qrels.jsonl")

                self.assertTrue(documents)
                self.assertTrue(queries)
                self.assertTrue(qrels)

                self.assertTrue(all({"doc_id", "doc"} <= row.keys() for row in documents))
                self.assertTrue(all({"query_id", "query"} <= row.keys() for row in queries))
                self.assertTrue(
                    all({"query_id", "positive_doc_ids"} <= row.keys() for row in qrels)
                )

                document_ids = [row["doc_id"] for row in documents]
                query_ids = [row["query_id"] for row in queries]
                qrel_query_ids = [row["query_id"] for row in qrels]
                self.assertEqual(len(document_ids), len(set(document_ids)))
                self.assertEqual(len(query_ids), len(set(query_ids)))
                self.assertEqual(len(qrel_query_ids), len(set(qrel_query_ids)))

                document_id_set = set(document_ids)
                query_id_set = set(query_ids)
                self.assertEqual(set(qrel_query_ids), query_id_set)
                for row in qrels:
                    self.assertTrue(row["positive_doc_ids"])
                    self.assertTrue(set(row["positive_doc_ids"]) <= document_id_set)


if __name__ == "__main__":
    unittest.main()
