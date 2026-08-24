from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ontologybench import TASKS, OntologyBenchBenchmark, list_tasks  # noqa: E402


class TaskLoadingTests(unittest.TestCase):
    def test_registry_exposes_canonical_paper_tier_labels(self):
        expected = {
            "1A-R_gene_document_retrieval": ("Tier 1", "Concept Grounding"),
            "1A-R_hpo_definition_retrieval": ("Tier 1", "Concept Grounding"),
            "1A-R_mondo_definition_retrieval": ("Tier 1", "Concept Grounding"),
            "1B-R_disorder_to_phenotype": ("Tier 2", "Relational Retrieval"),
            "1C-R_phenotype_to_disorder": ("Tier 2", "Relational Retrieval"),
            "2A-R_phenotype_to_gene": ("Tier 2", "Relational Retrieval"),
            "2B-R_disease_to_gene": ("Tier 2", "Relational Retrieval"),
            "3A-multi_phenotype_to_disorder": ("Tier 3", "Compositional Retrieval"),
        }

        self.assertEqual(
            {task.name: (task.tier, task.tier_name) for task in TASKS},
            expected,
        )

    def test_all_eight_packaged_tasks_load_through_public_api(self):
        task_names = list_tasks()
        self.assertEqual(len(task_names), 8)

        benchmark = OntologyBenchBenchmark()
        self.assertEqual([task.name for task in benchmark.tasks], task_names)

        for task_name in task_names:
            with self.subTest(task=task_name):
                queries, documents, qrels = benchmark.load_task(task_name)
                self.assertTrue(queries)
                self.assertTrue(documents)
                self.assertEqual(len(qrels), len(queries))


if __name__ == "__main__":
    unittest.main()
