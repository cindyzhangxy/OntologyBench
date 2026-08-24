import csv
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results" / "paper"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PaperResultTests(unittest.TestCase):
    def test_retrieval_results_cover_every_paper_model_and_task(self):
        rows = read_csv("retrieval_results.csv")
        embedding_rows = [row for row in rows if row["system_type"] == "text_retriever"]
        reference_rows = [row for row in rows if row["system_type"] == "ontology_reference"]

        self.assertEqual(len(embedding_rows), 12 * 8)
        self.assertEqual(len(reference_rows), 4)
        self.assertEqual(len({row["task"] for row in embedding_rows}), 8)

        index = {(row["model"], row["task"]): row for row in rows}
        self.assertAlmostEqual(
            float(index[("Qwen3-Embed-0.6B‡", "3A-multi_phenotype_to_disorder")]["ndcg_at_10"]),
            0.312561,
            places=6,
        )
        self.assertAlmostEqual(
            float(index[("Phenomizer-style", "3A-multi_phenotype_to_disorder")]["ndcg_at_10"]),
            0.738714,
            places=6,
        )

    def test_reranking_results_preserve_paper_values_and_provenance_limits(self):
        rows = read_csv("reranking_results.csv")
        self.assertEqual(len(rows), 9)
        index = {row["model"]: row for row in rows}

        self.assertEqual(index["Qwen3-Embed-0.6B‡"]["paper_ndcg_at_10"], "0.313")
        self.assertEqual(
            index["Qwen3-Embed-0.6B‡"]["provenance_status"],
            "reported_no_matching_saved_artifact",
        )
        self.assertEqual(index["Qwen3.6-27B"]["paper_ndcg_at_10"], "0.315")
        self.assertEqual(
            index["Qwen3-4B-Instruct-2507"]["provenance_status"],
            "reported_no_matching_saved_artifact",
        )
        self.assertEqual(
            index["Qwen3-4B-Thinking-2507"]["provenance_status"],
            "reported_invalid_output",
        )
        self.assertEqual(
            index["OpenAI (GPT-5.4-mini)"]["provenance_status"],
            "verified_scores_prompt_provenance_unresolved",
        )

    def test_manifest_records_only_relative_source_paths_and_sha256_hashes(self):
        manifest = json.loads((RESULTS_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_workspace"], "OntologyBench archival workspace")
        self.assertTrue(manifest["artifacts"])
        for artifact in manifest["artifacts"]:
            self.assertFalse(Path(artifact["source_path"]).is_absolute())
            self.assertEqual(len(artifact["sha256"]), 64)
            int(artifact["sha256"], 16)


if __name__ == "__main__":
    unittest.main()
