from pathlib import Path
import sys
import tomllib
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


class ReleaseContractTests(unittest.TestCase):
    def test_camera_ready_repository_files_exist(self):
        required = (
            ".github/workflows/ci.yml",
            ".gitignore",
            ".gitattributes",
            "DATA_LICENSE.md",
            "PROVENANCE.md",
            "MANIFEST.in",
            "assets/ontologybench_overview.pdf",
            "assets/ontologybench_overview.png",
            "paper/OntologyBench.pdf",
        )
        missing = [path for path in required if not (REPOSITORY_ROOT / path).is_file()]
        self.assertEqual(missing, [])

    def test_project_metadata_matches_the_final_paper(self):
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(
            project["description"],
            "A tiered benchmark for structured biomedical knowledge retrieval",
        )
        self.assertEqual(
            [author["name"] for author in project["authors"]],
            ["Xiao Yu Cindy Zhang", "Wyeth W. Wasserman", "Jian Zhu"],
        )
        self.assertEqual(project["license"], "MIT")
        self.assertNotIn("License :: OSI Approved :: MIT License", project["classifiers"])

    def test_public_preprocessing_does_not_require_gene_ontology(self):
        download = (REPOSITORY_ROOT / "src/scripts/download_data.sh").read_text(encoding="utf-8")
        runner = (REPOSITORY_ROOT / "src/scripts/preprocess/run_all.py").read_text(encoding="utf-8")
        self.assertNotIn("go.json", download)
        self.assertNotIn('"go.json"', runner)
        self.assertFalse((REPOSITORY_ROOT / "src/scripts/patch_go_data.py").exists())

    def test_readme_does_not_claim_distribution_of_training_state(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("checkpoints and trainer states are not distributed", readme.lower())
        self.assertIn("OntologyBench: Can Dense Retrieval Satisfy Structured Biomedical Constraints?", readme)


if __name__ == "__main__":
    unittest.main()
