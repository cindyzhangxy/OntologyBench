from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ontologybench.verify_data import verify_manifest  # noqa: E402


class VerifyDataTests(unittest.TestCase):
    def test_bundled_checksum_manifest_verifies(self):
        manifest = REPOSITORY_ROOT / "src/ontologybench/data/tasks_global/SHA256SUMS"
        self.assertEqual(verify_manifest(manifest, REPOSITORY_ROOT / "src"), [])

    def test_modified_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.txt").write_text("changed\n", encoding="utf-8")
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{'0' * 64}  data.txt\n", encoding="utf-8")

            failures = verify_manifest(manifest, root)

        self.assertEqual(failures, ["data.txt: checksum mismatch"])


if __name__ == "__main__":
    unittest.main()
