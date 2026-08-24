from pathlib import Path
import tomllib
import unittest

from setuptools import find_namespace_packages


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PackageContentsTests(unittest.TestCase):
    def test_package_discovery_excludes_local_and_generated_directories(self):
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)["tool"]["setuptools"]["packages"]["find"]

        packages = find_namespace_packages(
            where=REPOSITORY_ROOT / config["where"][0],
            include=config.get("include", ["*"]),
            exclude=config.get("exclude", []),
        )

        self.assertTrue(packages)
        self.assertTrue(
            all(
                package == "ontologybench"
                or package.startswith("ontologybench.")
                for package in packages
            ),
            packages,
        )
        self.assertFalse(
            any(
                package == prefix or package.startswith(f"{prefix}.")
                for package in packages
                for prefix in (
                    "ontologybench.data.tasks",
                    "ontologybench.figures",
                    "ontologybench.results",
                )
            ),
            packages,
        )


if __name__ == "__main__":
    unittest.main()
