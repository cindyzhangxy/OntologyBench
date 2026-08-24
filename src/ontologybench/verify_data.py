"""Cross-platform verification of the bundled benchmark artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "tasks_global" / "SHA256SUMS"


def verify_manifest(manifest: Path, root: Path) -> list[str]:
    """Return checksum or path failures for a GNU-style SHA-256 manifest."""
    manifest = Path(manifest)
    root = Path(root).resolve()
    failures: list[str] = []

    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split(maxsplit=1)
        except ValueError:
            failures.append(f"line {line_number}: malformed manifest entry")
            continue

        relative = relative.lstrip("*")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            failures.append(f"{relative}: path escapes verification root")
        elif not target.is_file():
            failures.append(f"{relative}: missing")
        else:
            with target.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual.lower() != expected.lower():
                failures.append(f"{relative}: checksum mismatch")

    return failures


def main() -> None:
    failures = verify_manifest(DEFAULT_MANIFEST, PACKAGE_PARENT)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print("[OK] All bundled OntologyBench task checksums match.")


if __name__ == "__main__":
    main()
