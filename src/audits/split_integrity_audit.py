#!/usr/bin/env python3
"""Audit query/relevant-document leakage across OntologyBench split roles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RoleTask:
    role: str
    task: str
    directory: Path
    queries: dict[str, str]
    relevant_pairs: set[tuple[str, str]]
    relevant_docs: set[str]


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def canonical_doc_id(doc_id: str) -> str:
    return doc_id.split("::", 1)[-1]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_role_task(role: str, directory: Path) -> RoleTask:
    query_rows = list(read_jsonl(directory / "queries.jsonl"))
    qrel_rows = list(read_jsonl(directory / "qrels.jsonl"))
    doc_rows = list(read_jsonl(directory / "docs.jsonl"))

    queries: dict[str, str] = {}
    for row in query_rows:
        query_id = str(row["query_id"])
        if query_id in queries:
            raise ValueError(f"{directory}: duplicate query_id {query_id}")
        queries[query_id] = normalize_text(str(row["query"]))

    docs = {canonical_doc_id(str(row["doc_id"])) for row in doc_rows}
    relevant_pairs: set[tuple[str, str]] = set()
    relevant_docs: set[str] = set()
    for row in qrel_rows:
        query_id = str(row["query_id"])
        if query_id not in queries:
            raise ValueError(f"{directory}: qrel references unknown query {query_id}")
        for raw_doc_id in row.get("positive_doc_ids", []):
            doc_id = canonical_doc_id(str(raw_doc_id))
            if doc_id not in docs:
                raise ValueError(f"{directory}: qrel references unknown doc {raw_doc_id}")
            relevant_pairs.add((queries[query_id], doc_id))
            relevant_docs.add(doc_id)

    return RoleTask(
        role=role,
        task=directory.name.removeprefix("[training]_"),
        directory=directory,
        queries=queries,
        relevant_pairs=relevant_pairs,
        relevant_docs=relevant_docs,
    )


def discover(data_root: Path, include_legacy: bool) -> list[RoleTask]:
    directories: list[tuple[str, Path]] = []
    directories.extend(
        ("training", path)
        for path in sorted(data_root.iterdir())
        if path.is_dir() and path.name.startswith("[training]_")
    )
    global_root = data_root / "tasks_global"
    directories.extend(("evaluation", path) for path in sorted(global_root.iterdir()) if path.is_dir())
    if include_legacy:
        legacy_root = data_root / "past"
        directories.extend(("legacy", path) for path in sorted(legacy_root.iterdir()) if path.is_dir())

    tasks = []
    for role, directory in directories:
        required = [directory / name for name in ("queries.jsonl", "qrels.jsonl", "docs.jsonl")]
        if all(path.exists() for path in required):
            tasks.append(load_role_task(role, directory))
    return tasks


def audit(tasks: list[RoleTask]) -> dict:
    by_task: dict[str, list[RoleTask]] = defaultdict(list)
    for item in tasks:
        by_task[item.task].append(item)

    comparisons = []
    failures = []
    for task_name, role_tasks in sorted(by_task.items()):
        for index, left in enumerate(role_tasks):
            for right in role_tasks[index + 1 :]:
                query_overlap = sorted(set(left.queries.values()) & set(right.queries.values()))
                relevant_overlap = sorted(left.relevant_pairs & right.relevant_pairs)
                comparison = {
                    "task": task_name,
                    "left_role": left.role,
                    "right_role": right.role,
                    "left_dir": str(left.directory),
                    "right_dir": str(right.directory),
                    "query_text_overlap": len(query_overlap),
                    "relevant_pair_overlap": len(relevant_overlap),
                    "relevant_doc_overlap": len(left.relevant_docs & right.relevant_docs),
                    "query_examples": query_overlap[:5],
                    "relevant_pair_examples": [list(pair) for pair in relevant_overlap[:5]],
                }
                comparisons.append(comparison)
                if relevant_overlap:
                    failures.append(comparison)

    return {
        "status": "PASS" if not failures else "FAIL",
        "criterion": "zero normalized query-text + relevant-document overlap across roles",
        "tasks_loaded": len(tasks),
        "comparisons": comparisons,
        "normalized_query_overlap_total": sum(
            item["query_text_overlap"] for item in comparisons
        ),
        "relevant_pair_overlap_total": sum(
            item["relevant_pair_overlap"] for item in comparisons
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--include-legacy", action="store_true", help="Include data/past as a comparison role.")
    parser.add_argument("--json-out", type=Path, help="Also write the machine-readable audit report here.")
    args = parser.parse_args()

    try:
        report = audit(discover(args.data_root, args.include_legacy))
    except (OSError, KeyError, ValueError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"STATUS: {report['status']}")
    print(f"TASKS LOADED: {report['tasks_loaded']}")
    print(f"ROLE COMPARISONS: {len(report['comparisons'])}")
    print(f"NORMALIZED QUERY TEXT OVERLAPS: {report['normalized_query_overlap_total']}")
    print(f"RELEVANT CROSS-ROLE OVERLAPS: {report['relevant_pair_overlap_total']}")
    for item in report["comparisons"]:
        print(
            f"{item['task']} [{item['left_role']} vs {item['right_role']}]: "
            f"relevant_pairs={item['relevant_pair_overlap']} "
            f"query_text={item['query_text_overlap']} "
            f"relevant_docs={item['relevant_doc_overlap']}"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
