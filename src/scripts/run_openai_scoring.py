#!/usr/bin/env python
"""Score saved OntologyBench Tier-3 candidates with the Appendix F prompt.

The runner is resumable: existing ``(query_id, candidate_doc_id)`` records are
not submitted again. It requires the optional ``openai`` dependency and an
``OPENAI_API_KEY`` environment variable. HPC scheduler configuration is
intentionally outside the release contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ontologybench.generative.scoring import (
    PROMPT_SHA256,
    SCORE_SCHEMA,
    build_messages,
    validate_score,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row["query_id"]), str(row["candidate_doc_id"]))
        for row in load_jsonl(path)
        if row.get("query_id") and row.get("candidate_doc_id")
    }


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from openai import OpenAI

    records = load_jsonl(args.input)
    if args.max_queries:
        records = records[: args.max_queries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output)
    client = OpenAI()

    with args.output.open("a", encoding="utf-8", newline="\n") as output:
        for record in records:
            phenotypes = record.get("phenotypes")
            if not isinstance(phenotypes, list):
                raise ValueError("Each input record must contain a 'phenotypes' list")
            for candidate in record["candidates"]:
                key = (str(record["query_id"]), str(candidate["doc_id"]))
                if key in done:
                    continue
                response = client.responses.create(
                    model=args.model,
                    input=build_messages(phenotypes, candidate["text"]),
                    text={"format": SCORE_SCHEMA},
                    temperature=0,
                    max_output_tokens=args.max_output_tokens,
                )
                raw_output = response_text(response)
                parsed = json.loads(raw_output)
                score = validate_score(parsed)
                output.write(
                    json.dumps(
                        {
                            "query_id": key[0],
                            "candidate_doc_id": key[1],
                            "score": score,
                            "model": args.model,
                            "prompt_sha256": PROMPT_SHA256,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
