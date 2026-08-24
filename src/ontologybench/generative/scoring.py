"""Canonical Appendix F prompt and output validation.

This module defines the public prompt contract. It does not assert that every
historical saved score was generated with this exact serialized prompt; see
``results/paper/README.md`` for the artifact-level provenance status.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


SYSTEM_INSTRUCTION = (
    "You are evaluating the textual compatibility between a candidate disorder description "
    "and a set of patient phenotypes.\n"
    "Use ONLY the information provided in the phenotype list and the candidate disorder "
    "description. Do NOT use external medical knowledge."
)

USER_TEMPLATE = """Patient phenotypes
{phenotypes}

Candidate disorder description
{candidate_description}

Task
Assign an integer compatibility score from 0 to 100 indicating how well the candidate disorder description supports the complete phenotype set.

Scoring guideline
- 0-19: incompatible
- 20-39: weak overlap
- 40-59: partial match
- 60-79: strong match
- 80-100: near-complete match

Output rules
Output exactly one JSON object containing only the score. Do not include explanations, Markdown, additional fields, or repeated output.
{{"score": 0}}"""

_FINGERPRINT_TEMPLATE = USER_TEMPLATE.replace("{{", "{").replace("}}", "}")
PROMPT_SHA256 = hashlib.sha256(
    f"{SYSTEM_INSTRUCTION}\n\n{_FINGERPRINT_TEMPLATE}".encode("utf-8")
).hexdigest()

SCORE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "ontologybench_candidate_score",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["score"],
        "additionalProperties": False,
    },
}


def build_messages(
    phenotypes: Sequence[str], candidate_description: str
) -> list[dict[str, str]]:
    """Return the two-message Appendix F prompt for one candidate."""

    cleaned = [str(phenotype).strip() for phenotype in phenotypes if str(phenotype).strip()]
    if not cleaned:
        raise ValueError("At least one non-empty phenotype is required")
    candidate_description = str(candidate_description).strip()
    if not candidate_description:
        raise ValueError("A non-empty candidate disorder description is required")

    phenotype_list = "\n".join(f"- {phenotype}" for phenotype in cleaned)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                phenotypes=phenotype_list,
                candidate_description=candidate_description,
            ),
        },
    ]


def validate_score(value: Any) -> int:
    """Validate and return the strict Appendix F integer score."""

    if not isinstance(value, Mapping) or set(value) != {"score"}:
        raise ValueError("Score output must be an object containing only 'score'")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError("Score must be an integer")
    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100 inclusive")
    return score
