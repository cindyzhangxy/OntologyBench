"""Task registry for the OntologyBench paper split."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskInfo:
    name: str
    tier: str
    tier_name: str
    query_type: str
    doc_type: str
    num_queries: int
    num_docs: int
    avg_positives: float


TASKS: tuple[TaskInfo, ...] = (
    TaskInfo(
        name="1A-R_gene_document_retrieval",
        tier="Tier 1",
        tier_name="Concept Grounding",
        query_type="Gene alias",
        doc_type="Gene document",
        num_queries=431,
        num_docs=431,
        avg_positives=1.00,
    ),
    TaskInfo(
        name="1A-R_hpo_definition_retrieval",
        tier="Tier 1",
        tier_name="Concept Grounding",
        query_type="Phenotype alias",
        doc_type="HPO definition",
        num_queries=4581,
        num_docs=1701,
        avg_positives=1.00,
    ),
    TaskInfo(
        name="1A-R_mondo_definition_retrieval",
        tier="Tier 1",
        tier_name="Concept Grounding",
        query_type="Disease alias",
        doc_type="MONDO definition",
        num_queries=9205,
        num_docs=1252,
        avg_positives=1.00,
    ),
    TaskInfo(
        name="1B-R_disorder_to_phenotype",
        tier="Tier 2",
        tier_name="Relational Retrieval",
        query_type="Disease definition",
        doc_type="HPO definition",
        num_queries=6291,
        num_docs=1710,
        avg_positives=5.63,
    ),
    TaskInfo(
        name="1C-R_phenotype_to_disorder",
        tier="Tier 2",
        tier_name="Relational Retrieval",
        query_type="Phenotype definition",
        doc_type="MONDO definition",
        num_queries=5229,
        num_docs=1372,
        avg_positives=6.57,
    ),
    TaskInfo(
        name="2A-R_phenotype_to_gene",
        tier="Tier 2",
        tier_name="Relational Retrieval",
        query_type="Phenotype definition",
        doc_type="Gene document",
        num_queries=1507,
        num_docs=447,
        avg_positives=1.00,
    ),
    TaskInfo(
        name="2B-R_disease_to_gene",
        tier="Tier 2",
        tier_name="Relational Retrieval",
        query_type="Disease definition",
        doc_type="Gene document",
        num_queries=6504,
        num_docs=493,
        avg_positives=5.31,
    ),
    TaskInfo(
        name="3A-multi_phenotype_to_disorder",
        tier="Tier 3",
        tier_name="Compositional Retrieval",
        query_type="Phenotype triplet",
        doc_type="MONDO definition",
        num_queries=2537,
        num_docs=1293,
        avg_positives=2.26,
    ),
)


def list_tasks() -> list[str]:
    return [task.name for task in TASKS]


def get_task(name: str) -> TaskInfo:
    for task in TASKS:
        if task.name == name:
            return task
    raise KeyError(f"Unknown OntologyBench task: {name}")
