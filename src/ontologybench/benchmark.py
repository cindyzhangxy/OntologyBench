"""MTEB-style loading and evaluation helpers for OntologyBench."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol

from ontologybench.evaluation.metrics import evaluate_run
from ontologybench.evaluation.utils import load_task
from ontologybench.registry import TASKS, TaskInfo, get_task


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "tasks_global"


class Retriever(Protocol):
    def run(self, queries: list[dict], docs: list[dict], top_k: int) -> dict[str, list[str]]:
        ...


class OntologyBenchBenchmark:
    """Load and evaluate OntologyBench retrieval tasks.

    Retrievers can be objects with a ``run(queries, docs, top_k)`` method or
    callables with the same signature.
    """

    def __init__(
        self,
        tasks: Iterable[str] | None = None,
        data_root: str | Path = DEFAULT_DATA_ROOT,
    ) -> None:
        self.data_root = Path(data_root)
        names = list(tasks) if tasks is not None else [task.name for task in TASKS]
        self.tasks: list[TaskInfo] = [get_task(name) for name in names]

    def load_task(self, task_name: str) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
        return load_task(self.data_root / task_name)

    def evaluate(
        self,
        retriever: Retriever | Callable[[list[dict], list[dict], int], dict[str, list[str]]],
        k_values: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, dict[str, float]]:
        max_k = max(k_values)
        results: dict[str, dict[str, float]] = {}

        for task in self.tasks:
            queries, docs, qrels = self.load_task(task.name)
            if hasattr(retriever, "run"):
                run = retriever.run(queries, docs, top_k=max_k)
            else:
                run = retriever(queries, docs, max_k)

            task_results: dict[str, float] = {}
            for k in k_values:
                task_results.update(evaluate_run(run, qrels, k=k))
            results[task.name] = task_results

        return results
