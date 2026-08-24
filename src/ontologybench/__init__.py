"""OntologyBench benchmark package."""

from ontologybench.registry import TASKS, get_task, list_tasks
from ontologybench.benchmark import OntologyBenchBenchmark

__all__ = [
    "OntologyBenchBenchmark",
    "TASKS",
    "get_task",
    "list_tasks",
]
