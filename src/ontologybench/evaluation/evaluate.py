# ontologybench/evaluation/evaluate.py

import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ontologybench.evaluation.utils import load_task
from ontologybench.evaluation.metrics import evaluate_run
from ontologybench.evaluation.backends import BM25Backend
from ontologybench.registry import list_tasks


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "tasks_global"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

EVAL_KS = (1, 5, 10)


def is_eval_task(name: str) -> bool:
    return not name.startswith("X")


def get_backend(args):
    if args.backend == "bm25":
        return BM25Backend()

    if args.backend == "sentence-transformer":
        from ontologybench.evaluation.backends.sentence_transformer import SentenceTransformerBackend

        return SentenceTransformerBackend(args.model_name)

    if args.backend == "llm":
        from ontologybench.evaluation.backends.llm_embedding import LLMEmbeddingBackend

        return LLMEmbeddingBackend(args.model_name, batch_size=32)

    if args.backend == "pylate":
        from ontologybench.evaluation.backends.pylate import PyLateReranker

        return PyLateReranker(args.reranker_model)

    raise ValueError(args.backend)


def get_candidate_backend(args):
    if args.backend == "pylate":
        from ontologybench.evaluation.backends.sentence_transformer import SentenceTransformerBackend

        return SentenceTransformerBackend(args.candidate_model)
    return None


def get_run_name(args):
    if args.backend == "bm25":
        return "BM25"
    if args.backend == "llm":
        return "BI-OpenAI"
    if args.backend == "sentence-transformer":
        return f"BI-{args.model_name}"
    if args.backend == "pylate":
        return f"RR-{args.candidate_model}→{args.reranker_model}"
    raise ValueError


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        required=False,
        choices=["bm25", "sentence-transformer", "llm", "pylate"],
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Task directory to evaluate; defaults to the paper split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory for result JSON files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Optional task names to evaluate. Defaults to all registered tasks.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print metrics without writing a result JSON file.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for local smoke tests.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List registered OntologyBench tasks and exit.",
    )

    # two-stage (reranker)
    parser.add_argument("--candidate-model", type=str)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--reranker-model", type=str)

    args = parser.parse_args()
    if args.list_tasks:
        for task_name in list_tasks():
            print(task_name)
        return

    if args.backend is None:
        parser.error("--backend is required unless --list-tasks is used")

    backend = get_backend(args)
    candidate_backend = get_candidate_backend(args)
    run_name = get_run_name(args)

    run_logger = None
    if not args.no_wandb:
        import wandb

        run_logger = wandb.init(
            project="retrieval_result_comparison",
            name=run_name,
            config=vars(args),
        )

    macro_accumulator = defaultdict(list)
    all_results = {}

    selected_tasks = set(args.tasks) if args.tasks else None

    for task_dir in sorted(args.data_root.iterdir()):
        if not task_dir.is_dir() or not is_eval_task(task_dir.name):
            continue
        if selected_tasks is not None and task_dir.name not in selected_tasks:
            continue

        queries, docs, qrels = load_task(task_dir)

        # -------- safety --------
        assert isinstance(queries, list)
        assert isinstance(docs, list)

        # =========================
        # Candidate generation
        # =========================
        if candidate_backend:
            cand_run = candidate_backend.run(
                queries,
                docs,
                top_k=args.candidate_k,
            )

            candidate_docs_per_query = {
                q["query_id"]: [
                    d for d in docs if d["doc_id"] in set(cand_run[q["query_id"]])
                ]
                for q in queries
            }
        else:
            candidate_docs_per_query = {
                q["query_id"]: docs for q in queries
            }

        # =========================
        # Final ranking
        # =========================
        run = {}

        if args.backend == "pylate":
            # PyLate is inherently per-query
            for q in queries:
                qid = q["query_id"]
                ranked = backend.rerank(
                    query_text=q["query"],
                    candidate_docs={
                        d["doc_id"]: d["doc"]
                        for d in candidate_docs_per_query[qid]
                    },
                    top_k=args.max_k,
                )
                run[qid] = ranked

        else:
            # Retrieval backends process all task queries in one batch.
            run = backend.run(
                queries,
                docs,
                top_k=args.max_k,
            )

        # =========================
        # Evaluation
        # =========================
        task_metrics = {}

        for k in EVAL_KS:
            if k > args.max_k:
                continue

            metrics = evaluate_run(run, qrels, k=k)

            for m, v in metrics.items():
                task_metrics[m] = v
                macro_accumulator[m].append(v)

        all_results[task_dir.name] = task_metrics

        if run_logger is not None:
            import wandb

            wandb.log({
                f"{task_dir.name}/{m}": v
                for m, v in task_metrics.items()
            })

        print(f"[OK] {task_dir.name}: {task_metrics}")

    # ---------- macro averages ----------
    macro_avg = {
        f"macro_avg/{m}": sum(vs) / len(vs)
        for m, vs in macro_accumulator.items()
    }

    if run_logger is not None:
        import wandb

        wandb.log(macro_avg)

    if not args.no_save:
        args.output_root.mkdir(parents=True, exist_ok=True)
        out_path = args.output_root / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }

        with open(out_path, "w") as f:
            json.dump(
                {
                    "args": serializable_args,
                    "results": all_results,
                    "macro_avg": macro_avg,
                },
                f,
                indent=2,
            )
        print(f"[OK] Results saved to {out_path}")

    if run_logger is not None:
        import wandb

        wandb.finish()



if __name__ == "__main__":
    main()


# ---------------------------
# Example usage (run from repo root)
# ---------------------------
# conda activate ontologybench-eval
# python -m ontologybench.evaluation.evaluate --backend bm25 --k 10
# python -m ontologybench.evaluation.evaluate --backend sentence-transformer --model-name all-MiniLM-L6-v2 --max-k 10
# python -m ontologybench.evaluation.evaluate --backend sentence-transformer --model-name FremyCompany/BioLORD-2023 --max-k 10
# python -m ontologybench.evaluation.evaluate --backend llm --model-name text-embedding-3-large --max-k 10
# python -m ontologybench.evaluation.evaluate --backend pylate --model-name lightonai/GTE-ModernColBERT-v1 --max-k 10
# python -m ontologybench.evaluation.evaluate \
#   --backend pylate \
#   --candidate-model FremyCompany/BioLORD-2023 \
#   --candidate-k 100 \
#   --reranker-model lightonai/GTE-ModernColBERT-v1 \
#   --k 10
