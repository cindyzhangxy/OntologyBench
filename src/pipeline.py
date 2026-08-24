import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREPROCESS_RUNNER = ROOT / "scripts" / "preprocess" / "run_all.py"


def run(command: list[str]) -> None:
    print(f"→ {' '.join(command)}")
    subprocess.check_call(command, cwd=ROOT)


def step_download() -> None:
    run(["bash", "scripts/download_data.sh"])


def step_preprocess() -> None:
    run([sys.executable, str(PREPROCESS_RUNNER)])


def step_build() -> None:
    run([sys.executable, "-m", "ontologybench.build_all_tasks"])


def step_validate() -> None:
    run([sys.executable, "ontologybench/validate_outputs.py"])


def step_clean() -> None:
    tasks_dir = ROOT / "ontologybench/data/tasks"
    print(f"Cleaning: {tasks_dir}")
    for p in tasks_dir.glob("*"):
        if p.is_file():
            p.unlink()
        else:
            for x in p.rglob("*"):
                if x.is_file():
                    x.unlink()
    print("✓ Cleaned.")


# ------------------------------------------------------
# Command dispatcher
# ------------------------------------------------------
def main():
    steps = {
        "download": step_download,
        "preprocess": step_preprocess,
        "build": step_build,
        "validate": step_validate,
        "clean": step_clean,
        "all": lambda: (step_download(), step_preprocess(), step_build(), step_validate()),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        print("Usage: python pipeline.py [download|preprocess|build|validate|clean|all]")
        sys.exit(1)

    print(f"[OntologyBench] Running step: {sys.argv[1]}")
    steps[sys.argv[1]]()


if __name__ == "__main__":
    main()
