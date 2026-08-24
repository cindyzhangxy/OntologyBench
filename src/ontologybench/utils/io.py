# ontologybench/utils/io.py

import json
import pandas as pd
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------
# Parquet
# ---------------------------------------------------------

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------
# JSONL
# ---------------------------------------------------------

def save_jsonl(records: Iterable[dict], path: str | Path):
    """Write list of dicts to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out
