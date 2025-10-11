import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path(r"d:/ai courses/JobHunt"),
    Path(r"d:\ai courses\JobHunt"),
]


def find_file(rel_path: Path, fallback_abs: Optional[str] = None) -> str:
    """Search for a file across common project roots; fall back to provided absolute path."""
    for base in PROJECT_ROOT_CANDIDATES:
        candidate = (base / rel_path).resolve()
        if candidate.exists():
            return str(candidate)
    return fallback_abs or str(rel_path)


def get_dataset_path() -> str:
    return find_file(Path("dataset/complete_dataset10k.csv"), r"d:\ai courses\JobHunt\dataset\complete_dataset10k.csv")


def get_embeddings_dir() -> str:
    # Directory with precomputed job embeddings job_embeddings.npy and job_ids.pkl
    for p in [
        Path("embeddings"),
        Path(r"d:/ai courses/JobHunt/embeddings"),
        Path(r"d:\ai courses\JobHunt\embeddings"),
    ]:
        if p.exists():
            return str(p.resolve())
    return r"d:\ai courses\JobHunt\embeddings"


def get_salary_artifacts_dir() -> str:
    # Directory containing salary model artifacts saved by the notebook
    for p in [
        Path(r"d:/ai courses/JobHunt/ai_model/salary_predection/model"),
        Path(r"d:\ai courses\JobHunt\ai_model\salary_predection\model"),
        Path("ai_model/salary_predection/model"),
    ]:
        if p.exists():
            return str(p.resolve())
    return r"d:\ai courses\JobHunt\ai_model\salary_predection\model"