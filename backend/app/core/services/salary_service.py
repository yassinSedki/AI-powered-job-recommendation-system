import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from joblib import load
import sys

from ..path_utils import get_salary_artifacts_dir

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class SalaryPredictionService:
    def __init__(self):
        artifacts_dir = Path(get_salary_artifacts_dir())
        self.model_path = artifacts_dir / "salary_predection_model.pkl"
        self.pca_path = artifacts_dir / "pca_embeddings.pkl"
        self.mapper_path = artifacts_dir / "feature_mapper.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Salary model not found at {self.model_path}")
        if not self.pca_path.exists():
            raise FileNotFoundError(f"PCA transformer not found at {self.pca_path}")
        if not self.mapper_path.exists():
            raise FileNotFoundError(f"feature_mapper.json not found at {self.mapper_path}")

        self.model = load(str(self.model_path))
        self.pca = load(str(self.pca_path))
        with open(self.mapper_path, "r") as f:
            mapper = json.load(f)
        self.feature_cols: List[str] = mapper.get("feature_cols", [])
        self.pca_cols: List[str] = mapper.get("pca_cols", [])

    @staticmethod
    def _normalize_work_type(work_type: Optional[Any]) -> Dict[str, int]:
        flags = {
            "is_full_time": 0,
            "is_part_time": 0,
            "is_internship": 0,
            "is_contract": 0,
        }
        if work_type is None:
            return flags
        # Accept a single category string or a collection of categories
        values: List[str] = []
        if isinstance(work_type, str):
            values = [work_type]
        elif isinstance(work_type, (list, tuple, set)):
            values = list(work_type)
        else:
            values = [str(work_type)]
        for v in values:
            s = str(v).strip().lower()
            if s in ("full-time", "full time"):
                flags["is_full_time"] = 1
            elif s in ("part-time", "part time"):
                flags["is_part_time"] = 1
            elif s == "internship":
                flags["is_internship"] = 1
            elif s == "contract":
                flags["is_contract"] = 1
        return flags

    @staticmethod
    def _normalize_qualification(qualification: Optional[str]) -> Dict[str, int]:
        flags = {"is_bachelor": 0, "is_master": 0, "is_phd": 0}
        if not qualification:
            return flags
        s = str(qualification).strip().lower()
        if s == "phd":
            flags["is_phd"] = 1
            flags["is_master"] = 1
            flags["is_bachelor"] = 1
        elif s == "master":
            flags["is_master"] = 1
            flags["is_bachelor"] = 1
        elif s == "bachelor":
            flags["is_bachelor"] = 1
        return flags

    @staticmethod
    def _normalize_gender_pref(pref: Optional[str]) -> Dict[str, int]:
        flags = {"prefers_male": 0, "prefers_female": 0, "has_preference": 0}
        if not pref:
            return flags
        s = str(pref).strip().lower()
        if s == "male":
            flags["prefers_male"] = 1
        elif s == "female":
            flags["prefers_female"] = 1
        return flags

    @staticmethod
    def _parse_experience(exp: Optional[Any], fallback: float = 0.0) -> float:
        # Accept direct numeric input; no text parsing
        try:
            return float(exp) if exp is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def predict(self, *,
                embedding_1024: np.ndarray,
                work_type: Optional[Any] = None,
                qualification: Optional[str] = None,
                preference: Optional[str] = None,
                experience_mid: Optional[Any] = None,
                latitude: Optional[float] = None,
                longitude: Optional[float] = None) -> float:
        # 1) Map categorical inputs to flags
        wt = self._normalize_work_type(work_type)
        qf = self._normalize_qualification(qualification)
        gp = self._normalize_gender_pref(preference)
        exp_mid = self._parse_experience(experience_mid, fallback=3.0)
        lat = float(latitude) if latitude is not None else 0.0
        lng = float(longitude) if longitude is not None else 0.0

        # 2) PCA transform embeddings
        emb = np.asarray(embedding_1024, dtype=float).reshape(1, -1)
        emb_pca = self.pca.transform(emb)
        # Build feature dict
        features: Dict[str, float] = {
            "experience_mid": float(exp_mid),
            "latitude": lat,
            "longitude": lng,
            **wt,
            **qf,
            **gp,
        }
        for i, val in enumerate(emb_pca.flatten(), start=1):
            features[f"emb_pca_{i}"] = float(val)
        # 3) Align to feature_cols order
        X = np.array([[features.get(col, 0.0) for col in self.feature_cols]], dtype=float)
        pred = float(self.model.predict(X)[0])
        return pred