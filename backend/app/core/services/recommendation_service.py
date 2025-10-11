import os
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from ..path_utils import get_dataset_path, get_embeddings_dir
from ai_model.feature_engineering.utils.text_processing import clean_and_combine_text
from ai_model.embedding import create_embedding
from ai_model.recommendation.core_recommendation import recommend_jobs


class RecommendationService:
    def __init__(self):
        self.dataset_path = get_dataset_path()
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        self.embeddings_dir = get_embeddings_dir()
        self.jobs_df = pd.read_csv(self.dataset_path)
        # Map dataset experience columns to scorer-required names
        # ExperienceScorer expects 'min_experience' and 'max_experience'
        if 'min_experience' not in self.jobs_df.columns and 'experience_min' in self.jobs_df.columns:
            self.jobs_df['min_experience'] = self.jobs_df['experience_min']
        if 'max_experience' not in self.jobs_df.columns and 'experience_max' in self.jobs_df.columns:
            self.jobs_df['max_experience'] = self.jobs_df['experience_max']

    def recommend(self,
                  role: Optional[str],
                  skills: Optional[str],
                  education: Optional[str],
                  work_type: Optional[Any],
                  experience: Optional[float],
                  latitude: Optional[float],
                  longitude: Optional[float],
                  gender: Optional[str],
                  max_recommendations: int = 10) -> List[Dict[str, Any]]:
        # Prepare text and embedding
        combined_text = clean_and_combine_text(role, skills)
        try:
            query_embedding = create_embedding(combined_text)
        except Exception:
            query_embedding = None

        recs = recommend_jobs(
            jobs_df=self.jobs_df,
            gender=gender,
            work_type=work_type,
            education=education,
            user_experience=experience,
            user_latitude=latitude,
            user_longitude=longitude,
            query_embedding=query_embedding,
            embeddings_dir=self.embeddings_dir,
            max_recommendations=max_recommendations,
        )
        return recs

    @staticmethod
    def _to_native(value: Any) -> Any:
        # Normalize pandas/NumPy scalars to built-in Python types and convert NaN to None
        try:
            if value is None:
                return None
            if pd.isna(value):
                return None
        except Exception:
            # pd.isna may throw on non-array-like objects; ignore
            pass
        # NumPy scalar conversions
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            v = float(value)
            if np.isnan(v):
                return None
            return v
        if isinstance(value, (np.bool_,)):
            return bool(value)
        # pandas Timestamp to ISO string if present
        try:
            import pandas as _pd
            if isinstance(value, _pd.Timestamp):
                return value.isoformat()
        except Exception:
            pass
        return value

    def get_job_by_id(self, job_id: Any) -> Optional[Dict[str, Any]]:
        # Support both 'Job_Id' and 'Job Id' columns
        id_cols = [col for col in ['Job_Id', 'Job Id'] if col in self.jobs_df.columns]
        if not id_cols:
            return None
        mask = None
        for col in id_cols:
            col_mask = (self.jobs_df[col].astype(str) == str(job_id))
            mask = col_mask if mask is None else (mask | col_mask)
        if mask is None or not mask.any():
            return None
        row = self.jobs_df.loc[mask].iloc[0]
        # Build enriched detail dict consistent with recommendation output keys
        detail = {
            'job_id': self._to_native(row.get('Job_Id', row.get('Job Id', 'N/A'))),
            'job_title': self._to_native(row.get('Job Title', 'N/A')),
            'role': self._to_native(row.get('Role', 'N/A')),
            'job_description': self._to_native(row.get('Job Description', 'N/A')),
            'benefits': self._to_native(row.get('Benefits', 'N/A')),
            'skills': self._to_native(row.get('skills', 'N/A')),
            'responsibilities': self._to_native(row.get('Responsibilities', 'N/A')),
            'company': self._to_native(row.get('Company', 'N/A')),
            'company_size': self._to_native(row.get('Company Size', 'N/A')),
            'company_profile': self._to_native(row.get('Company Profile', 'N/A')),
            'experience': self._to_native(row.get('Experience', 'N/A')),
            'salary_range': self._to_native(row.get('Salary Range', 'N/A')),
            'qualifications': self._to_native(row.get('Qualifications', 'N/A')),
            'location': self._to_native(row.get('location', 'N/A')),
            'country': self._to_native(row.get('Country', 'N/A')),
            'work_type': self._to_native(row.get('Work_Type', row.get('Work Type', 'N/A'))),
            'preference': self._to_native(row.get('Preference', 'N/A')),
            'latitude': self._to_native(row.get('latitude', None)),
            'longitude': self._to_native(row.get('longitude', None)),
            'total_score': 0.0,
            'individual_scores': {}
        }
        return detail