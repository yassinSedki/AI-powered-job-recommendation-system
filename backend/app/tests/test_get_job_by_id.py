import os
import sys
from pathlib import Path
# Ensure project root (JobHunt) is on sys.path so 'backend' package is importable during pytest collection
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backend.app.core.services.recommendation_service import RecommendationService


def make_service_with_df(df: pd.DataFrame) -> RecommendationService:
    """Helper to construct service and replace its jobs_df with a provided DataFrame."""
    svc = RecommendationService()
    # Replace the loaded dataset with our controlled test DataFrame
    svc.jobs_df = df
    return svc


def test_get_job_by_id_returns_detail_and_normalizes_values():
    # Build a DataFrame with mixed NumPy/pandas scalars and NaN to verify normalization
    df = pd.DataFrame({
        'Job_Id': [np.int64(12345)],
        'Job Title': ['Data Scientist'],
        'Role': ['ML Engineer'],
        'Job Description': ['Build ML models'],
        'Benefits': [np.nan],
        'skills': ['Python, ML, Data'],
        'Responsibilities': [None],
        'Company': ['Acme Corp'],
        'Company Size': [np.int64(500)],
        'Company Profile': ['Technology company'],
        'Experience': [np.float64(3.0)],
        'Salary Range': ['50k-70k'],
        'Qualifications': ['Master'],
        'location': ['New York'],
        'Country': ['USA'],
        # Use alternate work type column name to test fallback
        'Work Type': ['Full-time'],
        'Preference': ['both'],
        'latitude': [np.float64(40.7128)],
        'longitude': [np.float64(-74.0060)],
    })

    svc = make_service_with_df(df)
    res = svc.get_job_by_id(12345)

    assert res is not None, "Expected a job detail dict when ID exists"
    # Normalization checks
    assert res['job_id'] == 12345 and isinstance(res['job_id'], int)
    assert res['company_size'] == 500 and isinstance(res['company_size'], int)
    assert res['benefits'] is None, "NaN should be converted to None"
    assert res['responsibilities'] is None, "None should remain None"
    assert isinstance(res['latitude'], float) and isinstance(res['longitude'], float)
    assert res['work_type'] == 'Full-time', "Should read from 'Work Type' if 'Work_Type' absent"
    # Presence and defaults
    assert isinstance(res['total_score'], float) and res['total_score'] == 0.0
    assert isinstance(res['individual_scores'], dict)


def test_get_job_by_id_not_found_returns_none():
    df = pd.DataFrame({
        'Job_Id': [1],
        'Job Title': ['Any'],
    })
    svc = make_service_with_df(df)
    assert svc.get_job_by_id(999) is None


def test_get_job_by_id_supports_alternate_id_column():
    df = pd.DataFrame({
        'Job Id': ['ABC123'],
        'Job Title': ['QA Engineer'],
    })
    svc = make_service_with_df(df)
    res = svc.get_job_by_id('ABC123')
    assert res is not None
    assert res['job_id'] == 'ABC123'