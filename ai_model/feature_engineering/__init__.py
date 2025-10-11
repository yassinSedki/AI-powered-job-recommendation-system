"""Feature Engineering Package for JobHunt Dataset

This package provides a complete feature engineering pipeline for job data processing.
It includes modular transformations for qualifications, work types, experience, salary,
and text processing, all orchestrated through the main FeatureEngineeringPipeline class.

Main Components:
- FeatureEngineeringPipeline: Main orchestrator class
- Individual transformation functions from utils modules

Usage:
    from feature_engineering import FeatureEngineeringPipeline
    
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.transform(data)
"""

from .pipeline import FeatureEngineeringPipeline
from .utils.qualifications import classify_qualification, process_qualifications
from .utils.work_type import standardize_work_type, process_work_types
from .utils.experience import parse_experience_range, process_experience
from .utils.salary import parse_salary_range, process_salary
from .utils.text_processing import clean_and_combine_text, process_text_features

__version__ = "1.0.0"


__all__ = [
    "FeatureEngineeringPipeline",
    "classify_qualification",
    "process_qualifications",
    "standardize_work_type",
    "process_work_types",
    "parse_experience_range",
    "process_experience",
    "parse_salary_range",
    "process_salary",
    "clean_and_combine_text",
    "process_text_features"
]