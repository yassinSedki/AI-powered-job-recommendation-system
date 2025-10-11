"""Feature Engineering Pipeline for JobHunt Dataset

This package provides modular feature transformation functions for the JobHunt dataset.
Each transformation is implemented in a separate module for maximum reusability.

Modules:
- qualifications: Qualification categorization (Bachelor, Master, PhD)
- work_type: Work type standardization (Contract, Part-time, Internship, Full-time)
- experience: Experience range parsing and numerical conversion
- salary: Salary range parsing and numerical conversion
- text_processing: Text cleaning and normalization for roles, titles, and skills
- pipeline: Main orchestrator for all transformations
"""

from .qualifications import classify_qualification, process_qualifications
from .work_type import standardize_work_type, process_work_types
from .experience import parse_experience_range, process_experience
from .salary import parse_salary_range, process_salary
from .text_processing import clean_and_combine_text, process_text_features
# FeatureEngineeringPipeline is imported from parent package

__version__ = "1.0.0"


__all__ = [
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