"""
Filters Package for JobHunt Recommendation System

This package contains all job filter implementations and the base filter interface.
All filters inherit from BaseJobFilter and provide consistent filtering capabilities.

Available Filters:
- GenderFilter: Filter jobs based on gender preferences
- WorkTypeFilter: Filter jobs based on work type (Contract, Part-time, Internship, Full-time)
- EducationFilter: Filter jobs based on education requirements


Version: 1.0.0
"""

# Import base filter and exceptions
from .base_filter import (
    BaseJobFilter,
    FilterValidationError,
    FilterApplicationError
)

# Import all filter implementations
from .gender_filter import GenderFilter
from .work_type_filter import WorkTypeFilter
from .education_filter import EducationFilter

# Import convenience functions
from .gender_filter import (
    filter_by_gender,
    get_gender_options,
    filter_inclusive_jobs
)

from .work_type_filter import (
    filter_by_work_type,
    get_work_type_options
)

from .education_filter import (
    filter_by_education,
    get_education_options
)

# Define what gets imported with "from filters import *"
__all__ = [
    # Base classes and exceptions
    'BaseJobFilter',
    'FilterValidationError',
    'FilterApplicationError',
    
    # Filter implementations
    'GenderFilter',
    'WorkTypeFilter', 
    'EducationFilter',
    
    # Convenience functions
    'filter_by_gender',
    'get_gender_options',
    'filter_inclusive_jobs',
    'filter_by_work_type',
    'get_work_type_options',
    'filter_by_education',
    'get_education_options'
]

# Package metadata
__version__ = '1.0.0'


# Available filter types for easy reference
AVAILABLE_FILTERS = {
    'gender': GenderFilter,
    'work_type': WorkTypeFilter,
    'education': EducationFilter
}

def get_available_filter_types():
    """
    Get a list of available filter types.
    
    Returns:
        List[str]: List of available filter type names
    """
    return list(AVAILABLE_FILTERS.keys())

def create_filter(filter_type: str):
    """
    Factory function to create filter instances.
    
    Args:
        filter_type (str): Type of filter to create
        
    Returns:
        BaseJobFilter: Instance of the requested filter
        
    Raises:
        ValueError: If filter type is not available
    """
    if filter_type not in AVAILABLE_FILTERS:
        available_types = ', '.join(get_available_filter_types())
        raise ValueError(f"Filter type '{filter_type}' not available. "
                        f"Available types: {available_types}")
    
    filter_class = AVAILABLE_FILTERS[filter_type]
    return filter_class()