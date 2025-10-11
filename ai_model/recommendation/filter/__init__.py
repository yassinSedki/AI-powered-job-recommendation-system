"""
JobHunt Filter Package

This package provides a comprehensive filtering system for job data with the following components:

Filters:
- EducationFilter: Filter jobs by educational qualification requirements
- WorkTypeFilter: Filter jobs by work type categories (supports multiple selections)
- GenderFilter: Filter jobs by gender preferences with special "both" logic

Manager:
- FilterManager: Combine and manage multiple filters with AND/OR logic

Base:
- BaseJobFilter: Abstract base class for all filters

Usage Examples:
    # Individual filters
    from ai_model.recommendation.filter import EducationFilter, WorkTypeFilter, GenderFilter
    
    education_filter = EducationFilter()
    work_filter = WorkTypeFilter()
    gender_filter = GenderFilter()
    
    # Filter manager for complex scenarios
    from ai_model.recommendation.filter import FilterManager
    
    manager = FilterManager()
    filtered_jobs = manager.apply_combined_filter(df, {
        'education': {'criteria': 'Bachelor'},
        'work_type': {'criteria': ['Full-time', 'Remote']},
        'gender': {'criteria': 'both'}
    })
    
    # Convenience functions
    from ai_model.recommendation.filter import filter_jobs_for_candidate
    
    suitable_jobs = filter_jobs_for_candidate(
        df, 
        education_level='Master',
        work_types=['Full-time'],
        gender='female'
    )


Version: 1.0.0
"""

# Import all filter classes from the filters package
from .filters import (
    BaseJobFilter, 
    FilterValidationError, 
    FilterApplicationError,
    EducationFilter,
    WorkTypeFilter,
    GenderFilter,
    filter_by_education,
    get_education_options,
    filter_by_work_type,
    get_work_type_options,
    filter_by_gender,
    get_gender_options,
    filter_inclusive_jobs,
    AVAILABLE_FILTERS,
    get_available_filter_types,
    create_filter
)

# Import the FilterManager
from .filter_manager import (
    FilterManager,
    create_filter_manager,
    apply_filters,
    get_available_filter_options
)

# Package metadata
__version__ = "1.0.0"

__email__ = "ai-team@jobhunt.com"

# Define what gets imported with "from filter import *"
__all__ = [
    # Base classes and exceptions
    'BaseJobFilter',
    'FilterValidationError', 
    'FilterApplicationError',
    
    # Filter implementations
    'EducationFilter',
    'WorkTypeFilter', 
    'GenderFilter',
    
    # Filter manager
    'FilterManager',
    
    # Convenience functions - Education
    'filter_by_education',
    'get_education_options',
    
    # Convenience functions - Work Type
    'filter_by_work_type',
    'get_work_type_options',
    
    # Convenience functions - Gender
    'filter_by_gender',
    'get_gender_options',
    'filter_inclusive_jobs',
    
    # Filter management utilities
    'AVAILABLE_FILTERS',
    'get_available_filter_types',
    'create_filter',
    
    # Manager convenience functions
    'create_filter_manager',
    'apply_filters',
    'get_available_filter_options'
]

# Package-level convenience functions
def create_filter_suite():
    """
    Create a complete filter suite with all available filters.
    
    Returns:
        dict: Dictionary containing all filter instances
    """
    return {
        'education': EducationFilter(),
        'work_type': WorkTypeFilter(),
        'gender': GenderFilter(),
        'manager': FilterManager()
    }

def get_filter_info():
    """
    Get information about all available filters.
    
    Returns:
        dict: Information about each filter type
    """
    return {
        'education': {
            'class': 'EducationFilter',
            'description': 'Filter jobs by educational qualification requirements',
            'column': 'qualification_category',
            'valid_options': ['Bachelor', 'Master', 'PhD'],
            'supports_hierarchy': True
        },
        'work_type': {
            'class': 'WorkTypeFilter',
            'description': 'Filter jobs by work type categories',
            'column': 'Work_Type',
            'supports_multiple': True,
            'supports_partial_match': True,
            'supports_exclusion': True
        },
        'gender': {
            'class': 'GenderFilter',
            'description': 'Filter jobs by gender preferences with special "both" logic',
            'column': 'Preference',
            'valid_options': ['both', 'male', 'female'],
            'special_logic': 'Jobs marked as "both" are always included unless explicitly excluded'
        },
        'manager': {
            'class': 'FilterManager',
            'description': 'Combine multiple filters with AND/OR logic',
            'supports_sequential': True,
            'supports_presets': True,
            'supports_analysis': True
        }
    }

# Quick access to common filter combinations
COMMON_FILTER_PRESETS = {
    'entry_level': {
        'name': 'Entry Level Jobs',
        'description': 'Jobs suitable for entry-level candidates',
        'config': {
            'education': {'criteria': 'Bachelor'},
            'work_type': {'criteria': ['Full-time', 'Internship']},
            'gender': {'criteria': 'both'}
        }
    },
    'senior_level': {
        'name': 'Senior Level Jobs',
        'description': 'Jobs requiring advanced qualifications',
        'config': {
            'education': {'criteria': ['Master', 'PhD']},
            'work_type': {'criteria': ['Full-time']},
            'gender': {'criteria': 'both'}
        }
    },
    'flexible_work': {
        'name': 'Flexible Work Arrangements',
        'description': 'Jobs with flexible work arrangements',
        'config': {
            'work_type': {'criteria': ['Part-time', 'Contract']},
            'gender': {'criteria': 'both'}
        }
    },
    'all_work_types': {
        'name': 'All Work Types',
        'description': 'Jobs across all supported work types',
        'config': {
            'work_type': {'criteria': ['Full-time', 'Part-time', 'Contract', 'Internship']},
            'gender': {'criteria': 'both'}
        }
    }
}

def apply_preset_filter(df, preset_name, logic_operator='AND'):
    """
    Apply a predefined filter preset to the dataset.
    
    Args:
        df (pd.DataFrame): The job dataset
        preset_name (str): Name of the preset to apply
        logic_operator (str): Logic operator for combining filters
        
    Returns:
        pd.DataFrame: Filtered dataset
        
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in COMMON_FILTER_PRESETS:
        available_presets = list(COMMON_FILTER_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    manager = FilterManager()
    preset = COMMON_FILTER_PRESETS[preset_name]
    return manager.apply_combined_filter(df, preset['config'], logic_operator)

def list_available_presets():
    """
    List all available filter presets.
    
    Returns:
        dict: Information about available presets
    """
    return {name: {'name': preset['name'], 'description': preset['description']} 
            for name, preset in COMMON_FILTER_PRESETS.items()}