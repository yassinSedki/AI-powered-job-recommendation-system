"""Work Type Standardization Module

This module provides functions to standardize job work types into four categories:
- Contract
- Part-time
- Internship
- Full-time

Handles variations and inconsistencies in text data through pattern matching.

Functions:
- standardize_work_type: Standardize a single work type string
- process_work_types: Process a pandas Series of work types
"""

import pandas as pd
import numpy as np
import re
from typing import Union, Optional


def standardize_work_type(work_type: Union[str, None]) -> str:
    """
    Standardize a work type into one of four categories.
    
    Categories:
    - Contract: contract, contractor, freelance, temporary, temp, consulting
    - Part-time: part-time, part time, parttime, casual, hourly
    - Internship: intern, internship, trainee, apprentice, graduate program
    - Full-time: full-time, full time, fulltime, permanent, regular, staff
    
    Args:
        work_type (str or None): The work type string to standardize
        
    Returns:
        str: One of 'Contract', 'Part-time', 'Internship', 'Full-time', or 'Other'
        
    Examples:
        >>> standardize_work_type("Full Time")
        'Full-time'
        >>> standardize_work_type("Contract Position")
        'Contract'
        >>> standardize_work_type("Intern")
        'Internship'
        >>> standardize_work_type("Part-Time")
        'Part-time'
    """
    if pd.isna(work_type) or work_type is None:
        return 'Other'


def process_work_types(work_types: pd.Series) -> pd.DataFrame:
    """
    Process a pandas Series of work types and return structured features.
    
    Args:
        work_types (pd.Series): Series containing work type strings
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - work_type_category: Standardized category (Contract/Part-time/Internship/Full-time/Other)
            - is_contract: Binary indicator for Contract
            - is_part_time: Binary indicator for Part-time
            - is_internship: Binary indicator for Internship
            - is_full_time: Binary indicator for Full-time
            - has_work_type: Binary indicator for any valid work type
            
    Examples:
        >>> work_series = pd.Series(["Full Time", "Contract", "Intern"])
        >>> result = process_work_types(work_series)
        >>> print(result.columns.tolist())
        ['work_type_category', 'is_contract', 'is_part_time', 'is_internship', 'is_full_time', 'has_work_type']
    """
    # Apply standardization
    categories = work_types.apply(standardize_work_type)
    
    # Create structured output
    result = pd.DataFrame({
        'work_type_category': categories,
        'is_contract': (categories == 'Contract').astype(int),
        'is_part_time': (categories == 'Part-time').astype(int),
        'is_internship': (categories == 'Internship').astype(int),
        'is_full_time': (categories == 'Full-time').astype(int),
        'has_work_type': (categories != 'Other').astype(int)
    })
    
    return result


def get_work_type_stats(work_types: pd.Series) -> dict:
    """
    Get statistics about work type distribution.
    
    Args:
        work_types (pd.Series): Series containing work type strings
        
    Returns:
        dict: Dictionary with work type statistics
    """
    processed = process_work_types(work_types)
    
    stats = {
        'total_records': len(work_types),
        'has_work_type': processed['has_work_type'].sum(),
        'contract_count': processed['is_contract'].sum(),
        'part_time_count': processed['is_part_time'].sum(),
        'internship_count': processed['is_internship'].sum(),
        'full_time_count': processed['is_full_time'].sum(),
        'other_count': (processed['work_type_category'] == 'Other').sum(),
        'missing_count': work_types.isna().sum()
    }
    
    # Add percentages
    total = stats['total_records']
    for key in ['has_work_type', 'contract_count', 'part_time_count', 'internship_count', 'full_time_count', 'other_count', 'missing_count']:
        stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
    
    return stats
