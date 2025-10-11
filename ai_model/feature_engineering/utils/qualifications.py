"""Qualification Classification Module

This module provides functions to classify job qualifications into standardized categories:
- Bachelor: Any qualification starting with 'B'
- Master: Any qualification starting with 'M'
- PhD: Any qualification starting with 'P'

Functions:
- classify_qualification: Classify a single qualification string
- process_qualifications: Process a pandas Series of qualifications
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def classify_qualification(qualification: Union[str, None]) -> str:
    """
    Classify a qualification based on first letter.
    
    Args:
        qualification (str or None): The qualification string to classify
        
    Returns:
        str: 'Bachelor' if starts with 'B', 'Master' if starts with 'M', 'PhD' if starts with 'P', else 'Other'
    """
    if not qualification or pd.isna(qualification):
        return 'Other'
    
    first_letter = str(qualification).strip().upper()[0] if str(qualification).strip() else ''
    
    if first_letter == 'B':
        return 'Bachelor'
    elif first_letter == 'M':
        return 'Master'
    elif first_letter == 'P':
        return 'PhD'
    else:
        return 'Other'


def process_qualifications(qualifications: pd.Series) -> pd.DataFrame:
    """
    Process a pandas Series of qualifications and return structured features.
    
    Args:
        qualifications (pd.Series): Series containing qualification strings
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - qualification_category: Classified category (Bachelor/Master/PhD/Other)
            - is_bachelor: Binary indicator for Bachelor's degree (includes Master and PhD)
            - is_master: Binary indicator for Master's degree (includes PhD)
            - is_phd: Binary indicator for PhD
            - has_qualification: Binary indicator for any qualification
            
    Note:
        Hierarchical logic applied:
        - PhD holders are considered to have Master's and Bachelor's degrees
        - Master's holders are considered to have Bachelor's degrees
            
    Examples:
        >>> quals = pd.Series(["Bachelor of Science", "Master of Arts", "PhD in Computer Science"])
        >>> result = process_qualifications(quals)
        >>> print(result.columns.tolist())
        ['qualification_category', 'is_bachelor', 'is_master', 'is_phd', 'has_qualification']
    """
    # Apply classification
    categories = qualifications.apply(classify_qualification)
    
    # Create hierarchical binary indicators
    is_phd = (categories == 'PhD').astype(int)
    is_master = ((categories == 'Master') | (categories == 'PhD')).astype(int)
    is_bachelor = ((categories == 'Bachelor') | (categories == 'Master') | (categories == 'PhD')).astype(int)
    
    # Create structured output
    result = pd.DataFrame({
        'qualification_category': categories,
        'is_bachelor': is_bachelor,
        'is_master': is_master,
        'is_phd': is_phd,
        'has_qualification': (categories != 'Other').astype(int)
    })
    
    return result


def get_qualification_stats(qualifications: pd.Series) -> dict:
    """
    Get statistics about qualification distribution.
    
    Args:
        qualifications (pd.Series): Series containing qualification strings
        
    Returns:
        dict: Dictionary with qualification statistics
    """
    processed = process_qualifications(qualifications)
    
    stats = {
        'total_records': len(qualifications),
        'has_qualification': processed['has_qualification'].sum(),
        'bachelor_count': processed['is_bachelor'].sum(),
        'master_count': processed['is_master'].sum(),
        'phd_count': processed['is_phd'].sum(),
        'other_count': (processed['qualification_category'] == 'Other').sum(),
        'missing_count': qualifications.isna().sum()
    }
    
    # Add percentages
    total = stats['total_records']
    for key in ['has_qualification', 'bachelor_count', 'master_count', 'phd_count', 'other_count', 'missing_count']:
        stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
    
    return stats
