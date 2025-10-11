"""Experience Range Processing Module

This module provides functions to parse experience ranges from text and convert them
into structured numerical features:
- Minimum experience
- Maximum experience
- Mid-point experience

Handles various text formats like "1 to 9 Years", "5+ years", "Entry Level", etc.

Functions:
- parse_experience_range: Parse a single experience string
- process_experience: Process a pandas Series of experience strings
"""

import pandas as pd
import numpy as np
import re
from typing import Union, Optional, Tuple


def parse_experience_range(experience: Union[str, None]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse an experience string into minimum, maximum, and mid-point values.
    
    Handles various formats:
    - "1 to 9 Years" → (1.0, 9.0, 5.0)
    - "5+ years" → (5.0, None, 5.0)
    - "Entry Level" → (0.0, 2.0, 1.0)
    - "Senior" → (5.0, None, 5.0)
    - "3-5 years" → (3.0, 5.0, 4.0)
    
    Args:
        experience (str or None): The experience string to parse
        
    Returns:
        Tuple[float, float, float]: (min_exp, max_exp, mid_exp)
        Returns (None, None, None) if parsing fails
        
    Examples:
        >>> parse_experience_range("1 to 9 Years")
        (1.0, 9.0, 5.0)
        >>> parse_experience_range("5+ years")
        (5.0, None, 5.0)
        >>> parse_experience_range("Entry Level")
        (0.0, 2.0, 1.0)
    """
    if pd.isna(experience) or experience is None:
        return None, None, None
    
    # Convert to lowercase and clean
    exp_clean = str(experience).lower().strip()
    
    # Handle special cases first
    special_cases = {
        'entry level': (0.0, 2.0, 1.0),
        'entry-level': (0.0, 2.0, 1.0),
        'fresher': (0.0, 1.0, 0.5),
        'fresh graduate': (0.0, 1.0, 0.5),
        'no experience': (0.0, 0.0, 0.0),
        'junior': (1.0, 3.0, 2.0),
        'mid-level': (3.0, 7.0, 5.0),
        'senior': (5.0, 15.0, 10.0),
        'lead': (7.0, 15.0, 11.0),
        'manager': (8.0, 20.0, 14.0),
        'director': (10.0, 25.0, 17.5),
        'executive': (15.0, 30.0, 22.5)
    }
    
    for keyword, values in special_cases.items():
        if keyword in exp_clean:
            return values
    
    # Extract numbers using regex
    numbers = re.findall(r'\d+(?:\.\d+)?', exp_clean)
    
    if not numbers:
        return None, None, None
    
    # Convert to floats
    nums = [float(n) for n in numbers]
    
    # Pattern matching for different formats
    if 'to' in exp_clean or '-' in exp_clean:
        # Range format: "1 to 9", "3-5"
        if len(nums) >= 2:
            min_exp = min(nums[0], nums[1])
            max_exp = max(nums[0], nums[1])
            mid_exp = (min_exp + max_exp) / 2
            return min_exp, max_exp, mid_exp
    
    elif '+' in exp_clean or 'plus' in exp_clean or 'above' in exp_clean or 'more' in exp_clean:
        # Plus format: "5+", "5 plus", "5 or more"
        min_exp = nums[0]
        max_exp = None  # Open-ended
        mid_exp = min_exp  # Use minimum as mid-point for open ranges
        return min_exp, max_exp, mid_exp
    
    elif 'under' in exp_clean or 'less than' in exp_clean or 'below' in exp_clean:
        # Under format: "under 3", "less than 5"
        min_exp = 0.0
        max_exp = nums[0]
        mid_exp = max_exp / 2
        return min_exp, max_exp, mid_exp
    
    elif 'around' in exp_clean or 'approximately' in exp_clean or 'about' in exp_clean:
        # Approximate format: "around 5", "about 3 years"
        target = nums[0]
        min_exp = max(0, target - 1)
        max_exp = target + 1
        mid_exp = target
        return min_exp, max_exp, mid_exp
    
    else:
        # Single number: assume it's exact or a minimum
        exp_val = nums[0]
        if 'year' in exp_clean or 'yr' in exp_clean:
            # Treat as exact
            return exp_val, exp_val, exp_val
        else:
            # Treat as minimum
            return exp_val, None, exp_val
    
    return None, None, None


def process_experience(experience_series: pd.Series) -> pd.DataFrame:
    """
    Process a pandas Series of experience strings and return structured features.
    
    Args:
        experience_series (pd.Series): Series containing experience strings
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - experience_min: Minimum experience in years
            - experience_max: Maximum experience in years (None for open ranges)
            - experience_mid: Mid-point experience in years
            - has_experience_range: Binary indicator for valid experience range
            - is_entry_level: Binary indicator for entry level (0-2 years)
            - is_mid_level: Binary indicator for mid level (3-7 years)
            - is_senior_level: Binary indicator for senior level (8+ years)
            
    Examples:
        >>> exp_series = pd.Series(["1 to 9 Years", "5+ years", "Entry Level"])
        >>> result = process_experience(exp_series)
        >>> print(result.columns.tolist())
        ['experience_min', 'experience_max', 'experience_mid', 'has_experience_range', 'is_entry_level', 'is_mid_level', 'is_senior_level']
    """
    # Parse all experience strings
    parsed_data = experience_series.apply(lambda x: parse_experience_range(x))
    
    # Extract components
    min_exp = parsed_data.apply(lambda x: x[0] if x[0] is not None else np.nan)
    max_exp = parsed_data.apply(lambda x: x[1] if x[1] is not None else np.nan)
    mid_exp = parsed_data.apply(lambda x: x[2] if x[2] is not None else np.nan)
    
    # Create structured output
    result = pd.DataFrame({
        'experience_min': min_exp,
        'experience_max': max_exp,
        'experience_mid': mid_exp,
        'has_experience_range': (~min_exp.isna()).astype(int),
        'is_entry_level': ((min_exp >= 0) & (min_exp <= 2)).astype(int),
        'is_mid_level': ((min_exp >= 3) & (min_exp <= 7)).astype(int),
        'is_senior_level': (min_exp >= 8).astype(int)
    })
    
    return result


def get_experience_stats(experience_series: pd.Series) -> dict:
    """
    Get statistics about experience distribution.
    
    Args:
        experience_series (pd.Series): Series containing experience strings
        
    Returns:
        dict: Dictionary with experience statistics
    """
    processed = process_experience(experience_series)
    
    stats = {
        'total_records': len(experience_series),
        'has_experience': processed['has_experience_range'].sum(),
        'entry_level_count': processed['is_entry_level'].sum(),
        'mid_level_count': processed['is_mid_level'].sum(),
        'senior_level_count': processed['is_senior_level'].sum(),
        'missing_count': experience_series.isna().sum(),
        'min_experience_avg': processed['experience_min'].mean(),
        'max_experience_avg': processed['experience_max'].mean(),
        'mid_experience_avg': processed['experience_mid'].mean(),
        'min_experience_median': processed['experience_min'].median(),
        'max_experience_median': processed['experience_max'].median(),
        'mid_experience_median': processed['experience_mid'].median()
    }
    
    # Add percentages
    total = stats['total_records']
    for key in ['has_experience', 'entry_level_count', 'mid_level_count', 'senior_level_count', 'missing_count']:
        stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
    
    return stats

