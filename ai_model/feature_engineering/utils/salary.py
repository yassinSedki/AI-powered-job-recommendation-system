"""Salary Range Processing Module

This module provides functions to parse salary ranges from text or numbers and convert them
into structured numerical features:
- Minimum salary
- Maximum salary
- Mid-point salary

Handles various formats like "$50,000 - $80,000", "50K-80K", "₹5,00,000 per annum", etc.

Functions:
- parse_salary_range: Parse a single salary string
- process_salary: Process a pandas Series of salary strings
"""

import pandas as pd
import numpy as np
import re
from typing import Union, Optional, Tuple


def parse_salary_range(salary: Union[str, int, float, None]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse a salary string or number into minimum, maximum, and mid-point values.
    
    Handles various formats:
    - "$50,000 - $80,000" → (50000.0, 80000.0, 65000.0)
    - "50K-80K" → (50000.0, 80000.0, 65000.0)
    - "₹5,00,000 per annum" → (500000.0, 500000.0, 500000.0)
    - "$60K+" → (60000.0, None, 60000.0)
    - 75000 → (75000.0, 75000.0, 75000.0)
    
    Args:
        salary (str, int, float, or None): The salary to parse
        
    Returns:
        Tuple[float, float, float]: (min_salary, max_salary, mid_salary)
        Returns (None, None, None) if parsing fails
        
    Examples:
        >>> parse_salary_range("$50,000 - $80,000")
        (50000.0, 80000.0, 65000.0)
        >>> parse_salary_range("60K+")
        (60000.0, None, 60000.0)
        >>> parse_salary_range(75000)
        (75000.0, 75000.0, 75000.0)
    """
    if pd.isna(salary) or salary is None:
        return None, None, None
    
    # Handle numeric input
    if isinstance(salary, (int, float)):
        sal_val = float(salary)
        return sal_val, sal_val, sal_val
    
    # Convert to string and clean
    salary_clean = str(salary).lower().strip()
    
    # Remove common words that don't affect parsing
    remove_words = ['per', 'annum', 'year', 'yearly', 'annual', 'month', 'monthly', 'hour', 'hourly', 'salary', 'package']
    for word in remove_words:
        salary_clean = re.sub(rf'\b{word}\b', '', salary_clean)
    
    # Extract all numbers (including decimals)
    # This regex captures numbers with commas, decimals, and K/M suffixes
    number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?(?:[km])?'
    numbers = re.findall(number_pattern, salary_clean)
    
    if not numbers:
        return None, None, None
    
    # Convert numbers to actual values
    def convert_number(num_str):
        # Remove commas
        num_str = num_str.replace(',', '')
        
        # Handle K and M suffixes
        if num_str.endswith('k'):
            return float(num_str[:-1]) * 1000
        elif num_str.endswith('m'):
            return float(num_str[:-1]) * 1000000
        else:
            return float(num_str)
    
    try:
        nums = [convert_number(n) for n in numbers]
    except ValueError:
        return None, None, None
    
    # Handle different patterns
    if len(nums) == 0:
        return None, None, None
    
    elif len(nums) == 1:
        # Single number
        sal_val = nums[0]
        
        # Check for plus sign or similar indicators
        if '+' in salary_clean or 'plus' in salary_clean or 'above' in salary_clean or 'more' in salary_clean:
            # Open-ended range
            return sal_val, None, sal_val
        elif 'under' in salary_clean or 'below' in salary_clean or 'up to' in salary_clean or 'maximum' in salary_clean:
            # Upper limit
            return 0.0, sal_val, sal_val / 2
        else:
            # Exact salary
            return sal_val, sal_val, sal_val
    
    elif len(nums) >= 2:
        # Range format
        min_sal = min(nums[0], nums[1])
        max_sal = max(nums[0], nums[1])
        mid_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, mid_sal
    
    return None, None, None


def process_salary(salary_series: pd.Series) -> pd.DataFrame:
    """
    Process a pandas Series of salary data and return structured features.
    
    Args:
        salary_series (pd.Series): Series containing salary strings or numbers
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - salary_min: Minimum salary
            - salary_max: Maximum salary (None for open ranges)
            - salary_mid: Mid-point salary
            - has_salary_range: Binary indicator for valid salary range
            - salary_range_width: Width of salary range (max - min)
            - is_high_salary: Binary indicator for high salary (top 25%)
            - is_low_salary: Binary indicator for low salary (bottom 25%)
            
    Examples:
        >>> sal_series = pd.Series(["$50,000 - $80,000", "60K+", 75000])
        >>> result = process_salary(sal_series)
        >>> print(result.columns.tolist())
        ['salary_min', 'salary_max', 'salary_mid', 'has_salary_range', 'salary_range_width', 'is_high_salary', 'is_low_salary']
    """
    # Parse all salary strings
    parsed_data = salary_series.apply(lambda x: parse_salary_range(x))
    
    # Extract components
    min_sal = parsed_data.apply(lambda x: x[0] if x[0] is not None else np.nan)
    max_sal = parsed_data.apply(lambda x: x[1] if x[1] is not None else np.nan)
    mid_sal = parsed_data.apply(lambda x: x[2] if x[2] is not None else np.nan)
    
    # Calculate range width
    range_width = max_sal - min_sal
    range_width = range_width.fillna(0)  # Fill NaN with 0 for open ranges
    
    # Calculate percentiles for high/low salary classification
    mid_sal_valid = mid_sal.dropna()
    if len(mid_sal_valid) > 0:
        high_threshold = mid_sal_valid.quantile(0.75)
        low_threshold = mid_sal_valid.quantile(0.25)
    else:
        high_threshold = np.inf
        low_threshold = 0
    
    # Create structured output
    result = pd.DataFrame({
        'salary_min': min_sal,
        'salary_max': max_sal,
        'salary_mid': mid_sal,
        'has_salary_range': (~min_sal.isna()).astype(int),
        'salary_range_width': range_width,
        'is_high_salary': (mid_sal >= high_threshold).astype(int),
        'is_low_salary': (mid_sal <= low_threshold).astype(int)
    })
    
    return result


def get_salary_stats(salary_series: pd.Series) -> dict:
    """
    Get statistics about salary distribution.
    
    Args:
        salary_series (pd.Series): Series containing salary data
        
    Returns:
        dict: Dictionary with salary statistics
    """
    processed = process_salary(salary_series)
    
    stats = {
        'total_records': len(salary_series),
        'has_salary': processed['has_salary_range'].sum(),
        'missing_count': salary_series.isna().sum(),
        'min_salary_avg': processed['salary_min'].mean(),
        'max_salary_avg': processed['salary_max'].mean(),
        'mid_salary_avg': processed['salary_mid'].mean(),
        'min_salary_median': processed['salary_min'].median(),
        'max_salary_median': processed['salary_max'].median(),
        'mid_salary_median': processed['salary_mid'].median(),
        'salary_range_width_avg': processed['salary_range_width'].mean(),
        'high_salary_count': processed['is_high_salary'].sum(),
        'low_salary_count': processed['is_low_salary'].sum()
    }
    
    # Add percentiles
    mid_sal_valid = processed['salary_mid'].dropna()
    if len(mid_sal_valid) > 0:
        stats.update({
            'salary_p25': mid_sal_valid.quantile(0.25),
            'salary_p50': mid_sal_valid.quantile(0.50),
            'salary_p75': mid_sal_valid.quantile(0.75),
            'salary_p90': mid_sal_valid.quantile(0.90)
        })
    
    # Add percentages
    total = stats['total_records']
    for key in ['has_salary', 'missing_count', 'high_salary_count', 'low_salary_count']:
        stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
    
    return stats
