"""
Work Type Filter for JobHunt Recommendation System

This module provides filtering functionality based on work type categories.
It filters jobs based on the Work Type column in the dataset using exact matching.

Supported work types:
- Contract
- Part-Time  
- Intern
- Full-Time

The filter performs exact string matching without case sensitivity options
or partial matching. Input must be one of the 4 supported work types.


Version: 1.2.0
"""

from typing import List, Union, Dict, Any, Set
import pandas as pd
from .base_filter import BaseJobFilter, FilterValidationError, FilterApplicationError


class WorkTypeFilter(BaseJobFilter):
    """
    Filter jobs based on work type categories using exact matching.
    
    This filter supports exact matching for the 4 supported work types:
    Contract, Part-Time, Intern, and Full-Time.
    """
    
    # Supported work types for exact matching
    SUPPORTED_WORK_TYPES = [
        'Contract', 'Part-Time', 'Intern', 'Full-Time'
    ]
    
    # Flexible work types (subset of SUPPORTED_WORK_TYPES)
    FLEXIBLE_WORK_TYPES = [
        'Part-Time', 'Contract'
    ]
    
    def __init__(self):
        """Initialize the Work Type Filter."""
        super().__init__("WorkType")
        # Support both new and legacy column names
        self.filter_column = "Work_Type"
        self._filter_column_aliases = ["Work_Type", "Work Type"]

    def _resolve_filter_column(self, df: pd.DataFrame) -> str:
        """
        Resolve the actual work type column present in the DataFrame.
        Returns the first matching alias from _filter_column_aliases.
        Raises FilterApplicationError if none are found.
        """
        for col in self._filter_column_aliases:
            if col in df.columns:
                return col
        raise FilterApplicationError(
            f"Work type column not found. Expected one of {self._filter_column_aliases}. "
            f"Available columns: {list(df.columns)}"
        )
    
    def apply_filter(self, 
                    df: pd.DataFrame, 
                    filter_criteria: Union[str, List[str]]) -> pd.DataFrame:
        """
        Apply work type filter to the job dataset with exact matching.
        """
        try:
            # Resolve actual column present in the dataset
            filter_col = self._resolve_filter_column(df)
            # Validate inputs
            self.validate_dataframe(df, [filter_col])
            if not self.validate_criteria(filter_criteria):
                raise FilterValidationError(f"Invalid work type criteria: {filter_criteria}")
            
            # Normalize criteria to list
            if isinstance(filter_criteria, str):
                criteria_list = [filter_criteria]
            else:
                criteria_list = list(filter_criteria)
            
            original_count = len(df)
            
            # Apply exact matching filter
            filtered_df = self._apply_exact_match_filter(df, criteria_list, filter_col)
            
            filtered_count = len(filtered_df)
            self.log_filter_results(original_count, filtered_count, filter_criteria)
            
            return filtered_df
            
        except Exception as e:
            if isinstance(e, (FilterValidationError, FilterApplicationError)):
                raise
            raise FilterApplicationError(f"Failed to apply work type filter: {str(e)}")
    
    def _apply_exact_match_filter(self, 
                                 df: pd.DataFrame, 
                                 criteria_list: List[str],
                                 filter_column: str) -> pd.DataFrame:
        """
        Apply exact match filtering for work types using canonicalized values.
        """
        # Canonicalize dataset values to supported categories
        normalized = df[filter_column].apply(self._canonicalize_work_type)
        mask = normalized.isin(criteria_list)
        return df[mask]
    
    def _canonicalize_work_type(self, value: Any) -> str:
        """Map various dataset strings to canonical supported categories."""
        if pd.isna(value):
            return ''
        s = str(value).strip().lower()
        mapping = {
            # Full-Time variations
            'full-time': 'Full-Time',
            'full time': 'Full-Time',
            'fulltime': 'Full-Time',
            # Part-Time variations
            'part-time': 'Part-Time',
            'part time': 'Part-Time',
            'parttime': 'Part-Time',
            # Intern variations
            'intern': 'Intern',
            'internship': 'Intern',
            'trainee': 'Intern',
            'apprentice': 'Intern',
            # Contract variations
            'contract': 'Contract',
            'contractor': 'Contract',
            'temporary': 'Contract',
            'temp': 'Contract',
            'freelance': 'Contract',
            'consulting': 'Contract',
        }
        return mapping.get(s, str(value).strip())
    
    
    def validate_criteria(self, filter_criteria: Union[str, List[str]]) -> bool:
        """
        Validate the work type filter criteria.
        
        Args:
            filter_criteria (Union[str, List[str]]): The criteria to validate
            
        Returns:
            bool: True if criteria is valid, False otherwise
        """
        if not filter_criteria:
            return False
        
        # Convert to list for uniform processing
        if isinstance(filter_criteria, str):
            criteria_list = [filter_criteria]
        elif isinstance(filter_criteria, list):
            criteria_list = filter_criteria
        else:
            return False
        
        # Check if all criteria are non-empty strings and in SUPPORTED_WORK_TYPES
        for criteria in criteria_list:
            if not isinstance(criteria, str) or not criteria.strip():
                return False
            if criteria not in self.SUPPORTED_WORK_TYPES:
                return False
        
        return True
    
    def get_available_options(self, df: pd.DataFrame) -> List[str]:
        """
        Get available work type options from the dataset (canonicalized).
        """
        try:
            filter_col = self._resolve_filter_column(df)
            self.validate_dataframe(df, [filter_col])
            normalized = df[filter_col].dropna().apply(self._canonicalize_work_type)
            unique_work_types = [wt for wt in normalized.unique().tolist() if wt in self.SUPPORTED_WORK_TYPES]
            return sorted(unique_work_types) if unique_work_types else self.SUPPORTED_WORK_TYPES
        except Exception as e:
            self.logger.warning(f"Could not get available options: {str(e)}")
            return self.SUPPORTED_WORK_TYPES
    
    def get_work_type_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed statistics about work type distribution in the dataset.
        """
        try:
            filter_col = self._resolve_filter_column(df)
            self.validate_dataframe(df, [filter_col])
            
            work_type_counts = df[filter_col].value_counts()
            total_jobs = len(df)
            
            stats = {
                "total_jobs": total_jobs,
                "work_type_distribution": {},
                "missing_work_type_info": df[filter_col].isna().sum(),
                "unique_work_types": len(work_type_counts)
            }
            
            for work_type, count in work_type_counts.items():
                percentage = (count / total_jobs) * 100
                stats["work_type_distribution"][work_type] = {
                    "count": int(count),
                    "percentage": round(percentage, 2)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get work type statistics: {str(e)}")
            return {"error": str(e)}
    
    def filter_full_time_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to filter only full-time jobs.
        """
        return self.apply_filter(df, "Full-Time")
    

    
    def filter_flexible_jobs(self, df: pd.DataFrame, 
                           flexible_types: List[str] = None) -> pd.DataFrame:
        """
        Convenience method to filter flexible work arrangements.
        
        Args:
            df (pd.DataFrame): The job dataset
            flexible_types (List[str], optional): Custom list of flexible work types.
                Defaults to FLEXIBLE_WORK_TYPES if not provided.
            
        Returns:
            pd.DataFrame: Jobs with flexible work arrangements
        """
        if flexible_types is None:
            flexible_types = self.FLEXIBLE_WORK_TYPES
        
        # Validate that all flexible types are in SUPPORTED_WORK_TYPES
        invalid_types = [wt for wt in flexible_types if wt not in self.SUPPORTED_WORK_TYPES]
        if invalid_types:
            raise FilterValidationError(f"Invalid flexible work types: {invalid_types}. "
                                      f"Must be from: {self.SUPPORTED_WORK_TYPES}")
        
        return self.apply_filter(df, flexible_types)
    

    



# Convenience functions for easy usage
def filter_by_work_type(df: pd.DataFrame, 
                       work_types: Union[str, List[str]]) -> pd.DataFrame:
    """
    Convenience function to filter jobs by work type.
    
    Args:
        df (pd.DataFrame): The job dataset
        work_types (Union[str, List[str]]): Work type(s) to filter by
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filter_instance = WorkTypeFilter()
    return filter_instance.apply_filter(df, work_types)


def get_work_type_options(df: pd.DataFrame) -> List[str]:
    """
    Convenience function to get available work type options.
    
    Args:
        df (pd.DataFrame): The job dataset
        
    Returns:
        List[str]: Available work types
    """
    filter_instance = WorkTypeFilter()
    return filter_instance.get_available_options(df)