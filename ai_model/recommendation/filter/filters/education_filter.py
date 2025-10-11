"""
Education Filter for JobHunt Recommendation System

This module provides filtering functionality based on educational qualifications.
It filters jobs based on the qualification_category column in the dataset using exact matching.

Supported qualification categories:
- Bachelor: Bachelor's degree requirements
- Master: Master's degree requirements  
- PhD: Doctoral degree requirements

The filter works with exact matching - jobs are filtered to include only those
that have the same qualification level as specified in the input criteria.


Version: 1.1.1
"""

from typing import List, Union, Dict, Any
import pandas as pd
from .base_filter import BaseJobFilter, FilterValidationError, FilterApplicationError


class EducationFilter(BaseJobFilter):
    """
    Filter jobs based on educational qualification requirements using exact matching.
    
    This filter allows filtering jobs by their qualification_category,
    supporting a single education level only. Jobs are filtered to include
    only those that have the exact same qualification level as specified.
    """
    
    # Valid education levels in the dataset
    VALID_EDUCATION_LEVELS = ['Bachelor', 'Master', 'PhD']
    
    def __init__(self):
        """Initialize the Education Filter."""
        super().__init__("Education")
        self.filter_column = "qualification_category"
    
    def apply_filter(self, 
                    df: pd.DataFrame, 
                    filter_criteria: str,
                    **kwargs) -> pd.DataFrame:
        """
        Apply education filter to the job dataset.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_criteria (str): Single education level to filter by. Must be one of
                "Bachelor", "Master", or "PhD" (case-insensitive unless case_sensitive=True)
            **kwargs: Additional parameters:
                - case_sensitive (bool): Whether to match case sensitively (default: False)
        
        Returns:
            pd.DataFrame: Filtered dataset containing only jobs with exact matching education criteria
            
        Raises:
            FilterValidationError: If filter criteria is invalid or not a single string
            FilterApplicationError: If filtering operation fails
        """
        try:
            # Validate inputs
            self.validate_dataframe(df, [self.filter_column])
            if not isinstance(filter_criteria, str):
                raise FilterValidationError("Education criteria must be a single string.")
            if not self.validate_criteria(filter_criteria):
                raise FilterValidationError(f"Invalid education criteria: {filter_criteria}")
            
            # Get filter parameters
            case_sensitive = kwargs.get('case_sensitive', False)
            
            original_count = len(df)
            
            if case_sensitive:
                # Exact case-sensitive match
                criteria_value = filter_criteria.strip()
                df_column = df[self.filter_column].astype(str)
                filtered_df = df[df_column == criteria_value]
            else:
                # Case-insensitive match using casefold to preserve acronym semantics
                normalized_input = filter_criteria.strip().casefold()
                df_column_norm = df[self.filter_column].astype(str).str.strip().str.casefold()
                filtered_df = df[df_column_norm == normalized_input]
            
            filtered_count = len(filtered_df)
            self.log_filter_results(original_count, filtered_count, filter_criteria)
            
            return filtered_df
            
        except Exception as e:
            if isinstance(e, (FilterValidationError, FilterApplicationError)):
                raise
            raise FilterApplicationError(f"Failed to apply education filter: {str(e)}")
    

    
    def validate_criteria(self, filter_criteria: str) -> bool:
        """
        Validate the education filter criteria.
        
        Args:
            filter_criteria (str): The single criteria to validate
            
        Returns:
            bool: True if criteria is valid, False otherwise
        """
        if not filter_criteria or not isinstance(filter_criteria, str):
            return False
        
        # Case-insensitive validation using casefold; maps to canonical labels without breaking acronyms
        normalized_criteria = str(filter_criteria).strip().casefold()
        valid_map = {v.casefold(): v for v in self.VALID_EDUCATION_LEVELS}
        return normalized_criteria in valid_map
    
    def get_available_options(self, df: pd.DataFrame) -> List[str]:
        """
        Get available education options from the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            List[str]: List of available education levels
        """
        try:
            self.validate_dataframe(df, [self.filter_column])
            unique_qualifications = df[self.filter_column].dropna().unique().tolist()
            return sorted(unique_qualifications)
        except Exception as e:
            self.logger.warning(f"Could not get available options: {str(e)}")
            return self.VALID_EDUCATION_LEVELS
    
    def get_education_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed statistics about education distribution in the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            Dict[str, Any]: Statistics about education distribution
        """
        try:
            self.validate_dataframe(df, [self.filter_column])
            
            education_counts = df[self.filter_column].value_counts()
            total_jobs = len(df)
            
            stats = {
                "total_jobs": total_jobs,
                "education_distribution": {},
                "missing_education_info": df[self.filter_column].isna().sum()
            }
            
            for education, count in education_counts.items():
                percentage = (count / total_jobs) * 100
                stats["education_distribution"][education] = {
                    "count": int(count),
                    "percentage": round(percentage, 2)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get education statistics: {str(e)}")
            return {"error": str(e)}
    
    def filter_by_minimum_education(self, 
                                   df: pd.DataFrame, 
                                   minimum_education: str) -> pd.DataFrame:
        """
        Filter jobs requiring exactly the specified education level.
        Note: This method now works with exact matching only.
        
        Args:
            df (pd.DataFrame): The job dataset
            minimum_education (str): Education level required
            
        Returns:
            pd.DataFrame: Jobs requiring exactly the specified education level
        """
        return self.apply_filter(df, minimum_education)
    
    def filter_by_exact_education(self, 
                                 df: pd.DataFrame, 
                                 education_level: str) -> pd.DataFrame:
        """
        Filter jobs requiring exactly the specified education level.
        
        Args:
            df (pd.DataFrame): The job dataset
            education_level (str): Exact education level required
            
        Returns:
            pd.DataFrame: Jobs requiring exactly the specified education level
        """
        return self.apply_filter(df, education_level)


# Convenience functions for easy usage
def filter_by_education(df: pd.DataFrame, 
                       education: str, 
                       **kwargs) -> pd.DataFrame:
    """
    Convenience function to filter jobs by education level.
    
    Args:
        df (pd.DataFrame): The job dataset
        education (str): Single education level to filter by
        **kwargs: Additional filter parameters
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filter_instance = EducationFilter()
    return filter_instance.apply_filter(df, education, **kwargs)


def get_education_options(df: pd.DataFrame) -> List[str]:
    """
    Convenience function to get available education options.
    
    Args:
        df (pd.DataFrame): The job dataset
        
    Returns:
        List[str]: Available education levels
    """
    filter_instance = EducationFilter()
    return filter_instance.get_available_options(df)

    