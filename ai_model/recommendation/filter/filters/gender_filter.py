"""
Gender Filter for JobHunt Recommendation System

This module provides filtering functionality based on gender preferences.
It filters jobs based on the Preference column in the dataset with simple logic:
- User input is either "male" or "female"
- When filtering for "male": returns jobs marked as "male" AND jobs marked as "both"
- When filtering for "female": returns jobs marked as "female" AND jobs marked as "both"
- Jobs marked as "both" are automatically included for any gender filter

User input options:
- male: Returns male-specific jobs + gender-neutral jobs
- female: Returns female-specific jobs + gender-neutral jobs

Dataset values:
- both: Jobs open to all genders (automatically included)
- male: Jobs specifically for male candidates
- female: Jobs specifically for female candidates


Version: 1.2.0
"""

from typing import List, Union, Dict, Any, Optional
import pandas as pd
from .base_filter import BaseJobFilter, FilterValidationError, FilterApplicationError


class GenderFilter(BaseJobFilter):
    """
    Filter jobs based on gender preferences with automatic inclusion of gender-neutral jobs.
    
    This filter implements the business logic where:
    1. Users can only input "male" or "female" as filter criteria
    2. When filtering for "male": returns jobs marked as "male" + jobs marked as "both"
    3. When filtering for "female": returns jobs marked as "female" + jobs marked as "both"
    4. Jobs marked as "both" are automatically included for any gender filter
    """
    
    # Valid gender preferences
    VALID_GENDER_PREFERENCES = ['both', 'male', 'female']
    INCLUSIVE_PREFERENCE = 'both'
    
    def __init__(self):
        """Initialize the Gender Filter."""
        super().__init__("Gender")
        # Use 'Preference' column as identified in the dataset
        self.filter_column = "Preference"
    
    def apply_filter(self, 
                    df: pd.DataFrame, 
                    filter_criteria: Union[str, List[str]],
                    **kwargs) -> pd.DataFrame:
        """
        Apply gender filter to the job dataset with automatic inclusion of gender-neutral jobs.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_criteria (Union[str, List[str]]): Gender preference(s) to filter by
                Should be "male" or "female" (users don't filter for "both")
            **kwargs: Additional parameters:
                - case_sensitive (bool): Whether to match case sensitively (default: False)
        
        Returns:
            pd.DataFrame: Filtered dataset containing jobs matching gender criteria plus "both" jobs
            
        Raises:
            FilterValidationError: If filter criteria is invalid
            FilterApplicationError: If filtering operation fails
        """
        try:
            # Validate inputs
            self.validate_dataframe(df, [self.filter_column])
            if not self.validate_criteria(filter_criteria):
                raise FilterValidationError(f"Invalid gender criteria: {filter_criteria}")
            
            # Get filter parameters
            case_sensitive = kwargs.get('case_sensitive', False)
            
            # Normalize criteria to list
            if isinstance(filter_criteria, str):
                criteria_list = [filter_criteria]
            else:
                criteria_list = list(filter_criteria)
            
            original_count = len(df)
            
            # Apply gender filtering with simplified logic
            filtered_df = self._apply_gender_logic(df, criteria_list, case_sensitive)
            
            filtered_count = len(filtered_df)
            self.log_filter_results(original_count, filtered_count, filter_criteria)
            
            return filtered_df
            
        except Exception as e:
            if isinstance(e, (FilterValidationError, FilterApplicationError)):
                raise
            raise FilterApplicationError(f"Failed to apply gender filter: {str(e)}")
    
    def _apply_gender_logic(self, 
                           df: pd.DataFrame, 
                           criteria_list: List[str], 
                           case_sensitive: bool) -> pd.DataFrame:
        """
        Apply gender filtering logic with special handling for "both".
        
        Args:
            df (pd.DataFrame): The dataset
            criteria_list (List[str]): List of gender criteria
            case_sensitive (bool): Whether to match case sensitively
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        # Normalize case
        if case_sensitive:
            df_column = df[self.filter_column]
            search_criteria = criteria_list
            both_value = self.INCLUSIVE_PREFERENCE
        else:
            df_column = df[self.filter_column].str.lower()
            search_criteria = [criteria.lower() for criteria in criteria_list]
            both_value = self.INCLUSIVE_PREFERENCE.lower()
        
        # Simple logic: User input is always "male" or "female"
        # Accept jobs that match the specific gender AND jobs marked as "both"
        both_mask = df_column == both_value  # Always accept "both" jobs
        criteria_mask = df_column.isin(search_criteria)  # Accept specific gender
        mask = both_mask | criteria_mask
        
        return df[mask]
    
    def validate_criteria(self, filter_criteria: Union[str, List[str]]) -> bool:
        """
        Validate the gender filter criteria.
        
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
        
        # Check if all criteria are valid gender preferences
        normalized_criteria = [criteria.lower() for criteria in criteria_list]
        return all(criteria in self.VALID_GENDER_PREFERENCES for criteria in normalized_criteria)
    
    def get_available_options(self, df: pd.DataFrame) -> List[str]:
        """
        Get available gender preference options from the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            List[str]: List of available gender preferences
        """
        try:
            self.validate_dataframe(df, [self.filter_column])
            unique_preferences = df[self.filter_column].dropna().unique().tolist()
            # Remove empty strings and sort
            unique_preferences = [pref for pref in unique_preferences if str(pref).strip()]
            return sorted(unique_preferences)
        except Exception as e:
            self.logger.warning(f"Could not get available options: {str(e)}")
            return self.VALID_GENDER_PREFERENCES
    
    def get_gender_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed statistics about gender preference distribution in the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            Dict[str, Any]: Statistics about gender preference distribution
        """
        try:
            self.validate_dataframe(df, [self.filter_column])
            
            gender_counts = df[self.filter_column].value_counts()
            total_jobs = len(df)
            
            stats = {
                "total_jobs": total_jobs,
                "gender_distribution": {},
                "missing_gender_info": df[self.filter_column].isna().sum(),
                "inclusive_jobs_count": 0,
                "inclusive_jobs_percentage": 0.0
            }
            
            for gender, count in gender_counts.items():
                percentage = (count / total_jobs) * 100
                stats["gender_distribution"][gender] = {
                    "count": int(count),
                    "percentage": round(percentage, 2)
                }
                
                # Track inclusive jobs (marked as "both")
                if str(gender).lower() == self.INCLUSIVE_PREFERENCE:
                    stats["inclusive_jobs_count"] = int(count)
                    stats["inclusive_jobs_percentage"] = round(percentage, 2)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get gender statistics: {str(e)}")
            return {"error": str(e)}
    
    def filter_for_male_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter jobs suitable for male candidates.
        Includes both jobs specifically for males and jobs open to all genders ("both").
        
        Args:
            df (pd.DataFrame): The job dataset
            
        Returns:
            pd.DataFrame: Jobs suitable for male candidates
        """
        return self.apply_filter(df, "male")
    
    def filter_for_female_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter jobs suitable for female candidates.
        Includes both jobs specifically for females and jobs open to all genders ("both").
        
        Args:
            df (pd.DataFrame): The job dataset
            
        Returns:
            pd.DataFrame: Jobs suitable for female candidates
        """
        return self.apply_filter(df, "female")
    
    def filter_inclusive_jobs_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter only jobs that are open to all genders (marked as "both").
        
        Args:
            df (pd.DataFrame): The job dataset
            
        Returns:
            pd.DataFrame: Jobs open to all genders
        """
        return self.apply_filter(df, "both")
    
    def filter_gender_specific_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter jobs that specify a particular gender (exclude "both" jobs).
        
        Args:
            df (pd.DataFrame): The job dataset
            
        Returns:
            pd.DataFrame: Jobs with specific gender requirements
        """
        # Directly filter for male and female jobs, excluding "both"
        self.validate_dataframe(df, [self.filter_column])
        mask = df[self.filter_column].str.lower().isin(['male', 'female'])
        return df[mask]
    
    def get_jobs_by_gender_preference(self, 
                                     df: pd.DataFrame, 
                                     candidate_gender: str) -> pd.DataFrame:
        """
        Get jobs based on candidate's gender.
        Includes both jobs specifically for the candidate's gender and jobs open to all genders ("both").
        
        Args:
            df (pd.DataFrame): The job dataset
            candidate_gender (str): The candidate's gender ("male" or "female")
            
        Returns:
            pd.DataFrame: Jobs suitable for the candidate
        """
        if candidate_gender.lower() not in ['male', 'female']:
            raise FilterValidationError(f"Invalid candidate gender: {candidate_gender}")
        
        return self.apply_filter(df, candidate_gender)
    
    def analyze_gender_inclusivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the gender inclusivity of the job dataset.
        
        Args:
            df (pd.DataFrame): The job dataset
            
        Returns:
            Dict[str, Any]: Analysis of gender inclusivity
        """
        stats = self.get_gender_statistics(df)
        
        if "error" in stats:
            return stats
        
        total_jobs = stats["total_jobs"]
        inclusive_count = stats["inclusive_jobs_count"]
        
        analysis = {
            "total_jobs_analyzed": total_jobs,
            "inclusive_jobs": inclusive_count,
            "gender_specific_jobs": total_jobs - inclusive_count,
            "inclusivity_score": round((inclusive_count / total_jobs) * 100, 2) if total_jobs > 0 else 0,
            "recommendations": []
        }
        
        # Add recommendations based on inclusivity score
        inclusivity_score = analysis["inclusivity_score"]
        if inclusivity_score >= 80:
            analysis["recommendations"].append("Excellent gender inclusivity in job postings")
        elif inclusivity_score >= 60:
            analysis["recommendations"].append("Good gender inclusivity, consider increasing 'both' gender jobs")
        elif inclusivity_score >= 40:
            analysis["recommendations"].append("Moderate inclusivity, significant room for improvement")
        else:
            analysis["recommendations"].append("Low inclusivity, consider reviewing gender requirements")
        
        return analysis


# Convenience functions for easy usage
def filter_by_gender(df: pd.DataFrame, 
                    gender: Union[str, List[str]], 
                    **kwargs) -> pd.DataFrame:
    """
    Convenience function to filter jobs by gender preference.
    
    Args:
        df (pd.DataFrame): The job dataset
        gender (Union[str, List[str]]): Gender preference(s) to filter by
        **kwargs: Additional filter parameters
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filter_instance = GenderFilter()
    return filter_instance.apply_filter(df, gender, **kwargs)


def get_gender_options(df: pd.DataFrame) -> List[str]:
    """
    Convenience function to get available gender preference options.
    
    Args:
        df (pd.DataFrame): The job dataset
        
    Returns:
        List[str]: Available gender preferences
    """
    filter_instance = GenderFilter()
    return filter_instance.get_available_options(df)


def filter_inclusive_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to filter jobs open to all genders.
    
    Args:
        df (pd.DataFrame): The job dataset
        
    Returns:
        pd.DataFrame: Jobs open to all genders
    """
    filter_instance = GenderFilter()
    return filter_instance.filter_inclusive_jobs_only(df)