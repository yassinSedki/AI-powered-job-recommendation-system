"""
Base Filter Interface for JobHunt Recommendation System

This module provides the abstract base class for all job filters, ensuring
consistent interface and behavior across different filter implementations.


Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict, Optional
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseJobFilter(ABC):
    """
    Abstract base class for all job filters.
    
    This class defines the interface that all job filters must implement,
    ensuring consistency and maintainability across the filtering system.
    """
    
    def __init__(self, filter_name: str):
        """
        Initialize the base filter.
        
        Args:
            filter_name (str): Name of the filter for logging and identification
        """
        self.filter_name = filter_name
        self.logger = logging.getLogger(f"{__name__}.{filter_name}")
        
    @abstractmethod
    def apply_filter(self, 
                    df: pd.DataFrame, 
                    filter_criteria: Union[str, List[str], Dict[str, Any]],
                    **kwargs) -> pd.DataFrame:
        """
        Apply the filter to the job dataset.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_criteria: The criteria to filter by (format depends on filter type)
            **kwargs: Additional filter-specific parameters
            
        Returns:
            pd.DataFrame: Filtered dataset
            
        Raises:
            ValueError: If filter criteria is invalid
            KeyError: If required columns are missing from dataset
        """
        pass
    
    @abstractmethod
    def validate_criteria(self, filter_criteria: Any) -> bool:
        """
        Validate the filter criteria.
        
        Args:
            filter_criteria: The criteria to validate
            
        Returns:
            bool: True if criteria is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_options(self, df: pd.DataFrame) -> List[str]:
        """
        Get available filter options from the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            List[str]: List of available filter options
        """
        pass
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate that the dataframe contains required columns.
        
        Args:
            df (pd.DataFrame): The dataframe to validate
            required_columns (List[str]): List of required column names
            
        Raises:
            ValueError: If dataframe is empty or None
            KeyError: If required columns are missing
        """
        if df is None or df.empty:
            raise ValueError(f"{self.filter_name}: Dataset cannot be empty or None")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"{self.filter_name}: Missing required columns: {missing_columns}")
    
    def log_filter_results(self, 
                          original_count: int, 
                          filtered_count: int, 
                          filter_criteria: Any) -> None:
        """
        Log the results of the filtering operation.
        
        Args:
            original_count (int): Number of jobs before filtering
            filtered_count (int): Number of jobs after filtering
            filter_criteria: The criteria used for filtering
        """
        percentage_remaining = (filtered_count / original_count * 100) if original_count > 0 else 0
        
        self.logger.info(
            f"{self.filter_name} Filter Applied:\n"
            f"  Criteria: {filter_criteria}\n"
            f"  Original jobs: {original_count:,}\n"
            f"  Filtered jobs: {filtered_count:,}\n"
            f"  Remaining: {percentage_remaining:.1f}%"
        )
    
    def get_filter_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the filter's impact on the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            Dict[str, Any]: Summary information about the filter
        """
        return {
            "filter_name": self.filter_name,
            "total_jobs": len(df),
            "available_options": self.get_available_options(df),
            "filter_column": getattr(self, 'filter_column', 'Unknown')
        }


class FilterValidationError(Exception):
    """Custom exception for filter validation errors."""
    pass


class FilterApplicationError(Exception):
    """Custom exception for filter application errors."""
    pass