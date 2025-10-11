"""
Filter Manager for JobHunt Recommendation System

This module provides a centralized manager for all job filters, offering
a unified interface to apply multiple filters and manage filter operations.


Version: 1.0.0
"""

from typing import Dict, List, Union, Any, Optional, Type
import pandas as pd
import logging
from .filters import (
    BaseJobFilter, 
    FilterValidationError, 
    FilterApplicationError,
    GenderFilter,
    WorkTypeFilter,
    EducationFilter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterManager:
    """
    Centralized manager for all job filters.
    
    This class provides a unified interface to apply multiple filters,
    manage filter instances, and coordinate filtering operations across
    different filter types.
    """
    
    def __init__(self):
        """Initialize the Filter Manager with all available filters."""
        self.logger = logging.getLogger(f"{__name__}.FilterManager")
        
        # Initialize all available filters
        self._filters: Dict[str, BaseJobFilter] = {
            'gender': GenderFilter(),
            'work_type': WorkTypeFilter(),
            'education': EducationFilter()
        }
        
        # Track filter application order for optimization
        self._filter_order = ['education', 'work_type', 'gender']
        
        self.logger.info(f"FilterManager initialized with {len(self._filters)} filters")
    
    def get_available_filters(self) -> List[str]:
        """
        Get list of available filter types.
        
        Returns:
            List[str]: List of available filter names
        """
        return list(self._filters.keys())
    
    def get_filter(self, filter_name: str) -> BaseJobFilter:
        """
        Get a specific filter instance.
        
        Args:
            filter_name (str): Name of the filter to retrieve
            
        Returns:
            BaseJobFilter: The requested filter instance
            
        Raises:
            FilterValidationError: If filter name is not available
        """
        if filter_name not in self._filters:
            available_filters = ', '.join(self.get_available_filters())
            raise FilterValidationError(f"Filter '{filter_name}' not available. "
                                      f"Available filters: {available_filters}")
        
        return self._filters[filter_name]
    
    def apply_single_filter(self, 
                           df: pd.DataFrame,
                           filter_name: str,
                           filter_criteria: Union[str, List[str], Dict[str, Any]],
                           **kwargs) -> pd.DataFrame:
        """
        Apply a single filter to the dataset.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_name (str): Name of the filter to apply
            filter_criteria: The criteria for filtering
            **kwargs: Additional filter-specific parameters
            
        Returns:
            pd.DataFrame: Filtered dataset
            
        Raises:
            FilterValidationError: If filter name or criteria is invalid
            FilterApplicationError: If filtering operation fails
        """
        try:
            filter_instance = self.get_filter(filter_name)
            
            original_count = len(df)
            filtered_df = filter_instance.apply_filter(df, filter_criteria, **kwargs)
            filtered_count = len(filtered_df)
            
            self.logger.info(f"Applied {filter_name} filter: {original_count} -> {filtered_count} jobs")
            
            return filtered_df
            
        except Exception as e:
            if isinstance(e, (FilterValidationError, FilterApplicationError)):
                raise
            raise FilterApplicationError(f"Failed to apply {filter_name} filter: {str(e)}")
    
    def apply_filter(self, 
                    df: pd.DataFrame,
                    filter_name: str,
                    filter_criteria: Union[str, List[str], Dict[str, Any]],
                    **kwargs) -> pd.DataFrame:
        """
        Alias for apply_single_filter for compatibility.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_name (str): Name of the filter to apply
            filter_criteria: Criteria for the filter
            **kwargs: Additional arguments for the filter
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        return self.apply_single_filter(df, filter_name, filter_criteria, **kwargs)
    
    def apply_multiple_filters(self, 
                              df: pd.DataFrame,
                              filter_config: Dict[str, Union[str, List[str], Dict[str, Any]]],
                              filter_order: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply multiple filters to the dataset in sequence.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_config (Dict): Configuration for each filter
                Format: {filter_name: filter_criteria, ...}
            filter_order (List[str], optional): Custom order for applying filters.
                Defaults to optimized order.
                
        Returns:
            pd.DataFrame: Dataset after applying all filters
            
        Raises:
            FilterValidationError: If any filter configuration is invalid
            FilterApplicationError: If any filtering operation fails
        """
        if not filter_config:
            self.logger.warning("No filters specified in filter_config")
            return df
        
        # Use custom order or default optimized order
        if filter_order is None:
            # Apply filters in order that typically reduces dataset size most efficiently
            filter_order = [f for f in self._filter_order if f in filter_config]
        
        # Validate all filters exist before applying any
        for filter_name in filter_config:
            if filter_name not in self._filters:
                available_filters = ', '.join(self.get_available_filters())
                raise FilterValidationError(f"Filter '{filter_name}' not available. "
                                          f"Available filters: {available_filters}")
        
        filtered_df = df.copy()
        original_count = len(df)
        
        self.logger.info(f"Starting multi-filter operation on {original_count} jobs")
        
        # Apply filters in sequence
        for filter_name in filter_order:
            if filter_name in filter_config:
                filter_criteria = filter_config[filter_name]
                
                try:
                    before_count = len(filtered_df)
                    filtered_df = self.apply_single_filter(filtered_df, filter_name, filter_criteria)
                    after_count = len(filtered_df)
                    
                    self.logger.info(f"{filter_name} filter: {before_count} -> {after_count} jobs")
                    
                    # Early termination if no jobs remain
                    if len(filtered_df) == 0:
                        self.logger.warning("No jobs remaining after applying filters")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Failed to apply {filter_name} filter: {str(e)}")
                    raise
        
        final_count = len(filtered_df)
        self.logger.info(f"Multi-filter operation complete: {original_count} -> {final_count} jobs")
        
        return filtered_df
    
    def apply_combined_filter(self, 
                             df: pd.DataFrame,
                             filter_config: Dict[str, Union[str, List[str], Dict[str, Any]]],
                             logic_operator: str = 'AND',
                             filter_order: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply multiple filters with specified logic operator.
        
        Args:
            df (pd.DataFrame): The job dataset to filter
            filter_config (Dict): Configuration for each filter
            logic_operator (str): 'AND' or 'OR' logic for combining filters
            filter_order (List[str], optional): Custom order for applying filters
                
        Returns:
            pd.DataFrame: Dataset after applying combined filters
            
        Raises:
            FilterValidationError: If any filter configuration is invalid
            FilterApplicationError: If any filtering operation fails
        """
        if not filter_config:
            self.logger.warning("No filters specified in filter_config")
            return df
        
        if logic_operator.upper() == 'AND':
            # For AND logic, apply filters sequentially (intersection)
            return self.apply_multiple_filters(df, filter_config, filter_order)
        
        elif logic_operator.upper() == 'OR':
            # For OR logic, apply each filter separately and combine results (union)
            if not filter_config:
                return df
            
            combined_results = pd.DataFrame()
            original_count = len(df)
            
            self.logger.info(f"Starting OR-combined filter operation on {original_count} jobs")
            
            for filter_name, filter_criteria in filter_config.items():
                if filter_name not in self._filters:
                    available_filters = ', '.join(self.get_available_filters())
                    raise FilterValidationError(f"Filter '{filter_name}' not available. "
                                              f"Available filters: {available_filters}")
                
                try:
                    # Apply single filter to original dataset
                    filtered_subset = self.apply_single_filter(df, filter_name, filter_criteria)
                    
                    # Combine with previous results (union)
                    if combined_results.empty:
                        combined_results = filtered_subset.copy()
                    else:
                        combined_results = pd.concat([combined_results, filtered_subset]).drop_duplicates()
                    
                    self.logger.info(f"After {filter_name} filter (OR): {len(combined_results)} total jobs")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply {filter_name} filter: {str(e)}")
                    raise
            
            final_count = len(combined_results)
            self.logger.info(f"OR-combined filter operation complete: {original_count} -> {final_count} jobs")
            
            return combined_results
        
        else:
            raise FilterValidationError(f"Invalid logic_operator '{logic_operator}'. Must be 'AND' or 'OR'")
    
    def get_filter_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics for all available filters on the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            Dict[str, Any]: Statistics for each filter type
        """
        stats = {
            'dataset_size': len(df),
            'filters': {}
        }
        
        for filter_name, filter_instance in self._filters.items():
            try:
                if hasattr(filter_instance, 'get_available_options'):
                    options = filter_instance.get_available_options(df)
                    stats['filters'][filter_name] = {
                        'available_options': options,
                        'option_count': len(options)
                    }
                    
                # Get detailed statistics if available
                if hasattr(filter_instance, 'get_gender_statistics') and filter_name == 'gender':
                    stats['filters'][filter_name]['detailed_stats'] = filter_instance.get_gender_statistics(df)
                elif hasattr(filter_instance, 'get_work_type_statistics') and filter_name == 'work_type':
                    stats['filters'][filter_name]['detailed_stats'] = filter_instance.get_work_type_statistics(df)
                elif hasattr(filter_instance, 'get_education_statistics') and filter_name == 'education':
                    stats['filters'][filter_name]['detailed_stats'] = filter_instance.get_education_statistics(df)
                    
            except Exception as e:
                self.logger.warning(f"Could not get statistics for {filter_name} filter: {str(e)}")
                stats['filters'][filter_name] = {'error': str(e)}
        
        return stats
    
    def validate_filter_config(self, filter_config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate a filter configuration without applying it.
        
        Args:
            filter_config (Dict): Configuration to validate
            
        Returns:
            Dict[str, bool]: Validation results for each filter
        """
        validation_results = {}
        
        for filter_name, filter_criteria in filter_config.items():
            try:
                if filter_name not in self._filters:
                    validation_results[filter_name] = False
                    continue
                
                filter_instance = self.get_filter(filter_name)
                is_valid = filter_instance.validate_criteria(filter_criteria)
                validation_results[filter_name] = is_valid
                
            except Exception as e:
                self.logger.warning(f"Validation error for {filter_name}: {str(e)}")
                validation_results[filter_name] = False
        
        return validation_results
    
    def get_filter_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all filters and their impact.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive filter summary
        """
        summary = {
            'total_jobs': len(df),
            'available_filters': self.get_available_filters(),
            'filter_statistics': self.get_filter_statistics(df),
            'recommended_filter_order': self._filter_order
        }
        
        return summary


# Convenience functions for easy usage
def create_filter_manager() -> FilterManager:
    """
    Convenience function to create a FilterManager instance.
    
    Returns:
        FilterManager: A new FilterManager instance
    """
    return FilterManager()


def apply_filters(df: pd.DataFrame, 
                 filter_config: Dict[str, Any],
                 filter_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to apply multiple filters to a dataset.
    
    Args:
        df (pd.DataFrame): The job dataset to filter
        filter_config (Dict): Configuration for each filter
        filter_order (List[str], optional): Custom order for applying filters
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    manager = create_filter_manager()
    return manager.apply_multiple_filters(df, filter_config, filter_order)


def get_available_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Convenience function to get available options for all filters.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        
    Returns:
        Dict[str, List[str]]: Available options for each filter type
    """
    manager = create_filter_manager()
    stats = manager.get_filter_statistics(df)
    
    options = {}
    for filter_name, filter_stats in stats['filters'].items():
        if 'available_options' in filter_stats:
            options[filter_name] = filter_stats['available_options']
    
    return options