#!/usr/bin/env python3
"""
Base Job Scorer

This module provides the abstract base class for all job scoring implementations.
All scoring filters should inherit from BaseJobScorer and implement the required methods.


"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringError(Exception):
    """Base exception for scoring-related errors."""
    pass


class ScoringValidationError(ScoringError):
    """Exception raised when scoring validation fails."""
    pass


class ScoringApplicationError(ScoringError):
    """Exception raised when scoring application fails."""
    pass


class BaseJobScorer(ABC):
    """
    Abstract base class for job scoring implementations.
    
    This class defines the interface that all job scorers must implement.
    Each scorer should calculate a score between 0.0 and 1.0 for how well
    a job matches a user's criteria.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the base scorer.
        
        Args:
            name: Optional name for the scorer
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def calculate_score(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> pd.Series:
        """
        Calculate scores for jobs based on user criteria.
        
        Args:
            jobs_df: DataFrame containing job data
            user_criteria: Dictionary containing user's criteria for scoring
            
        Returns:
            pd.Series: Scores for each job (0.0 to 1.0, higher is better)
            
        Raises:
            ScoringValidationError: If input validation fails
            ScoringApplicationError: If scoring calculation fails
        """
        pass
    
    @abstractmethod
    def validate_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user criteria for this scorer.
        
        Args:
            criteria: User criteria to validate
            
        Returns:
            Dict containing validation results with 'valid' boolean and 'errors' list
            
        Raises:
            ScoringValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns in the jobs DataFrame.
        
        Returns:
            List of required column names
        """
        pass
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ScoringValidationError: If required columns are missing
        """
        if df.empty:
            raise ScoringValidationError("DataFrame is empty")
        
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ScoringValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Required columns: {required_columns}"
            )
    
    def normalize_score(self, scores: Union[pd.Series, np.ndarray], 
                       min_score: float = 0.0, max_score: float = 1.0) -> pd.Series:
        """
        Normalize scores to the range [min_score, max_score].
        
        Args:
            scores: Raw scores to normalize
            min_score: Minimum score value (default: 0.0)
            max_score: Maximum score value (default: 1.0)
            
        Returns:
            pd.Series: Normalized scores
        """
        if isinstance(scores, np.ndarray):
            scores = pd.Series(scores)
        
        if scores.empty:
            return scores
        
        # Handle case where all scores are the same
        if scores.min() == scores.max():
            return pd.Series([max_score] * len(scores), index=scores.index)
        
        # Min-max normalization
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        normalized = normalized * (max_score - min_score) + min_score
        
        return normalized
    
    def apply_score_weights(self, scores: pd.Series, weight: float = 1.0) -> pd.Series:
        """
        Apply weight to scores.
        
        Args:
            scores: Scores to weight
            weight: Weight to apply (default: 1.0)
            
        Returns:
            pd.Series: Weighted scores
        """
        return scores * weight
    
    def get_score_statistics(self, scores: pd.Series) -> Dict[str, float]:
        """
        Get statistics about the calculated scores.
        
        Args:
            scores: Calculated scores
            
        Returns:
            Dict containing score statistics
        """
        if scores.empty:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        return {
            'count': len(scores),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(scores.median())
        }
    
    def log_scoring_info(self, jobs_count: int, criteria: Dict[str, Any], 
                        scores: pd.Series) -> None:
        """
        Log information about the scoring process.
        
        Args:
            jobs_count: Number of jobs being scored
            criteria: User criteria used for scoring
            scores: Calculated scores
        """
        stats = self.get_score_statistics(scores)
        
        self.logger.info(
            f"{self.name} scoring completed: "
            f"{jobs_count} jobs, "
            f"avg score: {stats['mean']:.3f}, "
            f"score range: [{stats['min']:.3f}, {stats['max']:.3f}]"
        )
    
    def __str__(self) -> str:
        """String representation of the scorer."""
        return f"{self.name}Scorer"
    
    def __repr__(self) -> str:
        """Detailed string representation of the scorer."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ScoreAggregator:
    """
    Utility class for aggregating multiple scores.
    """
    
    @staticmethod
    def weighted_average(scores_dict: Dict[str, pd.Series], 
                        weights: Dict[str, float] = None) -> pd.Series:
        """
        Calculate weighted average of multiple score series.
        
        Args:
            scores_dict: Dictionary of score series {scorer_name: scores}
            weights: Dictionary of weights {scorer_name: weight}
            
        Returns:
            pd.Series: Aggregated scores
        """
        if not scores_dict:
            return pd.Series(dtype=float)
        
        if weights is None:
            weights = {name: 1.0 for name in scores_dict.keys()}
        
        # Ensure all series have the same index
        first_series = next(iter(scores_dict.values()))
        index = first_series.index
        
        weighted_sum = pd.Series(0.0, index=index)
        total_weight = 0.0
        
        for name, scores in scores_dict.items():
            weight = weights.get(name, 1.0)
            weighted_sum += scores * weight
            total_weight += weight
        
        if total_weight == 0:
            return pd.Series(0.0, index=index)
        
        return weighted_sum / total_weight
    
    @staticmethod
    def max_score(scores_dict: Dict[str, pd.Series]) -> pd.Series:
        """
        Take the maximum score for each job across all scorers.
        
        Args:
            scores_dict: Dictionary of score series
            
        Returns:
            pd.Series: Maximum scores
        """
        if not scores_dict:
            return pd.Series(dtype=float)
        
        scores_df = pd.DataFrame(scores_dict)
        return scores_df.max(axis=1)
    
    @staticmethod
    def min_score(scores_dict: Dict[str, pd.Series]) -> pd.Series:
        """
        Take the minimum score for each job across all scorers.
        
        Args:
            scores_dict: Dictionary of score series
            
        Returns:
            pd.Series: Minimum scores
        """
        if not scores_dict:
            return pd.Series(dtype=float)
        
        scores_df = pd.DataFrame(scores_dict)
        return scores_df.min(axis=1)