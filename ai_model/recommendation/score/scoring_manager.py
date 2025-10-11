"""
Scoring Manager for Job Recommendation System

This module provides the ScoringManager class that coordinates multiple scoring filters
to calculate comprehensive job scores based on various criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass

from .scores.base_scorer import BaseJobScorer, ScoringError, ScoreAggregator
from .scores.experience_scorer import ExperienceScorer
from .scores.location_scorer import LocationScorer
from .scores.embedding_scorer import EmbeddingSimilarityScorer


@dataclass
class ScorerConfig:
    """Configuration for a scorer including its weight and parameters."""
    scorer_class: type
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class ScoringManager:
    """
    Manages multiple job scoring filters and combines their results.
    
    This class provides a unified interface for applying multiple scoring criteria
    to job datasets and combining the results into a final score.
    """
    
    def __init__(self, normalize_scores: bool = True, default_weight: float = 1.0):
        """
        Initialize the ScoringManager.
        
        Args:
            normalize_scores: Whether to normalize individual scores before combining
            default_weight: Default weight for scorers when not specified
        """
        self.scorers: Dict[str, BaseJobScorer] = {}
        self.scorer_weights: Dict[str, float] = {}
        self.normalize_scores = normalize_scores
        self.default_weight = default_weight
        self.logger = logging.getLogger(__name__)
        
        # Available scorer configurations
        self.available_scorers = {
            'experience': ScorerConfig(
                scorer_class=ExperienceScorer,
                weight=1.0,
                parameters={}
            ),
            'location': ScorerConfig(
                scorer_class=LocationScorer,
                weight=0.8,
                parameters={
                    'preferred_distance_km': 50,
                    'max_distance_km': 200
                }
            ),
            'embedding': ScorerConfig(
                scorer_class=EmbeddingSimilarityScorer,
                weight=1.2,
                parameters={
                    'embeddings_dir': 'embeddings'
                }
            )
        }
    
    def add_scorer(self, name: str, scorer: BaseJobScorer, weight: float = None) -> 'ScoringManager':
        """
        Add a scorer to the manager.
        
        Args:
            name: Unique name for the scorer
            scorer: The scorer instance
            weight: Weight for combining scores (uses default_weight if None)
            
        Returns:
            Self for method chaining
            
        Raises:
            ScoringError: If scorer name already exists or scorer is invalid
        """
        if name in self.scorers:
            raise ScoringError(f"Scorer '{name}' already exists")
        
        if not isinstance(scorer, BaseJobScorer):
            raise ScoringError(f"Scorer must be an instance of BaseJobScorer")
        
        self.scorers[name] = scorer
        self.scorer_weights[name] = weight if weight is not None else self.default_weight
        
        self.logger.info(f"Added scorer '{name}' with weight {self.scorer_weights[name]}")
        return self
    
    def remove_scorer(self, name: str) -> 'ScoringManager':
        """
        Remove a scorer from the manager.
        
        Args:
            name: Name of the scorer to remove
            
        Returns:
            Self for method chaining
            
        Raises:
            ScoringError: If scorer doesn't exist
        """
        if name not in self.scorers:
            raise ScoringError(f"Scorer '{name}' not found")
        
        del self.scorers[name]
        del self.scorer_weights[name]
        
        self.logger.info(f"Removed scorer '{name}'")
        return self
    
    def configure_scorer(self, name: str, config: Union[Dict[str, Any], ScorerConfig]) -> 'ScoringManager':
        """
        Configure and add a scorer using predefined configurations.
        
        Args:
            name: Name of the scorer type ('experience', 'location')
            config: Configuration dictionary or ScorerConfig object
            
        Returns:
            Self for method chaining
            
        Raises:
            ScoringError: If scorer type is not available
        """
        if name not in self.available_scorers:
            raise ScoringError(f"Unknown scorer type '{name}'. Available: {list(self.available_scorers.keys())}")
        
        base_config = self.available_scorers[name]
        
        if isinstance(config, dict):
            # Merge with default configuration
            parameters = {**base_config.parameters, **config.get('parameters', {})}
            weight = config.get('weight', base_config.weight)
            enabled = config.get('enabled', base_config.enabled)
        else:
            parameters = {**base_config.parameters, **config.parameters}
            weight = config.weight
            enabled = config.enabled
        
        if enabled:
            scorer_instance = base_config.scorer_class(**parameters)
            self.add_scorer(name, scorer_instance, weight)
        
        return self
    
    def update_weight(self, name: str, weight: float) -> 'ScoringManager':
        """
        Update the weight of an existing scorer.
        
        Args:
            name: Name of the scorer
            weight: New weight value
            
        Returns:
            Self for method chaining
            
        Raises:
            ScoringError: If scorer doesn't exist
        """
        if name not in self.scorers:
            raise ScoringError(f"Scorer '{name}' not found")
        
        self.scorer_weights[name] = weight
        self.logger.info(f"Updated weight for scorer '{name}' to {weight}")
        return self
    
    def calculate_scores(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate scores for all jobs using all configured scorers.
        
        Args:
            jobs_df: DataFrame containing job data
            user_criteria: Dictionary containing user criteria for scoring
            
        Returns:
            DataFrame with original job data plus individual and combined scores
            
        Raises:
            ScoringError: If no scorers are configured or calculation fails
        """
        if not self.scorers:
            raise ScoringError("No scorers configured")
        
        if jobs_df.empty:
            self.logger.warning("Empty jobs DataFrame provided")
            return jobs_df.copy()
        
        result_df = jobs_df.copy()
        individual_scores = {}
        
        # Calculate individual scores
        for name, scorer in self.scorers.items():
            try:
                self.logger.debug(f"Calculating scores using '{name}' scorer")
                scores = scorer.calculate_score(jobs_df, user_criteria)
                
                if self.normalize_scores:
                    scores = scorer.normalize_score(scores)
                
                individual_scores[name] = scores
                result_df[f'{name}_score'] = scores
                
            except Exception as e:
                self.logger.error(f"Error calculating scores with '{name}' scorer: {str(e)}")
                raise ScoringError(f"Failed to calculate scores with '{name}' scorer: {str(e)}")
        
        # Calculate combined score
        combined_score = self._combine_scores(individual_scores)
        result_df['combined_score'] = combined_score
        
        # Sort by combined score (highest first)
        result_df = result_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        
        self.logger.info(f"Calculated scores for {len(result_df)} jobs using {len(self.scorers)} scorers")
        return result_df
    
    def _combine_scores(self, individual_scores: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine individual scores using weighted average.
        
        Args:
            individual_scores: Dictionary of scorer names to score series
            
        Returns:
            Combined score series
        """
        if not individual_scores:
            raise ScoringError("No individual scores to combine")
        
        # Use ScoreAggregator for combining scores
        # Build a weights dict and pass individual_scores directly
        weights = {name: self.scorer_weights[name] for name in individual_scores.keys()}
        return ScoreAggregator.weighted_average(individual_scores, weights)
    
    def get_top_jobs(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any], 
                     top_n: int = 10) -> pd.DataFrame:
        """
        Get top N jobs based on combined scoring.
        
        Args:
            jobs_df: DataFrame containing job data
            user_criteria: Dictionary containing user criteria for scoring
            top_n: Number of top jobs to return
            
        Returns:
            DataFrame with top N jobs sorted by combined score
        """
        scored_df = self.calculate_scores(jobs_df, user_criteria)
        return scored_df.head(top_n)
    
    def get_scoring_summary(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of scoring results including statistics for each scorer.
        
        Args:
            jobs_df: DataFrame containing job data
            user_criteria: Dictionary containing user criteria for scoring
            
        Returns:
            Dictionary containing scoring summary and statistics
        """
        scored_df = self.calculate_scores(jobs_df, user_criteria)
        
        summary = {
            'total_jobs': len(scored_df),
            'scorers_used': list(self.scorers.keys()),
            'scorer_weights': self.scorer_weights.copy(),
            'score_statistics': {}
        }
        
        # Calculate statistics for each scorer
        for name in self.scorers.keys():
            score_col = f'{name}_score'
            if score_col in scored_df.columns:
                scores = scored_df[score_col]
                summary['score_statistics'][name] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'median': float(scores.median())
                }
        
        # Combined score statistics
        if 'combined_score' in scored_df.columns:
            combined_scores = scored_df['combined_score']
            summary['score_statistics']['combined'] = {
                'mean': float(combined_scores.mean()),
                'std': float(combined_scores.std()),
                'min': float(combined_scores.min()),
                'max': float(combined_scores.max()),
                'median': float(combined_scores.median())
            }
        
        return summary
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current scoring configuration.
        
        Returns:
            List of validation warnings/issues
        """
        issues = []
        
        if not self.scorers:
            issues.append("No scorers configured")
        
        total_weight = sum(self.scorer_weights.values())
        if total_weight == 0:
            issues.append("Total weight of all scorers is zero")
        
        for name, scorer in self.scorers.items():
            try:
                # Test validation with empty criteria
                scorer.validate_criteria({})
            except Exception as e:
                issues.append(f"Scorer '{name}' validation failed: {str(e)}")
        
        return issues
    
    def get_required_columns(self) -> Dict[str, List[str]]:
        """
        Get required columns for all configured scorers.
        
        Returns:
            Dictionary mapping scorer names to their required columns
        """
        required_columns = {}
        
        for name, scorer in self.scorers.items():
            required_columns[name] = scorer.get_required_columns()
        
        return required_columns
    
    def __repr__(self) -> str:
        """String representation of the ScoringManager."""
        return f"ScoringManager(scorers={list(self.scorers.keys())}, weights={self.scorer_weights})"


def create_default_scoring_manager(embeddings_dir: str = 'embeddings') -> ScoringManager:
    """
    Create a ScoringManager with default configuration.
    
    Args:
        embeddings_dir: Directory containing pre-computed embeddings
    
    Returns:
        ScoringManager with experience, location, and embedding scorers configured
    """
    manager = ScoringManager()
    
    # Configure experience scorer
    manager.configure_scorer('experience', {
        'weight': 1.0,
        'parameters': {}
    })
    
    # Configure location scorer
    manager.configure_scorer('location', {
        'weight': 0.8,
        'parameters': {
            'preferred_distance_km': 50,
            'max_distance_km': 200
        }
    })
    
    # Configure embedding scorer
    manager.configure_scorer('embedding', {
        'weight': 1.2,
        'parameters': {
            'embeddings_dir': embeddings_dir
        }
    })
    
    return manager


def score_jobs(jobs_df: pd.DataFrame, user_criteria: Dict[str, Any], 
               scorer_config: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Convenience function to score jobs with default or custom configuration.
    
    Args:
        jobs_df: DataFrame containing job data
        user_criteria: Dictionary containing user criteria for scoring
        scorer_config: Optional custom scorer configuration
        
    Returns:
        DataFrame with scored jobs
    """
    if scorer_config is None:
        manager = create_default_scoring_manager()
    else:
        manager = ScoringManager()
        for scorer_name, config in scorer_config.items():
            manager.configure_scorer(scorer_name, config)
    
    return manager.calculate_scores(jobs_df, user_criteria)