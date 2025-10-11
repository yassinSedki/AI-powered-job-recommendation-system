"""
Job Scoring System

This package provides a comprehensive job scoring system for the recommendation engine.
It includes various scoring filters and a manager to coordinate multiple scoring criteria.

Main Components:
- BaseJobScorer: Abstract base class for all scorers
- ExperienceScorer: Scores jobs based on experience requirements
- LocationScorer: Scores jobs based on geographic distance
- ScoringManager: Coordinates multiple scorers and combines results

Usage:
    from ai_model.recommendation.score import ScoringManager, ExperienceScorer, LocationScorer
    
    # Create individual scorers
    exp_scorer = ExperienceScorer()
    loc_scorer = LocationScorer(preferred_distance_km=30)
    
    # Or use the manager for coordinated scoring
    manager = ScoringManager()
    manager.add_scorer('experience', exp_scorer, weight=1.0)
    manager.add_scorer('location', loc_scorer, weight=0.8)
    
    # Score jobs
    scored_jobs = manager.calculate_scores(jobs_df, user_criteria)
"""

from .scores import (
    BaseJobScorer,
    ScoringError,
    ScoringValidationError,
    ScoreAggregator,
    ExperienceScorer,
    LocationScorer,
    EmbeddingSimilarityScorer,
    EmbeddingLoader,
    EmbeddingScorerError
)

# Import convenience functions from individual modules
from .scores.experience_scorer import score_by_experience
from .scores.location_scorer import score_by_location

from .scoring_manager import (
    ScoringManager,
    ScorerConfig,
    create_default_scoring_manager,
    score_jobs
)

# Package metadata
__version__ = "1.0.0"

__description__ = "Job scoring system for recommendation engine"

# Main exports
__all__ = [
    # Base classes and errors
    'BaseJobScorer',
    'ScoringError',
    'ScoringValidationError',
    'ScoreAggregator',
    
    # Individual scorers
    'ExperienceScorer',
    'LocationScorer',
    'EmbeddingSimilarityScorer',
    
    # Utility classes
    'EmbeddingLoader',
    'EmbeddingScorerError',
    
    # Scoring manager
    'ScoringManager',
    'ScorerConfig',
    
    # Convenience functions
    'create_default_scoring_manager',
    'score_jobs',
    'score_by_experience',
    'score_by_location',
]

# Available scorer types for configuration
AVAILABLE_SCORERS = {
    'experience': ExperienceScorer,
    'location': LocationScorer,
    'embedding': EmbeddingSimilarityScorer,
}

def get_available_scorers():
    """
    Get a dictionary of available scorer types.
    
    Returns:
        Dict mapping scorer names to their classes
    """
    return AVAILABLE_SCORERS.copy()

def create_scorer(scorer_type: str, **kwargs):
    """
    Factory function to create a scorer by type.
    
    Args:
        scorer_type: Type of scorer ('experience', 'location')
        **kwargs: Parameters to pass to the scorer constructor
        
    Returns:
        Scorer instance
        
    Raises:
        ValueError: If scorer_type is not recognized
    """
    if scorer_type not in AVAILABLE_SCORERS:
        available = list(AVAILABLE_SCORERS.keys())
        raise ValueError(f"Unknown scorer type '{scorer_type}'. Available: {available}")
    
    scorer_class = AVAILABLE_SCORERS[scorer_type]
    return scorer_class(**kwargs)

def quick_score(jobs_df, user_criteria, scorers=None, weights=None):
    """
    Quick scoring function with minimal setup.
    
    Args:
        jobs_df: DataFrame with job data
        user_criteria: User criteria dictionary
        scorers: List of scorer types to use (default: ['experience', 'location'])
        weights: Dictionary of weights for each scorer (default: equal weights)
        
    Returns:
        DataFrame with scored jobs
    """
    if scorers is None:
        scorers = ['experience', 'location']
    
    if weights is None:
        weights = {scorer: 1.0 for scorer in scorers}
    
    manager = ScoringManager()
    
    for scorer_type in scorers:
        weight = weights.get(scorer_type, 1.0)
        scorer = create_scorer(scorer_type)
        manager.add_scorer(scorer_type, scorer, weight)
    
    return manager.calculate_scores(jobs_df, user_criteria)