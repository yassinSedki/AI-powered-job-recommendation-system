"""
Scores Module

This module contains all the individual scorer implementations for the job recommendation system.
Each scorer implements the BaseJobScorer interface and provides specific scoring logic.

Available Scorers:
- BaseJobScorer: Abstract base class for all scorers
- ExperienceScorer: Scores jobs based on experience requirements
- LocationScorer: Scores jobs based on location preferences
- EmbeddingSimilarityScorer: Scores jobs based on embedding similarity


"""

from .base_scorer import (
    BaseJobScorer,
    ScoringError,
    ScoringValidationError,
    ScoreAggregator
)
from .experience_scorer import ExperienceScorer
from .location_scorer import LocationScorer
from .embedding_scorer import (
    EmbeddingSimilarityScorer,
    EmbeddingLoader,
    EmbeddingScorerError
)

__all__ = [
    # Base classes and exceptions
    'BaseJobScorer',
    'ScoringError', 
    'ScoringValidationError',
    'ScoreAggregator',
    
    # Scorer implementations
    'ExperienceScorer',
    'LocationScorer',
    'EmbeddingSimilarityScorer',
    
    # Utility classes
    'EmbeddingLoader',
    'EmbeddingScorerError'
]