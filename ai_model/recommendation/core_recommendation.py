#!/usr/bin/env python3
"""
Core Recommendation Logic

This module provides a streamlined recommendation function that takes core parameters
directly from the user for both filtering and scoring, without complex user profile management.

The function focuses on the essential parameters needed by the filter and scoring systems:
- Filter parameters: gender, work_type, education
- Scoring parameters: experience, location (lat/lng), query_embedding


"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging

# Import filter and scoring components
from .filter.filter_manager import FilterManager
from .score.scoring_manager import ScoringManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recommend_jobs(
    jobs_df: pd.DataFrame,
    # Filter parameters
    gender: Optional[str] = None,
    work_type: Optional[Union[str, List[str]]] = None,
    education: Optional[Union[str, List[str]]] = None,
    # Scoring parameters
    user_experience: Optional[float] = None,
    user_latitude: Optional[float] = None,
    user_longitude: Optional[float] = None,
    query_embedding: Optional[np.ndarray] = None,
    # Configuration parameters
    embeddings_dir: str = "d:\\ai courses\\JobHunt\\ai_model\\embeddings",
    max_recommendations: int = 10,
    filter_logic: str = "AND",
    scoring_weights: Optional[Dict[str, float]] = None,
    # Distance preferences for location scoring
    preferred_distance_km: float = 50,
    max_distance_km: float = 200
) -> List[Dict[str, Any]]:
    """
    Generate job recommendations based on core user parameters.
    
    Args:
        jobs_df: DataFrame containing job data
        
        Filter Parameters:
        gender: User's gender preference ("male" or "female")
        work_type: Desired work type(s) ("Contract", "Part-time", "Internship", "Full-time")
        education: Education level(s) ("Bachelor", "Master", "PhD")
        
        Scoring Parameters:
        user_experience: User's years of experience (float)
        user_latitude: User's latitude coordinate (float)
        user_longitude: User's longitude coordinate (float)
        query_embedding: User's query as embedding vector (numpy array)
        
        Configuration:
        embeddings_dir: Directory containing pre-computed job embeddings
        max_recommendations: Maximum number of recommendations to return
        filter_logic: Logic for combining filters ("AND" or "OR")
        scoring_weights: Custom weights for scoring components
        preferred_distance_km: Preferred distance for location scoring
        max_distance_km: Maximum acceptable distance for location scoring
        
    Returns:
        List of job recommendations with scores and details
        
    Raises:
        ValueError: If required parameters are missing or invalid
        FileNotFoundError: If job data or embeddings are not found
    """
    
    try:
        # Validate input data
        if jobs_df is None or jobs_df.empty:
            raise ValueError("jobs_df cannot be None or empty")
        
        logger.info(f"Starting recommendation process with {len(jobs_df)} jobs")
        
        # Step 1: Apply filters if any filter criteria provided
        filtered_df = jobs_df.copy()
        
        filter_criteria = {}
        if gender is not None:
            filter_criteria['gender'] = gender
        if work_type is not None:
            filter_criteria['work_type'] = work_type
        if education is not None:
            filter_criteria['education'] = education
        
        if filter_criteria:
            logger.info(f"Applying filters: {filter_criteria}")
            filter_manager = FilterManager()
            
            # Use combined filter with logic operator (AND/OR)
            filtered_df = filter_manager.apply_combined_filter(
                filtered_df,
                filter_criteria,
                logic_operator=filter_logic
            )
            
            logger.info(f"After filtering: {len(filtered_df)} jobs remaining")
        
        # If no jobs remain after filtering, return empty list
        if filtered_df.empty:
            logger.warning("No jobs remain after filtering")
            return []
        
        # Step 2: Apply scoring if any scoring criteria provided
        scoring_criteria = {}
        
        if user_experience is not None:
            scoring_criteria['years_experience'] = user_experience
        
        if user_latitude is not None and user_longitude is not None:
            scoring_criteria['latitude'] = user_latitude
            scoring_criteria['longitude'] = user_longitude
        
        if query_embedding is not None:
            scoring_criteria['query_embedding'] = query_embedding
        
        if scoring_criteria:
            logger.info(f"Applying scoring with criteria: {list(scoring_criteria.keys())}")
            
            # Create scoring manager with custom configuration
            scoring_manager = ScoringManager()
            
            # Configure scorers based on available criteria
            if user_experience is not None:
                weight = scoring_weights.get('experience', 0.7) if scoring_weights else 0.7
                scoring_manager.configure_scorer('experience', {
                    'weight': weight,
                    'parameters': {}
                })
            
            if 'latitude' in scoring_criteria and 'longitude' in scoring_criteria:
                weight = scoring_weights.get('location', 0.8) if scoring_weights else 0.8
                scoring_manager.configure_scorer('location', {
                    'weight': weight,
                    'parameters': {
                        'preferred_distance_km': preferred_distance_km,
                        'max_distance_km': max_distance_km
                    }
                })
            
            if 'query_embedding' in scoring_criteria:
                weight = scoring_weights.get('embedding', 1.5) if scoring_weights else 1.7
                scoring_manager.configure_scorer('embedding', {
                    'weight': weight,
                    'parameters': {
                        'embeddings_dir': embeddings_dir
                    }
                })
            
            # Calculate scores
            scored_df = scoring_manager.calculate_scores(filtered_df, scoring_criteria)
            
            # Map combined_score to total_score for downstream compatibility
            if 'combined_score' in scored_df.columns:
                scored_df['total_score'] = scored_df['combined_score']
            
            # Sort by total score (descending)
            scored_df = scored_df.sort_values('total_score', ascending=False)
            
        else:
            # No scoring criteria provided, use original order
            logger.info("No scoring criteria provided, using original job order")
            scored_df = filtered_df.copy()
            scored_df['total_score'] = 1.0  # Default score
        
        # Step 3: Format recommendations
        recommendations = []
        top_jobs = scored_df.head(max_recommendations)
        
        for idx, row in top_jobs.iterrows():
            recommendation = {
                'job_id': row.get('Job_Id', row.get('Job Id', 'N/A')),
                'job_title': row.get('Job Title', 'N/A'),
                'role': row.get('Role', 'N/A'),
                'job_description': row.get('Job Description', 'N/A'),
                'benefits': row.get('Benefits', 'N/A'),
                'skills': row.get('skills', 'N/A'),
                'responsibilities': row.get('Responsibilities', 'N/A'),
                'company': row.get('Company', 'N/A'),
                'company_size': row.get('Company Size', 'N/A'),
                'company_profile': row.get('Company Profile', 'N/A'),
                'experience': row.get('Experience', 'N/A'),
                'salary_range': row.get('Salary Range', 'N/A'),
                'qualifications': row.get('Qualifications', 'N/A'),
                'location': row.get('location', 'N/A'),
                'country': row.get('Country', 'N/A'),
                'work_type': row.get('Work_Type', row.get('Work Type', 'N/A')),
                'preference': row.get('Preference', 'N/A'),
                'latitude': row.get('latitude', 'N/A'),
                'longitude': row.get('longitude', 'N/A'),
                'total_score': float(row.get('total_score', 0.0)),
                'individual_scores': {}
            }
            
            # Add individual scores if available
            for col in row.index:
                if col.endswith('_score') and col != 'total_score':
                    recommendation['individual_scores'][col] = float(row[col])
            
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec['job_title']} at {rec['company']} (Score: {rec['total_score']:.3f})")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in recommendation process: {str(e)}")
        raise


def quick_recommend(
    jobs_df: pd.DataFrame,
    user_params: Dict[str, Any],
    max_recommendations: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience function for quick recommendations using a parameter dictionary.
    
    Args:
        jobs_df: DataFrame containing job data
        user_params: Dictionary containing user parameters:
            - 'gender': str (optional)
            - 'work_type': str or list (optional)
            - 'education': str or list (optional)
            - 'experience': float (optional)
            - 'latitude': float (optional)
            - 'longitude': float (optional)
            - 'query_embedding': np.ndarray (optional)
        max_recommendations: Maximum number of recommendations
        
    Returns:
        List of job recommendations
    """
    
    return recommend_jobs(
        jobs_df=jobs_df,
        gender=user_params.get('gender'),
        work_type=user_params.get('work_type'),
        education=user_params.get('education'),
        user_experience=user_params.get('experience'),
        user_latitude=user_params.get('latitude'),
        user_longitude=user_params.get('longitude'),
        query_embedding=user_params.get('query_embedding'),
        max_recommendations=max_recommendations
    )


def get_recommendation_parameters_info() -> Dict[str, Any]:
    """
    Get information about available recommendation parameters.
    
    Returns:
        Dictionary containing parameter information and valid values
    """
    
    return {
        'filter_parameters': {
            'gender': {
                'type': 'string',
                'valid_values': ['male', 'female'],
                'description': 'User gender preference for job filtering'
            },
            'work_type': {
                'type': 'string or list of strings',
                'valid_values': ['Contract', 'Part-time', 'Internship', 'Full-time'],
                'description': 'Desired work type(s)'
            },
            'education': {
                'type': 'string or list of strings',
                'valid_values': ['Bachelor', 'Master', 'PhD'],
                'description': 'Required education level(s)'
            }
        },
        'scoring_parameters': {
            'user_experience': {
                'type': 'float',
                'description': 'User years of experience for experience-based scoring'
            },
            'user_latitude': {
                'type': 'float',
                'description': 'User latitude coordinate for location-based scoring'
            },
            'user_longitude': {
                'type': 'float',
                'description': 'User longitude coordinate for location-based scoring'
            },
            'query_embedding': {
                'type': 'numpy.ndarray',
                'description': 'User query embedding vector for similarity-based scoring'
            }
        },
        'configuration_parameters': {
            'max_recommendations': {
                'type': 'int',
                'default': 10,
                'description': 'Maximum number of recommendations to return'
            },
            'filter_logic': {
                'type': 'string',
                'valid_values': ['AND', 'OR'],
                'default': 'AND',
                'description': 'Logic for combining multiple filters'
            },
            'scoring_weights': {
                'type': 'dict',
                'description': 'Custom weights for scoring components (experience, location, embedding)'
            },
            'preferred_distance_km': {
                'type': 'float',
                'default': 50,
                'description': 'Preferred distance for location scoring'
            },
            'max_distance_km': {
                'type': 'float',
                'default': 200,
                'description': 'Maximum acceptable distance for location scoring'
            }
        }
    }


# Example usage functions for testing
def create_sample_user_params() -> Dict[str, Any]:
    """Create sample user parameters for testing."""
    return {
        'gender': 'male',
        'work_type': ['Full-time', 'Contract'],
        'education': 'Bachelor',
        'experience': 3.5,
        'latitude': 40.7128,  # New York City
        'longitude': -74.0060,
        # query_embedding would be provided by the backend
    }


def test_recommendation_with_sample_data(jobs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Test the recommendation function with sample parameters."""
    
    sample_params = create_sample_user_params()
    
    logger.info("Testing recommendation with sample parameters:")
    logger.info(f"Parameters: {sample_params}")
    
    recommendations = quick_recommend(jobs_df, sample_params, max_recommendations=5)
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['job_title']} at {rec['company']} (Score: {rec['total_score']:.3f})")
    
    return recommendations