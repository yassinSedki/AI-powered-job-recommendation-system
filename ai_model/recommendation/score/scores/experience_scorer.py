#!/usr/bin/env python3
"""
Experience Job Scorer

This module implements job scoring based on user's years of experience
compared to job requirements (min_experience, max_experience).

The scoring algorithm gives higher scores to jobs where the user's experience
is closer to the maximum required experience, with penalties for being
under-qualified or over-qualified.


"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .base_scorer import BaseJobScorer, ScoringValidationError, ScoringApplicationError


class ExperienceScorer(BaseJobScorer):
    """
    Scorer that evaluates jobs based on experience requirements.
    
    Scoring Logic:
    - Perfect match (user experience = max_experience): Score = 1.0
    - Within range (min_experience <= user <= max_experience): Score = 0.7 to 1.0
    - Under-qualified (user < min_experience): Score decreases based on gap
    - Over-qualified (user > max_experience): Score decreases based on excess
    """
    
    def __init__(self, 
                 under_qualified_penalty: float = 0.1,
                 over_qualified_penalty: float = 0.05,
                 max_penalty: float = 0.9):
        """
        Initialize the Experience Scorer.
        
        Args:
            under_qualified_penalty: Penalty per year under minimum (default: 0.1)
            over_qualified_penalty: Penalty per year over maximum (default: 0.05)
            max_penalty: Maximum penalty to apply (default: 0.9, min score = 0.1)
        """
        super().__init__("Experience")
        self.under_qualified_penalty = under_qualified_penalty
        self.over_qualified_penalty = over_qualified_penalty
        self.max_penalty = max_penalty
    
    def get_required_columns(self) -> List[str]:
        """
        Get required columns for experience scoring.
        
        Returns:
            List of required column names
        """
        return ['min_experience', 'max_experience']
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user criteria for experience scoring.
        
        Args:
            criteria: User criteria containing 'years_experience'
            
        Returns:
            Dict with validation results
        """
        errors = []
        
        if 'years_experience' not in criteria:
            errors.append("Missing 'years_experience' in criteria")
        else:
            years_exp = criteria['years_experience']
            
            # Check if it's a number
            try:
                years_exp = float(years_exp)
                if years_exp < 0:
                    errors.append("'years_experience' must be non-negative")
                elif years_exp > 50:  # Reasonable upper limit
                    errors.append("'years_experience' seems unreasonably high (>50 years)")
            except (ValueError, TypeError):
                errors.append("'years_experience' must be a number")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def calculate_score(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> pd.Series:
        """
        Calculate experience scores for jobs.
        
        Args:
            jobs_df: DataFrame with job data including min_experience, max_experience
            user_criteria: Dict containing 'years_experience'
            
        Returns:
            pd.Series: Experience scores (0.0 to 1.0)
        """
        # Validate inputs
        self.validate_dataframe(jobs_df)
        validation = self.validate_criteria(user_criteria)
        
        if not validation['valid']:
            raise ScoringValidationError(f"Invalid criteria: {validation['errors']}")
        
        if jobs_df.empty:
            return pd.Series(dtype=float)
        
        try:
            user_experience = float(user_criteria['years_experience'])
            
            # Get job experience requirements
            min_exp = jobs_df['min_experience'].fillna(0)
            max_exp = jobs_df['max_experience'].fillna(min_exp)
            
            # Ensure max_exp >= min_exp
            max_exp = np.maximum(max_exp, min_exp)
            
            # Calculate scores
            scores = self._calculate_experience_scores(user_experience, min_exp, max_exp)
            
            # Log scoring information
            self.log_scoring_info(len(jobs_df), user_criteria, scores)
            
            return scores
            
        except Exception as e:
            raise ScoringApplicationError(f"Failed to calculate experience scores: {str(e)}")
    
    def _calculate_experience_scores(self, user_exp: float, 
                                   min_exp: pd.Series, 
                                   max_exp: pd.Series) -> pd.Series:
        """
        Calculate experience scores based on user experience and job requirements.
        
        Args:
            user_exp: User's years of experience
            min_exp: Minimum experience required for jobs
            max_exp: Maximum experience required for jobs
            
        Returns:
            pd.Series: Calculated scores
        """
        scores = pd.Series(index=min_exp.index, dtype=float)
        
        # Case 1: Perfect match (user experience = max_experience)
        perfect_match = (user_exp == max_exp)
        scores[perfect_match] = 1.0
        
        # Case 2: Within range (min_experience <= user <= max_experience)
        within_range = (user_exp >= min_exp) & (user_exp <= max_exp) & (~perfect_match)
        if within_range.any():
            # Linear interpolation from 0.7 at min to 1.0 at max
            range_scores = 0.7 + 0.3 * (user_exp - min_exp[within_range]) / (
                max_exp[within_range] - min_exp[within_range] + 1e-6
            )
            scores[within_range] = range_scores
        
        # Case 3: Under-qualified (user < min_experience)
        under_qualified = (user_exp < min_exp)
        if under_qualified.any():
            experience_gap = min_exp[under_qualified] - user_exp
            penalty = np.minimum(
                experience_gap * self.under_qualified_penalty,
                self.max_penalty
            )
            scores[under_qualified] = np.maximum(1.0 - penalty, 0.1)
        
        # Case 4: Over-qualified (user > max_experience)
        over_qualified = (user_exp > max_exp)
        if over_qualified.any():
            experience_excess = user_exp - max_exp[over_qualified]
            penalty = np.minimum(
                experience_excess * self.over_qualified_penalty,
                self.max_penalty
            )
            scores[over_qualified] = np.maximum(1.0 - penalty, 0.1)
        
        return scores
    
    def get_experience_match_category(self, user_experience: float, 
                                    min_exp: float, max_exp: float) -> str:
        """
        Categorize the experience match for a single job.
        
        Args:
            user_experience: User's years of experience
            min_exp: Job's minimum experience requirement
            max_exp: Job's maximum experience requirement
            
        Returns:
            str: Match category ('perfect', 'within_range', 'under_qualified', 'over_qualified')
        """
        if user_experience == max_exp:
            return 'perfect'
        elif min_exp <= user_experience <= max_exp:
            return 'within_range'
        elif user_experience < min_exp:
            return 'under_qualified'
        else:
            return 'over_qualified'
    
    def analyze_experience_distribution(self, jobs_df: pd.DataFrame, 
                                      user_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the distribution of experience requirements and matches.
        
        Args:
            jobs_df: DataFrame with job data
            user_criteria: User criteria with experience
            
        Returns:
            Dict with analysis results
        """
        if jobs_df.empty:
            return {'error': 'No jobs to analyze'}
        
        validation = self.validate_criteria(user_criteria)
        if not validation['valid']:
            return {'error': f"Invalid criteria: {validation['errors']}"}
        
        user_exp = float(user_criteria['years_experience'])
        min_exp = jobs_df['min_experience'].fillna(0)
        max_exp = jobs_df['max_experience'].fillna(min_exp)
        
        # Categorize all jobs
        categories = []
        for i in range(len(jobs_df)):
            category = self.get_experience_match_category(
                user_exp, min_exp.iloc[i], max_exp.iloc[i]
            )
            categories.append(category)
        
        category_counts = pd.Series(categories).value_counts()
        
        return {
            'user_experience': user_exp,
            'total_jobs': len(jobs_df),
            'experience_requirements': {
                'min_experience': {
                    'mean': float(min_exp.mean()),
                    'median': float(min_exp.median()),
                    'min': float(min_exp.min()),
                    'max': float(min_exp.max())
                },
                'max_experience': {
                    'mean': float(max_exp.mean()),
                    'median': float(max_exp.median()),
                    'min': float(max_exp.min()),
                    'max': float(max_exp.max())
                }
            },
            'match_distribution': {
                'perfect_match': int(category_counts.get('perfect', 0)),
                'within_range': int(category_counts.get('within_range', 0)),
                'under_qualified': int(category_counts.get('under_qualified', 0)),
                'over_qualified': int(category_counts.get('over_qualified', 0))
            },
            'match_percentages': {
                'perfect_match': float(category_counts.get('perfect', 0) / len(jobs_df) * 100),
                'within_range': float(category_counts.get('within_range', 0) / len(jobs_df) * 100),
                'under_qualified': float(category_counts.get('under_qualified', 0) / len(jobs_df) * 100),
                'over_qualified': float(category_counts.get('over_qualified', 0) / len(jobs_df) * 100)
            }
        }


def score_by_experience(jobs_df: pd.DataFrame, 
                       years_experience: float,
                       under_qualified_penalty: float = 0.1,
                       over_qualified_penalty: float = 0.05) -> pd.Series:
    """
    Convenience function to score jobs by experience.
    
    Args:
        jobs_df: DataFrame with job data
        years_experience: User's years of experience
        under_qualified_penalty: Penalty for being under-qualified
        over_qualified_penalty: Penalty for being over-qualified
        
    Returns:
        pd.Series: Experience scores
    """
    scorer = ExperienceScorer(
        under_qualified_penalty=under_qualified_penalty,
        over_qualified_penalty=over_qualified_penalty
    )
    
    criteria = {'years_experience': years_experience}
    return scorer.calculate_score(jobs_df, criteria)