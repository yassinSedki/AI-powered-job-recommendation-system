#!/usr/bin/env python3
"""
Location Job Scorer

This module implements job scoring based on geographic distance between
user location and job location using latitude and longitude coordinates.

The scoring algorithm uses the Haversine formula to calculate distances
and provides higher scores for closer jobs with configurable distance thresholds.


"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import math
from .base_scorer import BaseJobScorer, ScoringValidationError, ScoringApplicationError


class LocationScorer(BaseJobScorer):
    """
    Scorer that evaluates jobs based on geographic distance from user location.
    
    Scoring Logic:
    - Distance = 0: Score = 1.0 (perfect match)
    - Distance <= preferred_distance: Score = 0.8 to 1.0 (linear decay)
    - Distance <= max_distance: Score = 0.2 to 0.8 (linear decay)
    - Distance > max_distance: Score = 0.0 to 0.2 (exponential decay)
    """
    
    def __init__(self, 
                 preferred_distance_km: float = 25.0,
                 max_distance_km: float = 100.0,
                 distance_unit: str = 'km'):
        """
        Initialize the Location Scorer.
        
        Args:
            preferred_distance_km: Distance within which jobs get high scores (default: 25km)
            max_distance_km: Maximum reasonable distance for jobs (default: 100km)
            distance_unit: Unit for distance calculations ('km' or 'miles')
        """
        super().__init__("Location")
        self.preferred_distance = preferred_distance_km
        self.max_distance = max_distance_km
        self.distance_unit = distance_unit.lower()
        
        # Convert to miles if needed
        if self.distance_unit == 'miles':
            self.preferred_distance *= 0.621371  # km to miles
            self.max_distance *= 0.621371
    
    def get_required_columns(self) -> List[str]:
        """
        Get required columns for location scoring.
        
        Returns:
            List of required column names
        """
        return ['latitude', 'longitude']
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user criteria for location scoring.
        
        Args:
            criteria: User criteria containing 'latitude' and 'longitude'
            
        Returns:
            Dict with validation results
        """
        errors = []
        
        # Check for required fields
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in criteria:
                errors.append(f"Missing '{field}' in criteria")
        
        if not errors:  # Only validate values if fields exist
            try:
                lat = float(criteria['latitude'])
                lon = float(criteria['longitude'])
                
                # Validate latitude range
                if not -90 <= lat <= 90:
                    errors.append("Latitude must be between -90 and 90 degrees")
                
                # Validate longitude range
                if not -180 <= lon <= 180:
                    errors.append("Longitude must be between -180 and 180 degrees")
                    
            except (ValueError, TypeError):
                errors.append("Latitude and longitude must be numbers")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def calculate_score(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> pd.Series:
        """
        Calculate location scores for jobs based on distance.
        
        Args:
            jobs_df: DataFrame with job data including latitude, longitude
            user_criteria: Dict containing 'latitude' and 'longitude'
            
        Returns:
            pd.Series: Location scores (0.0 to 1.0)
        """
        # Validate inputs
        self.validate_dataframe(jobs_df)
        validation = self.validate_criteria(user_criteria)
        
        if not validation['valid']:
            raise ScoringValidationError(f"Invalid criteria: {validation['errors']}")
        
        if jobs_df.empty:
            return pd.Series(dtype=float)
        
        try:
            user_lat = float(user_criteria['latitude'])
            user_lon = float(user_criteria['longitude'])
            
            # Calculate distances
            distances = self._calculate_distances(
                user_lat, user_lon,
                jobs_df['latitude'], jobs_df['longitude']
            )
            
            # Calculate scores based on distances
            scores = self._calculate_distance_scores(distances)
            
            # Log scoring information
            self.log_scoring_info(len(jobs_df), user_criteria, scores)
            
            return scores
            
        except Exception as e:
            raise ScoringApplicationError(f"Failed to calculate location scores: {str(e)}")
    
    def _calculate_distances(self, user_lat: float, user_lon: float,
                           job_lats: pd.Series, job_lons: pd.Series) -> pd.Series:
        """
        Calculate distances using the Haversine formula.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            job_lats: Job latitudes
            job_lons: Job longitudes
            
        Returns:
            pd.Series: Distances in kilometers (or miles if specified)
        """
        # Convert to radians
        user_lat_rad = math.radians(user_lat)
        user_lon_rad = math.radians(user_lon)
        job_lats_rad = np.radians(job_lats)
        job_lons_rad = np.radians(job_lons)
        
        # Haversine formula
        dlat = job_lats_rad - user_lat_rad
        dlon = job_lons_rad - user_lon_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(user_lat_rad) * np.cos(job_lats_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius_km = 6371.0
        
        distances = earth_radius_km * c
        
        # Convert to miles if needed
        if self.distance_unit == 'miles':
            distances *= 0.621371
        
        return distances
    
    def _calculate_distance_scores(self, distances: pd.Series) -> pd.Series:
        """
        Calculate scores based on distances.
        
        Args:
            distances: Distances in km or miles
            
        Returns:
            pd.Series: Location scores
        """
        scores = pd.Series(index=distances.index, dtype=float)
        
        # Perfect match (distance = 0)
        perfect_match = (distances == 0)
        scores[perfect_match] = 1.0
        
        # Within preferred distance (high scores)
        within_preferred = (distances > 0) & (distances <= self.preferred_distance)
        if within_preferred.any():
            # Linear decay from 1.0 to 0.8
            preferred_scores = 1.0 - 0.2 * (distances[within_preferred] / self.preferred_distance)
            scores[within_preferred] = preferred_scores
        
        # Within max distance (medium scores)
        within_max = (distances > self.preferred_distance) & (distances <= self.max_distance)
        if within_max.any():
            # Linear decay from 0.8 to 0.2
            distance_ratio = ((distances[within_max] - self.preferred_distance) / 
                            (self.max_distance - self.preferred_distance))
            max_scores = 0.8 - 0.6 * distance_ratio
            scores[within_max] = max_scores
        
        # Beyond max distance (low scores with exponential decay)
        beyond_max = (distances > self.max_distance)
        if beyond_max.any():
            # Exponential decay from 0.2 to near 0
            excess_distance = distances[beyond_max] - self.max_distance
            decay_factor = np.exp(-excess_distance / (self.max_distance * 0.5))
            beyond_scores = 0.2 * decay_factor
            scores[beyond_max] = beyond_scores
        
        return scores
    
    def get_distance_category(self, distance: float) -> str:
        """
        Categorize distance for a single job.
        
        Args:
            distance: Distance in km or miles
            
        Returns:
            str: Distance category
        """
        if distance == 0:
            return 'same_location'
        elif distance <= self.preferred_distance:
            return 'preferred_distance'
        elif distance <= self.max_distance:
            return 'acceptable_distance'
        else:
            return 'far_distance'
    
    def analyze_location_distribution(self, jobs_df: pd.DataFrame, 
                                    user_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the distribution of job locations and distances.
        
        Args:
            jobs_df: DataFrame with job data
            user_criteria: User criteria with location
            
        Returns:
            Dict with analysis results
        """
        if jobs_df.empty:
            return {'error': 'No jobs to analyze'}
        
        validation = self.validate_criteria(user_criteria)
        if not validation['valid']:
            return {'error': f"Invalid criteria: {validation['errors']}"}
        
        user_lat = float(user_criteria['latitude'])
        user_lon = float(user_criteria['longitude'])
        
        # Calculate distances
        distances = self._calculate_distances(
            user_lat, user_lon,
            jobs_df['latitude'], jobs_df['longitude']
        )
        
        # Categorize distances
        categories = []
        for distance in distances:
            category = self.get_distance_category(distance)
            categories.append(category)
        
        category_counts = pd.Series(categories).value_counts()
        
        return {
            'user_location': {
                'latitude': user_lat,
                'longitude': user_lon
            },
            'total_jobs': len(jobs_df),
            'distance_statistics': {
                'mean_distance': float(distances.mean()),
                'median_distance': float(distances.median()),
                'min_distance': float(distances.min()),
                'max_distance': float(distances.max()),
                'std_distance': float(distances.std())
            },
            'distance_thresholds': {
                'preferred_distance': self.preferred_distance,
                'max_distance': self.max_distance,
                'distance_unit': self.distance_unit
            },
            'distance_distribution': {
                'same_location': int(category_counts.get('same_location', 0)),
                'preferred_distance': int(category_counts.get('preferred_distance', 0)),
                'acceptable_distance': int(category_counts.get('acceptable_distance', 0)),
                'far_distance': int(category_counts.get('far_distance', 0))
            },
            'distance_percentages': {
                'same_location': float(category_counts.get('same_location', 0) / len(jobs_df) * 100),
                'preferred_distance': float(category_counts.get('preferred_distance', 0) / len(jobs_df) * 100),
                'acceptable_distance': float(category_counts.get('acceptable_distance', 0) / len(jobs_df) * 100),
                'far_distance': float(category_counts.get('far_distance', 0) / len(jobs_df) * 100)
            }
        }
    
    def find_jobs_within_radius(self, jobs_df: pd.DataFrame, 
                               user_criteria: Dict[str, Any],
                               radius_km: float) -> pd.DataFrame:
        """
        Find all jobs within a specified radius.
        
        Args:
            jobs_df: DataFrame with job data
            user_criteria: User criteria with location
            radius_km: Radius in kilometers
            
        Returns:
            pd.DataFrame: Jobs within the specified radius
        """
        validation = self.validate_criteria(user_criteria)
        if not validation['valid']:
            raise ScoringValidationError(f"Invalid criteria: {validation['errors']}")
        
        if jobs_df.empty:
            return jobs_df
        
        user_lat = float(user_criteria['latitude'])
        user_lon = float(user_criteria['longitude'])
        
        distances = self._calculate_distances(
            user_lat, user_lon,
            jobs_df['latitude'], jobs_df['longitude']
        )
        
        # Convert radius to appropriate unit
        if self.distance_unit == 'miles':
            radius_km *= 0.621371
        
        within_radius = distances <= radius_km
        result_df = jobs_df[within_radius].copy()
        result_df['distance'] = distances[within_radius]
        
        return result_df.sort_values('distance')


def score_by_location(jobs_df: pd.DataFrame, 
                     user_latitude: float,
                     user_longitude: float,
                     preferred_distance_km: float = 25.0,
                     max_distance_km: float = 100.0) -> pd.Series:
    """
    Convenience function to score jobs by location.
    
    Args:
        jobs_df: DataFrame with job data
        user_latitude: User's latitude
        user_longitude: User's longitude
        preferred_distance_km: Preferred distance threshold
        max_distance_km: Maximum distance threshold
        
    Returns:
        pd.Series: Location scores
    """
    scorer = LocationScorer(
        preferred_distance_km=preferred_distance_km,
        max_distance_km=max_distance_km
    )
    
    criteria = {
        'latitude': user_latitude,
        'longitude': user_longitude
    }
    
    return scorer.calculate_score(jobs_df, criteria)