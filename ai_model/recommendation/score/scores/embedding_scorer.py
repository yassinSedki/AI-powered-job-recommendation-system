#!/usr/bin/env python3
"""
Embedding Similarity Scorer

This module provides embedding-based similarity scoring for job recommendations.
It uses pre-computed embeddings to calculate cosine similarity between user queries
and job descriptions.


"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .base_scorer import BaseJobScorer, ScoringError, ScoringValidationError

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingScorerError(ScoringError):
    """Exception raised for embedding scorer specific errors."""
    pass


class EmbeddingLoader:
    """Utility class for loading and managing pre-computed embeddings."""
    
    def __init__(self, embeddings_dir: str):
        """
        Initialize the embedding loader.
        
        Args:
            embeddings_dir: Directory containing embedding files
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.job_ids = None
        self.metadata = None
        self._is_loaded = False
    
    def load_embeddings(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Load pre-computed embeddings from files.
        
        Returns:
            Tuple of (embeddings_array, job_ids_list, metadata_dict)
        
        Raises:
            EmbeddingScorerError: If embedding files cannot be loaded
        """
        try:
            # Load embeddings
            embeddings_file = self.embeddings_dir / "job_embeddings.npy"
            if not embeddings_file.exists():
                raise EmbeddingScorerError(f"Embeddings file not found: {embeddings_file}")
            
            self.embeddings = np.load(embeddings_file)
            logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
            
            # Load job IDs
            job_ids_file = self.embeddings_dir / "job_ids.pkl"
            if not job_ids_file.exists():
                raise EmbeddingScorerError(f"Job IDs file not found: {job_ids_file}")
            
            with open(job_ids_file, 'rb') as f:
                self.job_ids = pickle.load(f)
            logger.info(f"Loaded {len(self.job_ids)} job IDs")
            
            # Load metadata
            metadata_file = self.embeddings_dir / "embedding_metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata: {self.metadata.get('provider_info', {}).get('model', 'unknown')}")
            else:
                self.metadata = {}
                logger.warning("Metadata file not found, using empty metadata")
            
            # Validate alignment
            if len(self.embeddings) != len(self.job_ids):
                raise EmbeddingScorerError(
                    f"Mismatch between embeddings ({len(self.embeddings)}) and job IDs ({len(self.job_ids)})"
                )
            
            self._is_loaded = True
            return self.embeddings, self.job_ids, self.metadata
            
        except Exception as e:
            raise EmbeddingScorerError(f"Failed to load embeddings: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if embeddings are loaded."""
        return self._is_loaded
    
    def get_embedding_by_job_id(self, job_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific job ID.
        
        Args:
            job_id: The job ID to get embedding for
            
        Returns:
            Embedding vector or None if not found
        """
        if not self._is_loaded:
            raise EmbeddingScorerError("Embeddings not loaded. Call load_embeddings() first.")
        
        try:
            job_index = self.job_ids.index(job_id)
            return self.embeddings[job_index]
        except ValueError:
            return None


class EmbeddingSimilarityScorer(BaseJobScorer):
    """
    Embedding-based similarity scorer for job recommendations.
    
    This scorer calculates cosine similarity between a user query embedding
    and pre-computed job embeddings to provide similarity scores.
    """
    
    def __init__(self, embeddings_dir: str, name: str = "embedding_similarity"):
        """
        Initialize the embedding similarity scorer.
        
        Args:
            embeddings_dir: Directory containing pre-computed embeddings
            name: Name of the scorer
        """
        super().__init__(name)
        self.embeddings_dir = embeddings_dir
        self.embedding_loader = EmbeddingLoader(embeddings_dir)
        self._embeddings_loaded = False
    
    def load_embeddings(self) -> None:
        """Load pre-computed embeddings."""
        if not self._embeddings_loaded:
            self.embedding_loader.load_embeddings()
            self._embeddings_loaded = True
            logger.info("Embeddings loaded successfully for similarity scoring")
    
    def calculate_similarity(self, query_embedding: np.ndarray, 
                           job_embeddings: np.ndarray = None) -> np.ndarray:
        """
        Calculate cosine similarity between query and job embeddings.
        
        Args:
            query_embedding: User query embedding vector
            job_embeddings: Job embeddings matrix (optional, uses loaded if None)
            
        Returns:
            Array of similarity scores
        """
        if job_embeddings is None:
            if not self._embeddings_loaded:
                self.load_embeddings()
            job_embeddings = self.embedding_loader.embeddings
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, job_embeddings)
        return similarities.flatten()
    
    def find_top_similar_jobs(self, query_embedding: np.ndarray, 
                             top_k: int = 10) -> Tuple[List[str], np.ndarray]:
        """
        Find top-k most similar jobs to the query.
        
        Args:
            query_embedding: User query embedding vector
            top_k: Number of top similar jobs to return
            
        Returns:
            Tuple of (job_ids, similarity_scores)
        """
        if not self._embeddings_loaded:
            self.load_embeddings()
        
        # Calculate similarities
        similarities = self.calculate_similarity(query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Get corresponding job IDs and scores
        top_job_ids = [self.embedding_loader.job_ids[i] for i in top_indices]
        top_scores = similarities[top_indices]
        
        return top_job_ids, top_scores
    
    def calculate_score(self, jobs_df: pd.DataFrame, user_criteria: Dict[str, Any]) -> pd.Series:
        """
        Calculate similarity scores for jobs based on user query embedding.
        
        Args:
            jobs_df: DataFrame containing job data
            user_criteria: Dictionary containing 'query_embedding' key
            
        Returns:
            Series of similarity scores for each job
        """
        # Validate input
        self.validate_dataframe(jobs_df)
        validated_criteria = self.validate_criteria(user_criteria)
        
        query_embedding = validated_criteria['query_embedding']
        
        # Load embeddings if not already loaded
        if not self._embeddings_loaded:
            self.load_embeddings()
        
        # Create job ID to index mapping (normalize IDs to string to avoid type mismatches)
        job_id_to_index = {str(job_id): idx for idx, job_id in enumerate(self.embedding_loader.job_ids)}
        
        # Initialize scores array
        scores = np.zeros(len(jobs_df))
        
        # Calculate scores for jobs that have embeddings
        matched_count = 0
        for i, job_id in enumerate(jobs_df['Job_Id']):
            key = str(job_id)
            if key in job_id_to_index:
                matched_count += 1
                job_index = job_id_to_index[key]
                job_embedding = self.embedding_loader.embeddings[job_index].reshape(1, -1)
                similarity = cosine_similarity(query_embedding.reshape(1, -1), job_embedding)[0, 0]
                scores[i] = max(0.0, similarity)  # Ensure non-negative scores
            else:
                scores[i] = 0.0  # No embedding available
        
        # Convert to pandas Series
        score_series = pd.Series(scores, index=jobs_df.index)
        
        # Log scoring info
        self.log_scoring_info(len(jobs_df), user_criteria, score_series)
        try:
            total = len(jobs_df)
            self.logger.info(f"Embedding scorer matched {matched_count}/{total} job IDs ({(matched_count/total)*100:.1f}%)")
        except Exception:
            pass
        
        return score_series
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user criteria for embedding similarity scoring.
        
        Args:
            criteria: User criteria dictionary
            
        Returns:
            Validated criteria dictionary
            
        Raises:
            ScoringValidationError: If validation fails
        """
        if not isinstance(criteria, dict):
            raise ScoringValidationError("Criteria must be a dictionary")
        
        if 'query_embedding' not in criteria:
            raise ScoringValidationError("Missing required 'query_embedding' in criteria")
        
        query_embedding = criteria['query_embedding']
        
        if not isinstance(query_embedding, np.ndarray):
            raise ScoringValidationError("query_embedding must be a numpy array")
        
        if query_embedding.ndim not in [1, 2]:
            raise ScoringValidationError("query_embedding must be 1D or 2D array")
        
        if query_embedding.ndim == 2 and query_embedding.shape[0] != 1:
            raise ScoringValidationError("2D query_embedding must have shape (1, n_features)")
        
        # Validate embedding dimension if metadata is available
        if self._embeddings_loaded and self.embedding_loader.metadata:
            expected_dim = self.embedding_loader.metadata.get('embedding_dim')
            actual_dim = query_embedding.shape[-1]
            if expected_dim and actual_dim != expected_dim:
                raise ScoringValidationError(
                    f"Query embedding dimension ({actual_dim}) doesn't match "
                    f"expected dimension ({expected_dim})"
                )
        
        return criteria.copy()
    
    def get_required_columns(self) -> List[str]:
        """
        Get required columns for embedding similarity scoring.
        
        Returns:
            List of required column names
        """
        return ['Job_Id']
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about loaded embeddings.
        
        Returns:
            Dictionary with embedding information
        """
        if not self._embeddings_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "num_embeddings": len(self.embedding_loader.embeddings),
            "embedding_dimension": self.embedding_loader.embeddings.shape[1],
            "metadata": self.embedding_loader.metadata
        }