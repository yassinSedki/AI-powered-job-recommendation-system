"""Core Embedding Functions with Dependency Injection

This module provides high-level embedding functions that work with any provider
through dependency injection, promoting maintainability and testability.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime

from .base_provider import EmbeddingProvider
from .config import get_config
from .cohere_provider import CohereProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service class for embedding operations with dependency injection."""
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """
        Initialize embedding service.
        
        Args:
            provider: Embedding provider instance (auto-detected if None)
        """
        self.provider = provider or self._create_default_provider()
        logger.info(f"Initialized EmbeddingService with {self.provider.provider_name} provider")
    
    def _create_default_provider(self) -> EmbeddingProvider:
        """Create default provider based on configuration."""
        config = get_config()
        
        try:
            default_provider_name = config.get_default_provider()
            provider_config = config.get_provider_config(default_provider_name)
            
            # Create provider based on name
            if default_provider_name == 'cohere':
                return CohereProvider(provider_config)
            else:
                raise ValueError(f"Unsupported provider: {default_provider_name}")
                
        except Exception as e:
            logger.error(f"Failed to create default provider: {e}")
            raise RuntimeError(f"Provider initialization failed: {e}")
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding generation fails
        """
        return self.provider.create_embedding(text)
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts list is invalid
            RuntimeError: If embedding generation fails
        """
        return self.provider.create_embeddings_batch(texts, batch_size)
    
    def process_dataset(self,
                       jobs_data: Union[List[Dict], pd.DataFrame],
                       text_col: str = 'combined_text',
                       job_id_col: str = 'Job_Id',
                       batch_size: int = None,
                       output_dir: str = "embeddings") -> Dict[str, str]:
        """
        Process a complete dataset and save embeddings with metadata.
        
        Args:
            jobs_data: Job data as DataFrame or list of dicts
            text_col: Column name containing text to embed
            job_id_col: Column name containing job IDs
            batch_size: Batch size for processing
            output_dir: Directory to save embeddings
            
        Returns:
            Dict with paths to saved files
            
        Raises:
            ValueError: If required columns are missing
            RuntimeError: If processing fails
        """
        # Convert to DataFrame if needed
        if isinstance(jobs_data, list):
            df = pd.DataFrame(jobs_data)
        else:
            df = jobs_data.copy()
        
        # Validate required columns
        if job_id_col not in df.columns:
            raise ValueError(f"Job ID column '{job_id_col}' not found")
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found")
        
        # Extract data
        job_ids = df[job_id_col].tolist()
        texts = df[text_col].tolist()
        
        logger.info(f"Processing {len(job_ids)} jobs with {self.provider.provider_name} provider...")
        
        # Generate embeddings
        embeddings = self.create_embeddings_batch(texts, batch_size)
        
        # Validate alignment
        if len(embeddings) != len(job_ids):
            raise RuntimeError("Mismatch between embeddings and job IDs")
        
        # Save embeddings and metadata
        return save_embeddings(
            embeddings=embeddings,
            job_ids=job_ids,
            provider_info=self.provider.get_provider_info(),
            output_dir=output_dir
        )
    
    def get_provider_info(self) -> Dict[str, any]:
        """Get information about the current provider."""
        return self.provider.get_provider_info()


# Global service instance
_global_service = None


def get_embedding_service(provider: Optional[EmbeddingProvider] = None) -> EmbeddingService:
    """
    Get global embedding service instance.
    
    Args:
        provider: Embedding provider instance (optional)
        
    Returns:
        EmbeddingService instance
    """
    global _global_service
    
    if _global_service is None or provider is not None:
        _global_service = EmbeddingService(provider)
    
    return _global_service


def reset_service() -> None:
    """Reset global service (useful for testing)."""
    global _global_service
    _global_service = None


# Convenience functions that use the global service
def create_embedding(text: str, provider: Optional[EmbeddingProvider] = None) -> np.ndarray:
    """
    Create embedding for a single text using the global service.
    
    Args:
        text: Input text to embed
        provider: Optional provider (uses global service provider if None)
        
    Returns:
        np.ndarray: Embedding vector
    """
    service = get_embedding_service(provider)
    return service.create_embedding(text)


def create_embeddings_batch(texts: List[str],
                           batch_size: int = None,
                           provider: Optional[EmbeddingProvider] = None) -> np.ndarray:
    """
    Create embeddings for multiple texts using the global service.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for processing
        provider: Optional provider (uses global service provider if None)
        
    Returns:
        np.ndarray: Array of embeddings (n_texts, embedding_dim)
    """
    service = get_embedding_service(provider)
    return service.create_embeddings_batch(texts, batch_size)


def process_dataset(jobs_data: Union[List[Dict], pd.DataFrame],
                   text_col: str = 'combined_text',
                   job_id_col: str = 'Job_Id',
                   batch_size: int = None,
                   output_dir: str = "embeddings",
                   provider: Optional[EmbeddingProvider] = None) -> Dict[str, str]:
    """
    Process a complete dataset using the global service.
    
    Args:
        jobs_data: Job data as DataFrame or list of dicts
        text_col: Column name containing text to embed
        job_id_col: Column name containing job IDs
        batch_size: Batch size for processing
        output_dir: Directory to save embeddings
        provider: Optional provider (uses global service provider if None)
        
    Returns:
        Dict with paths to saved files
    """
    service = get_embedding_service(provider)
    return service.process_dataset(jobs_data, text_col, job_id_col, batch_size, output_dir)


def save_embeddings(embeddings: np.ndarray,
                   job_ids: List,
                   provider_info: Dict[str, any],
                   output_dir: str = "embeddings",
                   embeddings_filename: str = "job_embeddings.npy",
                   job_ids_filename: str = "job_ids.pkl",
                   metadata_filename: str = "embedding_metadata.json") -> Dict[str, str]:
    """
    Save embeddings and job IDs to files with metadata.
    
    Args:
        embeddings: Embedding array
        job_ids: List of job IDs
        provider_info: Information about the provider used
        output_dir: Directory to save files
        embeddings_filename: Name for embeddings file
        job_ids_filename: Name for job IDs file
        metadata_filename: Name for metadata file
        
    Returns:
        Dict with paths to saved files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # File paths
    embeddings_path = output_path / embeddings_filename
    job_ids_path = output_path / job_ids_filename
    metadata_path = output_path / metadata_filename
    
    try:
        # Save embeddings
        np.save(embeddings_path, embeddings)
        logger.info(f"✅ Embeddings saved to: {embeddings_path}")
        
        # Save job IDs
        with open(job_ids_path, 'wb') as f:
            pickle.dump(job_ids, f)
        logger.info(f"✅ Job IDs saved to: {job_ids_path}")
        
        # Save metadata
        metadata = {
            "provider_info": provider_info,
            "embedding_dim": int(embeddings.shape[1]),
            "num_jobs": int(embeddings.shape[0]),
            "created_at": datetime.now().isoformat(),
            "embeddings_file": embeddings_filename,
            "job_ids_file": job_ids_filename
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved to: {metadata_path}")
        
        return {
            "embeddings": str(embeddings_path),
            "job_ids": str(job_ids_path),
            "metadata": str(metadata_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to save files: {e}")
        raise


def load_embeddings(embeddings_path: str,
                   job_ids_path: str) -> tuple[np.ndarray, List]:
    """
    Load embeddings and job IDs from files.
    
    Args:
        embeddings_path: Path to embeddings file
        job_ids_path: Path to job IDs file
        
    Returns:
        Tuple of (embeddings, job_ids)
    """
    try:
        # Load embeddings
        embeddings = np.load(embeddings_path)
        logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")
        
        # Load job IDs
        with open(job_ids_path, 'rb') as f:
            job_ids = pickle.load(f)
        logger.info(f"✅ Loaded {len(job_ids)} job IDs")
        
        # Validate alignment
        if len(embeddings) != len(job_ids):
            raise ValueError("Mismatch between embeddings and job IDs")
        
        return embeddings, job_ids
        
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        raise