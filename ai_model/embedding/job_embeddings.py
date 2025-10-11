"""Job Embeddings Module

Simplified interface for job embeddings using the core embedding system.
This module provides job-specific convenience functions without redundant wrappers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from .embedding_model import (
    EmbeddingService, 
    get_embedding_service,
    EmbeddingProvider,
    CohereProvider,
    get_config
)


# Convenience functions for job embeddings - using core service directly
def create_job_embedding(text: str, 
                        provider: Optional[EmbeddingProvider] = None) -> np.ndarray:
    """
    Create embedding for a single job description.
    
    Args:
        text: Job description text
        provider: Embedding provider (optional)
        
    Returns:
        np.ndarray: Embedding vector
    """
    service = get_embedding_service(provider)
    return service.create_embedding(text)


def create_job_embeddings_batch(texts: List[str],
                               batch_size: int = None,
                               provider: Optional[EmbeddingProvider] = None) -> np.ndarray:
    """
    Create embeddings for multiple job descriptions.
    
    Args:
        texts: List of job description texts
        batch_size: Batch size for processing
        provider: Embedding provider (optional)
        
    Returns:
        np.ndarray: Array of embeddings
    """
    service = get_embedding_service(provider)
    return service.create_embeddings_batch(texts, batch_size)


def process_job_dataset(jobs_data: Union[List[Dict], pd.DataFrame],
                       text_col: str = 'combined_text',
                       job_id_col: str = 'Job_Id',
                       batch_size: int = None,
                       output_dir: str = "embeddings",
                       provider: Optional[EmbeddingProvider] = None) -> Dict[str, str]:
    """
    Process a complete job dataset and save embeddings.
    
    Args:
        jobs_data: Job data as DataFrame or list of dicts
        text_col: Column name containing job descriptions
        job_id_col: Column name containing job IDs
        batch_size: Batch size for processing
        output_dir: Directory to save embeddings
        provider: Embedding provider (optional)
        
    Returns:
        Dict with paths to saved files
    """
    service = get_embedding_service(provider)
    return service.process_dataset(jobs_data, text_col, job_id_col, batch_size, output_dir)


# Legacy functions for backward compatibility (deprecated)
def create_job_embedding_legacy(text: str, 
                               model_name: str = 'embed-english-v3.0',
                               api_key: Optional[str] = None) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    
    DEPRECATED: Use create_job_embedding() instead.
    """
    import warnings
    warnings.warn(
        "create_job_embedding_legacy is deprecated. Use create_job_embedding() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create provider with legacy parameters
    config = get_config()
    provider_config = {
        'api_key': api_key or config.get_provider_config('cohere')['api_key'],
        'model': model_name,
        'batch_size': 32
    }
    provider = CohereProvider(provider_config)
    
    return create_job_embedding(text, provider)