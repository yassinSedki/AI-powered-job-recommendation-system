"""Embedding Module

Provides embedding functionality with multiple provider support.
Simplified architecture with direct service usage.
"""

from .embedding_model import (
    EmbeddingService,
    EmbeddingProvider, 
    CohereProvider,
    EmbeddingConfig,
    get_config,
    get_embedding_service,
    create_embedding,
    create_embeddings_batch,
    process_dataset,
    save_embeddings,
    load_embeddings
)

from .job_embeddings import (
    create_job_embedding,
    create_job_embeddings_batch,
    process_job_dataset
)

__all__ = [
    # Core service and providers
    'EmbeddingService',
    'EmbeddingProvider',
    'CohereProvider',
    'EmbeddingConfig',
    'get_config',
    'get_embedding_service',
    
    # Core convenience functions
    'create_embedding',
    'create_embeddings_batch', 
    'process_dataset',
    'save_embeddings',
    'load_embeddings',
    
    # Job-specific convenience functions
    'create_job_embedding',
    'create_job_embeddings_batch',
    'process_job_dataset'
]