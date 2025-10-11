"""
Embedding Model Package

This package provides a flexible embedding system with support for multiple providers
through dependency injection, promoting maintainability and extensibility.

Key Features:
- Provider-agnostic architecture (Cohere, OpenAI, HuggingFace support)
- Dependency injection for easy testing and provider switching
- Secure configuration management
- Single and batch processing
- Dataset processing with metadata
- File I/O for embeddings

Architecture:
- EmbeddingService: Main service class with dependency injection
- EmbeddingProvider: Abstract base class for providers
- CohereProvider: Cohere implementation
- EmbeddingConfig: Configuration management
"""

# Core service and provider classes
from .EmbeddingService import (
    EmbeddingService,
    get_embedding_service,
    reset_service
)

from .base_provider import EmbeddingProvider
from .cohere_provider import CohereProvider
from .config import EmbeddingConfig, get_config, reset_config

# Convenience functions (backward compatibility)
from .EmbeddingService import (
    create_embedding,
    create_embeddings_batch,
    process_dataset,
    save_embeddings,
    load_embeddings
)

__all__ = [
    # Service classes (recommended for new code)
    'EmbeddingService',
    'get_embedding_service',
    'reset_service',
    
    # Provider classes
    'EmbeddingProvider',
    'CohereProvider',
    
    # Configuration
    'EmbeddingConfig',
    'get_config',
    'reset_config',
    
    # Convenience functions (backward compatibility)
    'create_embedding',
    'create_embeddings_batch',
    'process_dataset',
    'save_embeddings',
    'load_embeddings'
]