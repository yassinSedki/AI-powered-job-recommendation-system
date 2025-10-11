"""Abstract Base Class for Embedding Providers

This module defines the interface that all embedding providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding provider with configuration.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration for this provider."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
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
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the current model name."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dict containing provider information
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "config": {k: v for k, v in self.config.items() if 'key' not in k.lower()}  # Exclude API keys
        }
    
    def validate_text(self, text: str) -> str:
        """
        Validate and clean input text.
        
        Args:
            text: Input text to validate
            
        Returns:
            str: Cleaned text
            
        Raises:
            ValueError: If text is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text cannot be empty or whitespace only")
        
        return cleaned_text
    
    def validate_texts(self, texts: List[str]) -> List[str]:
        """
        Validate and clean a list of texts.
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List[str]: List of cleaned texts
            
        Raises:
            ValueError: If texts list is invalid
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        cleaned_texts = []
        for i, text in enumerate(texts):
            try:
                cleaned_text = self.validate_text(text)
                cleaned_texts.append(cleaned_text)
            except ValueError:
                # Use placeholder for invalid texts
                cleaned_texts.append("no description available")
        
        return cleaned_texts