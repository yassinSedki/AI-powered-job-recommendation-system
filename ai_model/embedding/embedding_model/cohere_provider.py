"""Cohere Embedding Provider

This module implements the Cohere embedding provider following the abstract interface.
"""

import numpy as np
import logging
from typing import List, Dict, Any
import cohere

from .base_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class CohereProvider(EmbeddingProvider):
    """Cohere embedding provider implementation."""
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        'embed-english-v3.0': 1024,
        'embed-english-light-v3.0': 384,
        'embed-multilingual-v3.0': 1024,
        'embed-multilingual-light-v3.0': 384,
        'embed-english-v2.0': 4096,
        'embed-english-light-v2.0': 1024,
        'embed-multilingual-v2.0': 768
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Cohere provider.
        
        Args:
            config: Configuration dictionary with 'api_key', 'model', etc.
        """
        super().__init__(config)
        self._client = None
        self._initialize_client()
    
    def _validate_config(self) -> None:
        """Validate Cohere-specific configuration."""
        required_fields = ['api_key', 'model']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate model
        model = self.config['model']
        if model not in self.MODEL_DIMENSIONS:
            available_models = list(self.MODEL_DIMENSIONS.keys())
            raise ValueError(f"Unsupported Cohere model: {model}. Available models: {available_models}")
        
        # Set default batch size if not provided
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
    
    def _initialize_client(self) -> None:
        """Initialize the Cohere client."""
        try:
            self._client = cohere.Client(self.config['api_key'])
            logger.info(f"Initialized Cohere client with model: {self.config['model']}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise RuntimeError(f"Cohere client initialization failed: {e}")
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text using Cohere.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding generation fails
        """
        # Validate and clean text
        cleaned_text = self.validate_text(text)
        
        try:
            # Generate embedding
            response = self._client.embed(
                texts=[cleaned_text],
                model=self.config['model'],
                input_type='search_document'
            )
            
            # Extract embedding
            embedding = np.array(response.embeddings[0])
            logger.debug(f"Generated embedding with shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Cohere embedding generation failed: {e}")
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Create embeddings for multiple texts using Cohere.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            np.ndarray: Array of embeddings (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts list is invalid
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Validate and clean texts
        cleaned_texts = self.validate_texts(texts)
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(cleaned_texts), batch_size):
                batch_texts = cleaned_texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                
                # Generate embeddings for batch
                response = self._client.embed(
                    texts=batch_texts,
                    model=self.config['model'],
                    input_type='search_document'
                )
                
                # Convert to numpy array
                batch_embeddings = np.array(response.embeddings)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Cohere batch embedding generation failed: {e}")
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "cohere"
    
    @property
    def model_name(self) -> str:
        """Return the current model name."""
        return self.config['model']
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        return self.MODEL_DIMENSIONS[self.config['model']]
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Cohere models.
        
        Returns:
            List of model names
        """
        return list(self.MODEL_DIMENSIONS.keys())
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Model name (uses current model if None)
            
        Returns:
            Dict containing model information
        """
        if model_name is None:
            model_name = self.model_name
        
        if model_name not in self.MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model: {model_name}")
        
        return {
            "model_name": model_name,
            "provider": self.provider_name,
            "embedding_dimension": self.MODEL_DIMENSIONS[model_name],
            "input_type": "search_document"
        }