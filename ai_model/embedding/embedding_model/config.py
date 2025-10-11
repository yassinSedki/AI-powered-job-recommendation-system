"""Configuration Management for Embedding System

This module handles configuration loading, validation, and management for embedding providers.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in project root (4 levels up from this file)
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.getLogger(__name__).info(f"Loaded environment variables from {env_file}")
    else:
        # Try loading from current directory as fallback
        load_dotenv()
        logging.getLogger(__name__).info("Attempted to load .env from current directory")
except ImportError:
    logging.getLogger(__name__).warning("python-dotenv not installed. Environment variables from .env file will not be loaded automatically.")

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    """Configuration manager for embedding system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        config = {}
        
        # Load from file if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_from_env()
        config.update(env_config)
        
        return config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Cohere configuration
        if os.getenv('COHERE_API_KEY'):
            env_config['cohere'] = {
                'api_key': os.getenv('COHERE_API_KEY'),
                'model': os.getenv('COHERE_MODEL', 'embed-english-v3.0'),
                'batch_size': int(os.getenv('COHERE_BATCH_SIZE', '32'))
            }
        
       
        
        # General settings
        # env_config['general'] = {
        #     'default_provider': os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'cohere'),
        #     'default_output_dir': os.getenv('DEFAULT_EMBEDDINGS_OUTPUT_DIR', 'embeddings'),
        #     'default_text_column': os.getenv('DEFAULT_TEXT_COLUMN', 'combined_text'),
        #     'default_job_id_column': os.getenv('DEFAULT_JOB_ID_COLUMN', 'Job_Id'),
        #     'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        #     'debug_mode': os.getenv('EMBEDDING_DEBUG', 'false').lower() == 'true'
        # }
        
        return env_config
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'cohere', 'openai', 'huggingface')
            
        Returns:
            Dict containing provider configuration
            
        Raises:
            ValueError: If provider is not configured
        """
        if provider not in self._config:
            raise ValueError(f"Provider '{provider}' is not configured. Available providers: {list(self._config.keys())}")
        
        return self._config[provider].copy()
    
    def get_default_provider(self) -> str:
        """
        Get the default provider name.
        
        Returns:
            str: Default provider name
            
        Raises:
            ValueError: If no providers are configured
        """
        # Check if default provider is set in general config
        if 'general' in self._config and 'default_provider' in self._config['general']:
            default_provider = self._config['general']['default_provider']
            if default_provider in self._config:
                return default_provider
        
        if not self._config:
            raise ValueError("No embedding providers are configured")
        
        # Priority order: cohere, openai, huggingface
        for provider in ['cohere', 'openai', 'huggingface']:
            if provider in self._config:
                return provider
        
        # Return first available provider
        return list(self._config.keys())[0]
    
    def list_available_providers(self) -> list[str]:
        """
        List all available providers.
        
        Returns:
            List of provider names
        """
        return list(self._config.keys())
    
    def validate_provider_config(self, provider: str) -> bool:
        """
        Validate configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            bool: True if configuration is valid
        """
        try:
            config = self.get_provider_config(provider)
            
            # Check required fields
            if 'api_key' not in config:
                logger.error(f"Missing API key for provider '{provider}'")
                return False
            
            if 'model' not in config:
                logger.error(f"Missing model for provider '{provider}'")
                return False
            
            return True
            
        except ValueError:
            return False
    
    def save_config(self, output_file: str) -> None:
        """
        Save current configuration to file (excluding API keys).
        
        Args:
            output_file: Path to save configuration
        """
        # Create safe config without API keys
        safe_config = {}
        for provider, config in self._config.items():
            safe_config[provider] = {k: v for k, v in config.items() if k != 'api_key'}
        
        with open(output_file, 'w') as f:
            json.dump(safe_config, f, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def get_default_text_column(self) -> str:
        """Get the default text column name."""
        if 'general' in self._config:
            return self._config['general'].get('default_text_column', 'combined_text')
        return 'combined_text'
    
    def get_default_job_id_column(self) -> str:
        """Get the default job ID column name."""
        if 'general' in self._config:
            return self._config['general'].get('default_job_id_column', 'Job_Id')
        return 'Job_Id'
    
    def get_default_output_dir(self) -> str:
        """Get the default output directory."""
        if 'general' in self._config:
            return self._config['general'].get('default_output_dir', 'embeddings')
        return 'embeddings'
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        if 'general' in self._config:
            return self._config['general'].get('debug_mode', False)
        return False


# Global configuration instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> EmbeddingConfig:
    """
    Get global configuration instance.
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        EmbeddingConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = EmbeddingConfig(config_file)
    
    return _global_config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _global_config
    _global_config = None