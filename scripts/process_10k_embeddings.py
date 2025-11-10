"""
Process 10K Dataset with Embeddings

This script applies the existing embedding logic to the complete_dataset10k.csv file,
creating embeddings for all job descriptions using the configured embedding service.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import math

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import embedding modules
from ai_model.embedding import (
    process_job_dataset,
    get_embedding_service,
    EmbeddingConfig,
    get_config
)

def setup_logging() -> logging.Logger:
    """Setup logging for the embedding process."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"embedding_10k_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def calculate_rate_limiting(total_jobs: int, requests_per_minute: int = 100) -> Dict[str, Any]:
    """
    Calculate optimal batch size and timing for rate limiting.
    
    Args:
        total_jobs: Total number of jobs to process
        requests_per_minute: API rate limit (default: 100 for Cohere Trial)
        
    Returns:
        Dict with batch size, delay, and estimated time
    """
    # Conservative approach: use 90% of rate limit to avoid hitting limits
    safe_requests_per_minute = int(requests_per_minute * 0.9)
    
    # Calculate optimal batch size (each job = 1 request)
    # We want to process in chunks that fit within the rate limit
    optimal_batch_size = min(safe_requests_per_minute, total_jobs)
    
    # Calculate delay between batches (in seconds)
    if total_jobs > safe_requests_per_minute:
        delay_between_batches = 60.0  # Wait 1 minute between batches
    else:
        delay_between_batches = 0.0
    
    # Calculate total estimated time
    total_batches = math.ceil(total_jobs / optimal_batch_size)
    estimated_time_minutes = total_batches * (delay_between_batches / 60.0)
    
    return {
        'batch_size': optimal_batch_size,
        'delay_between_batches': delay_between_batches,
        'total_batches': total_batches,
        'estimated_time_minutes': estimated_time_minutes,
        'safe_requests_per_minute': safe_requests_per_minute
    }

def validate_dataset(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Validate the dataset structure and required columns."""
    required_columns = ['Job_Id', 'combined_text']
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for null values in critical columns
    null_job_ids = df['Job_Id'].isnull().sum()
    null_texts = df['combined_text'].isnull().sum()
    
    if null_job_ids > 0:
        logger.warning(f"Found {null_job_ids} null Job_Id values")
    
    if null_texts > 0:
        logger.warning(f"Found {null_texts} null combined_text values")
        # Remove rows with null text
        df_clean = df.dropna(subset=['combined_text'])
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    logger.info("Dataset validation passed")
    return df

def analyze_text_data(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze the text data to provide insights."""
    text_col = 'combined_text'
    
    # Text length statistics
    text_lengths = df[text_col].str.len()
    
    stats = {
        'total_jobs': len(df),
        'unique_job_ids': df['Job_Id'].nunique(),
        'text_length_stats': {
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'std': text_lengths.std()
        },
        'empty_texts': (df[text_col].str.strip() == '').sum(),
        'very_short_texts': (text_lengths < 50).sum(),
        'very_long_texts': (text_lengths > 2000).sum()
    }
    
    logger.info("Text Data Analysis:")
    logger.info(f"  Total jobs: {stats['total_jobs']:,}")
    logger.info(f"  Unique Job IDs: {stats['unique_job_ids']:,}")
    logger.info(f"  Text length - Min: {stats['text_length_stats']['min']}, Max: {stats['text_length_stats']['max']}")
    logger.info(f"  Text length - Mean: {stats['text_length_stats']['mean']:.1f}, Median: {stats['text_length_stats']['median']:.1f}")
    logger.info(f"  Empty texts: {stats['empty_texts']}")
    logger.info(f"  Very short texts (<50 chars): {stats['very_short_texts']}")
    logger.info(f"  Very long texts (>2000 chars): {stats['very_long_texts']}")
    
    return stats

def check_embedding_config(logger: logging.Logger) -> bool:
    """Check if embedding configuration is properly set up."""
    try:
        config = get_config()
        
        # Check if we have a valid provider
        default_provider = config.get_default_provider()
        if not default_provider:
            logger.error("No default embedding provider configured")
            return False
        
        logger.info(f"Default embedding provider: {default_provider}")
        
        # Check provider configuration
        provider_config = config.get_provider_config(default_provider)
        if not provider_config:
            logger.error(f"No configuration found for provider: {default_provider}")
            return False
        
        # Check API key
        if not provider_config.get('api_key'):
            logger.error(f"No API key configured for provider: {default_provider}")
            return False
        
        logger.info(f"Provider model: {provider_config.get('model', 'default')}")
        logger.info("Embedding configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"Error checking embedding configuration: {e}")
        return False

def process_embeddings_with_progress(df: pd.DataFrame, 
                                   output_dir: str,
                                   rate_limit_info: Dict[str, Any],
                                   logger: logging.Logger) -> Dict[str, str]:
    """Process embeddings with rate limiting and detailed progress tracking."""
    
    logger.info("="*60)
    logger.info("STARTING EMBEDDING PROCESSING WITH RATE LIMITING")
    logger.info("="*60)
    
    batch_size = rate_limit_info['batch_size']
    delay_between_batches = rate_limit_info['delay_between_batches']
    total_batches = rate_limit_info['total_batches']
    
    logger.info(f"Rate limiting configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total batches: {total_batches}")
    logger.info(f"  Delay between batches: {delay_between_batches:.1f} seconds")
    logger.info(f"  Estimated total time: {rate_limit_info['estimated_time_minutes']:.1f} minutes")
    
    start_time = time.time()
    
    try:
        # Process in batches with rate limiting
        all_embeddings = []
        all_job_ids = []
        
        for batch_num in range(total_batches):
            batch_start_idx = batch_num * batch_size
            batch_end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[batch_start_idx:batch_end_idx].copy()
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                       f"(jobs {batch_start_idx + 1}-{batch_end_idx})")
            
            batch_start_time = time.time()
            
            # Process this batch
            batch_result = process_job_dataset(
                jobs_data=batch_df,
                text_col='combined_text',
                job_id_col='Job_Id',
                batch_size=len(batch_df),  # Process entire batch at once
                output_dir=f"{output_dir}_temp_batch_{batch_num}"
            )
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            logger.info(f"Batch {batch_num + 1} completed in {batch_time:.2f} seconds")
            
            # Load and accumulate results
            batch_embeddings = np.load(batch_result['embeddings'])
            with open(batch_result['job_ids'], 'rb') as f:
                import pickle
                batch_job_ids = pickle.load(f)
            
            all_embeddings.append(batch_embeddings)
            all_job_ids.extend(batch_job_ids)
            
            # Clean up temporary files and directory
            temp_dir = f"{output_dir}_temp_batch_{batch_num}"
            for file_path in batch_result.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove the temporary directory if it exists and is empty
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                try:
                    os.rmdir(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except OSError:
                    logger.warning(f"Could not remove temporary directory: {temp_dir} (may not be empty)")
            
            # Rate limiting: wait between batches (except for the last batch)
            if batch_num < total_batches - 1 and delay_between_batches > 0:
                logger.info(f"Rate limiting: waiting {delay_between_batches:.1f} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        # Combine all embeddings
        logger.info("Combining all batch results...")
        final_embeddings = np.vstack(all_embeddings)
        
        # Save final results
        from ai_model.embedding.embedding_model.EmbeddingService import save_embeddings
        from ai_model.embedding import get_embedding_service
        
        service = get_embedding_service()
        provider_info = service.get_provider_info()
        
        result = save_embeddings(
            embeddings=final_embeddings,
            job_ids=all_job_ids,
            provider_info=provider_info,
            output_dir=output_dir
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("="*60)
        logger.info("EMBEDDING PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"Average time per job: {processing_time/len(df):.4f} seconds")
        logger.info(f"Final embeddings shape: {final_embeddings.shape}")
        
        # Log result files
        for file_type, file_path in result.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logger.info(f"{file_type}: {file_path} ({file_size:.2f} MB)")
            else:
                logger.warning(f"{file_type}: {file_path} (FILE NOT FOUND)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during embedding processing: {e}")
        raise

def validate_embeddings(result: Dict[str, str], 
                       original_count: int,
                       logger: logging.Logger) -> bool:
    """Validate the generated embeddings."""
    
    logger.info("="*60)
    logger.info("VALIDATING EMBEDDINGS")
    logger.info("="*60)
    
    try:
        # Check if embeddings file exists
        embeddings_file = result.get('embeddings_file')
        if not embeddings_file or not os.path.exists(embeddings_file):
            logger.error("Embeddings file not found")
            return False
        
        # Load and validate embeddings
        embeddings = np.load(embeddings_file)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Check dimensions
        if len(embeddings) != original_count:
            logger.error(f"Embedding count mismatch: expected {original_count}, got {len(embeddings)}")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embeddings).any():
            logger.error("Found NaN values in embeddings")
            return False
        
        if np.isinf(embeddings).any():
            logger.error("Found infinite values in embeddings")
            return False
        
        # Basic statistics
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        logger.info(f"Embedding mean: {embeddings.mean():.4f}")
        logger.info(f"Embedding std: {embeddings.std():.4f}")
        
        logger.info("✅ Embeddings validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating embeddings: {e}")
        return False

def main():
    """Main function to process the 10K dataset with embeddings."""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("STARTING 10K DATASET EMBEDDING PROCESSING")
    logger.info("="*80)
    
    try:
        # Configuration
        dataset_file = project_root / "dataset" / "dataset_with_features_10k.csv"
        output_dir = project_root / "embeddings" 
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset file: {dataset_file}")
        logger.info(f"Output directory: {output_dir}")
        
        # Check embedding configuration
        if not check_embedding_config(logger):
            logger.error("Embedding configuration check failed. Please check your .env file.")
            return False
        
        # Load dataset
        logger.info("Loading dataset...")
        if not dataset_file.exists():
            logger.error(f"Dataset file not found: {dataset_file}")
            return False
        
        df = pd.read_csv(dataset_file)
        
        # Validate dataset
        df_clean = validate_dataset(df, logger)
        if df_clean is False:
            logger.error("Dataset validation failed")
            return False
        
        if isinstance(df_clean, pd.DataFrame):
            df = df_clean
        
        # Analyze text data
        text_stats = analyze_text_data(df, logger)
        
        # Calculate rate limiting configuration
        logger.info(f"Processing {len(df)} jobs for embeddings...")
        
        rate_limit_info = calculate_rate_limiting(len(df))
        logger.info(f"Rate limiting calculated for {len(df)} jobs:")
        logger.info(f"  Batch size: {rate_limit_info['batch_size']}")
        logger.info(f"  Total batches: {rate_limit_info['total_batches']}")
        logger.info(f"  Estimated time: {rate_limit_info['estimated_time_minutes']:.1f} minutes")
        
        # Process embeddings
        result = process_embeddings_with_progress(
            df=df,
            output_dir=str(output_dir),
            rate_limit_info=rate_limit_info,
            logger=logger
        )
        
        # Validate embeddings
        if validate_embeddings(result, len(df), logger):
            logger.info("="*80)
            logger.info("🎉 EMBEDDING PROCESSING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info("Generated files:")
            for file_type, file_path in result.items():
                logger.info(f"  📁 {file_type}: {file_path}")
            
            return True
        else:
            logger.error("Embedding validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)