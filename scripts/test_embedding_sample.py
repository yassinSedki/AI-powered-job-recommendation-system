"""
Test Embedding Sample

A simple script to test the embedding functionality with a small sample
from the 10K dataset before processing the entire dataset.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import embedding modules
from ai_model.embedding import (
    create_job_embedding,
    create_job_embeddings_batch,
    get_embedding_service,
    get_config
)

def test_single_embedding():
    """Test creating a single embedding."""
    print("🧪 Testing single embedding creation...")
    
    # Sample job description
    sample_text = """
    Software Engineer - Full Stack Developer
     React, Node.js, Python, and databases.
    """
    
    try:
        # Create single embedding
        embedding = create_job_embedding(sample_text)
        
        print(f"✅ Single embedding created successfully!")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding type: {type(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating single embedding: {e}")
        return False

def test_batch_embeddings():
    """Test creating batch embeddings."""
    print("\n🧪 Testing batch embedding creation...")
    
    # Sample job descriptions
    sample_texts = [
        "Data Scientist with Python and machine learning experience",
        "Marketing Manager for digital campaigns and social media",
        "Project Manager with Agile and Scrum methodology experience",
        "UX Designer with Figma and user research skills",
        "DevOps Engineer with AWS and Docker expertise"
    ]
    
    try:
        # Create batch embeddings
        embeddings = create_job_embeddings_batch(sample_texts, batch_size=2)
        
        print(f"✅ Batch embeddings created successfully!")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Number of texts: {len(sample_texts)}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating batch embeddings: {e}")
        return False

def test_dataset_sample():
    """Test with a small sample from the actual 10K dataset."""
    print("\n🧪 Testing with dataset sample...")
    
    try:
        # Load a small sample from the 10K dataset
        dataset_file = project_root / "dataset" / "complete_dataset10k.csv"
        
        if not dataset_file.exists():
            print(f"❌ Dataset file not found: {dataset_file}")
            return False
        
        # Load first 5 rows
        df_sample = pd.read_csv(dataset_file, nrows=5)
        
        print(f"📊 Loaded sample: {len(df_sample)} rows")
        print(f"   Columns: {list(df_sample.columns)}")
        
        # Check required columns
        if 'combined_text' not in df_sample.columns:
            print("❌ 'combined_text' column not found")
            return False
        
        if 'Job_Id' not in df_sample.columns:
            print("❌ 'Job_Id' column not found")
            return False
        
        # Test embeddings for sample
        sample_texts = df_sample['combined_text'].tolist()
        sample_ids = df_sample['Job_Id'].tolist()
        
        print(f"📝 Sample job IDs: {sample_ids}")
        print(f"📝 Sample text lengths: {[len(text) for text in sample_texts]}")
        
        # Create embeddings
        embeddings = create_job_embeddings_batch(sample_texts, batch_size=2)
        
        print(f"✅ Dataset sample embeddings created successfully!")
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Show some statistics
        print(f"   Embedding stats:")
        print(f"     Min: {embeddings.min():.4f}")
        print(f"     Max: {embeddings.max():.4f}")
        print(f"     Mean: {embeddings.mean():.4f}")
        print(f"     Std: {embeddings.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset sample: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def check_configuration():
    """Check if the embedding configuration is set up correctly."""
    print("🔧 Checking embedding configuration...")
    
    try:
        config = get_config()
        
        # Get default provider
        default_provider = config.get_default_provider()
        print(f"   Default provider: {default_provider}")
        
        if not default_provider:
            print("❌ No default provider configured")
            return False
        
        # Get provider config
        provider_config = config.get_provider_config(default_provider)
        
        if not provider_config:
            print(f"❌ No configuration for provider: {default_provider}")
            return False
        
        # Check API key (don't print it)
        has_api_key = bool(provider_config.get('api_key'))
        print(f"   API key configured: {has_api_key}")
        
        if not has_api_key:
            print("❌ No API key configured")
            return False
        
        # Check model
        model = provider_config.get('model', 'default')
        print(f"   Model: {model}")
        
        print("✅ Configuration looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("="*60)
    print("🚀 EMBEDDING FUNCTIONALITY TEST")
    print("="*60)
    
    # Check configuration first
    if not check_configuration():
        print("\n❌ Configuration check failed. Please check your .env file.")
        return False
    
    # Run tests
    tests = [
        ("Single Embedding", test_single_embedding),
        ("Batch Embeddings", test_batch_embeddings),
        ("Dataset Sample", test_dataset_sample)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to process the full 10K dataset.")
        print("💡 Run: python scripts/process_10k_embeddings.py")
    else:
        print("\n⚠️  Some tests failed. Please check your configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)