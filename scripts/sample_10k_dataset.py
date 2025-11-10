"""
Sample 10K rows from the complete dataset with features

This script creates a random sample of 10,000 rows from the complete_dataset_with_features.csv
while maintaining the same structure and all engineered features.
"""

import pandas as pd
import numpy as np

def create_10k_sample():
    """Create a 10K sample from the complete dataset."""
    
    print("Loading complete dataset with features...")
    
    # Load the complete dataset
    input_file = "dataset/dataset_with_features.csv"
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df):,} rows, {len(df.columns)} columns")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create random sample of 10,000 rows
    sample_size = min(10000, len(df))  # In case dataset has fewer than 10K rows
    sampled_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Sample created: {len(sampled_df):,} rows")
    
    # Save the sampled dataset
    output_file = "dataset/dataset_with_features_10k.csv"
    sampled_df.to_csv(output_file, index=False)
    
    print(f"Sample saved to: {output_file}")
    
    # Display summary statistics
    print("\n" + "="*50)
    print("DATASET SAMPLING SUMMARY")
    print("="*50)
    print(f"Original rows: {len(df):,}")
    print(f"Sampled rows: {len(sampled_df):,}")
    print(f"Sampling ratio: {(len(sampled_df) / len(df)) * 100:.1f}%")
    print(f"Columns preserved: {len(sampled_df.columns)}")
    
    # Show diversity metrics
    if 'Job Title' in sampled_df.columns:
        print(f"Unique job titles in sample: {sampled_df['Job Title'].nunique():,}")
    
    if 'Company' in sampled_df.columns:
        print(f"Unique companies in sample: {sampled_df['Company'].nunique():,}")
    
    if 'Country' in sampled_df.columns:
        print(f"Unique countries in sample: {sampled_df['Country'].nunique():,}")
    
    wt_col = 'Work_Type' if 'Work_Type' in sampled_df.columns else ('Work Type' if 'Work Type' in sampled_df.columns else None)
    if wt_col:
        print(f"Work types in sample: {sampled_df[wt_col].nunique()}")
        print("Work type distribution:")
        work_type_counts = sampled_df[wt_col].value_counts()
        for work_type, count in work_type_counts.items():
            print(f"  {work_type}: {count:,} ({count/len(sampled_df)*100:.1f}%)")
    
    # Show feature columns preserved
    feature_columns = [col for col in sampled_df.columns if any(prefix in col for prefix in 
                      ['is_', 'has_', 'experience_', 'salary_', 'qualification_'])]
    
    print(f"\nEngineered features preserved: {len(feature_columns)}")
    print("Feature categories:")
    categories = {}
    for col in feature_columns:
        if col.startswith('is_'):
            categories.setdefault('Boolean flags (is_)', []).append(col)
        elif col.startswith('has_'):
            categories.setdefault('Boolean flags (has_)', []).append(col)
        elif col.startswith('experience_'):
            categories.setdefault('Experience features', []).append(col)
        elif col.startswith('salary_'):
            categories.setdefault('Salary features', []).append(col)
        elif col.startswith('qualification_'):
            categories.setdefault('Qualification features', []).append(col)
    
    for category, cols in categories.items():
        print(f"  {category}: {len(cols)} features")
    
    print("\n✅ Dataset sampling completed successfully!")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    create_10k_sample()