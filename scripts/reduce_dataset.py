import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def stratified_sample_dataset(df, target_size=100000, random_state=42):
    """
    Reduce dataset using stratified random sampling to maintain representativeness.
    
    Args:
        df: Original dataframe
        target_size: Desired number of rows (default: 100,000)
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled dataframe
    """
    print(f"Original dataset size: {len(df):,} rows")
    
    # Define stratification variables (most important categorical variables)
    wt_col = 'Work_Type' if 'Work_Type' in df.columns else ('Work Type' if 'Work Type' in df.columns else None)
    strata_columns = [wt_col or 'Work Type', 'Job Portal', 'Preference']
    
    # Create a combined stratification key
    df['strata_key'] = df[strata_columns].astype(str).agg('_'.join, axis=1)
    
    # Calculate sampling ratio
    sampling_ratio = target_size / len(df)
    print(f"Sampling ratio: {sampling_ratio:.1%}")
    
    # Get strata distribution
    strata_counts = df['strata_key'].value_counts()
    print(f"\nNumber of unique strata: {len(strata_counts)}")
    
    sampled_dfs = []
    
    for stratum, count in strata_counts.items():
        # Calculate target sample size for this stratum
        stratum_target = max(1, int(count * sampling_ratio))
        
        # Get data for this stratum
        stratum_data = df[df['strata_key'] == stratum]
        
        # Sample from this stratum
        if len(stratum_data) <= stratum_target:
            # If stratum is smaller than target, take all
            sampled_stratum = stratum_data
        else:
            # Random sample from stratum
            sampled_stratum = stratum_data.sample(
                n=stratum_target, 
                random_state=random_state
            )
        
        sampled_dfs.append(sampled_stratum)
    
    # Combine all sampled strata
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we're over target, randomly reduce
    if len(result_df) > target_size:
        result_df = result_df.sample(n=target_size, random_state=random_state)
    
    # If we're under target, add more samples
    elif len(result_df) < target_size:
        remaining_needed = target_size - len(result_df)
        # Get remaining data not in sample
        remaining_data = df[~df.index.isin(result_df.index)]
        additional_sample = remaining_data.sample(
            n=min(remaining_needed, len(remaining_data)), 
            random_state=random_state
        )
        result_df = pd.concat([result_df, additional_sample], ignore_index=True)
    
    # Remove the temporary strata_key column
    result_df = result_df.drop('strata_key', axis=1)
    
    print(f"Final sample size: {len(result_df):,} rows")
    return result_df

def simple_random_sample(df, target_size=100000, random_state=42):
    """
    Simple random sampling - fastest method.
    """
    print(f"Using simple random sampling...")
    print(f"Original size: {len(df):,} rows")
    
    sampled_df = df.sample(n=target_size, random_state=random_state)
    print(f"Sample size: {len(sampled_df):,} rows")
    
    return sampled_df

def systematic_sample(df, target_size=100000):
    """
    Systematic sampling - select every k-th row.
    """
    print(f"Using systematic sampling...")
    print(f"Original size: {len(df):,} rows")
    
    # Calculate step size
    step = len(df) // target_size
    
    # Select every k-th row
    indices = range(0, len(df), step)[:target_size]
    sampled_df = df.iloc[indices].copy()
    
    print(f"Sample size: {len(sampled_df):,} rows")
    print(f"Step size: {step}")
    
    return sampled_df

def compare_distributions(original_df, sampled_df, columns_to_compare):
    """
    Compare distributions between original and sampled data.
    """
    print("\n=== DISTRIBUTION COMPARISON ===")
    
    for col in columns_to_compare:
        if col in original_df.columns and col in sampled_df.columns:
            print(f"\n{col}:")
            
            # Original distribution
            orig_dist = original_df[col].value_counts(normalize=True).sort_index()
            
            # Sample distribution
            sample_dist = sampled_df[col].value_counts(normalize=True).sort_index()
            
            # Compare top categories
            comparison_df = pd.DataFrame({
                'Original_%': (orig_dist * 100).round(2),
                'Sample_%': (sample_dist * 100).round(2)
            }).fillna(0)
            
            comparison_df['Difference'] = (comparison_df['Sample_%'] - comparison_df['Original_%']).round(2)
            
            print(comparison_df.head(10))

def main():
    print("=== DATASET REDUCTION TOOL ===")
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('dataset/job_descriptions.csv')
    
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Choose sampling method
    print("\n=== SAMPLING METHODS AVAILABLE ===")
    print("1. Stratified Random Sampling (RECOMMENDED - maintains representativeness)")
    print("2. Simple Random Sampling (fastest)")
    print("3. Systematic Sampling (evenly distributed)")
    
    # For this implementation, we'll use stratified sampling (best approach)
    print("\nUsing STRATIFIED RANDOM SAMPLING (Best Approach)...")
    
    # Apply stratified sampling
    sampled_df = stratified_sample_dataset(df, target_size=100000)
    
    # Save the reduced dataset
    output_file = 'dataset/job_descriptions_100k_sample.csv'
    sampled_df.to_csv(output_file, index=False)
    print(f"\nReduced dataset saved to: {output_file}")
    
    # Compare distributions
    comparison_columns = [wt_col or 'Work Type', 'Job Portal', 'Preference', 'Country']
    compare_distributions(df, sampled_df, comparison_columns)
    
    # Memory comparison
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    sample_memory = sampled_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\n=== MEMORY REDUCTION ===")
    print(f"Original: {original_memory:.1f} MB")
    print(f"Sample: {sample_memory:.1f} MB")
    print(f"Reduction: {((original_memory - sample_memory) / original_memory * 100):.1f}%")
    
    print("\n=== PROCESS COMPLETE ===")
    print(f"✅ Successfully reduced dataset from {len(df):,} to {len(sampled_df):,} rows")
    print(f"✅ Maintained representativeness across key variables")
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    main()

# Alternative: Quick sampling functions for different approaches
def quick_sample_100k(method='stratified'):
    """
    Quick function to sample 100K rows using different methods.
    
    Args:
        method: 'stratified', 'random', or 'systematic'
    """
    df = pd.read_csv('dataset/job_descriptions.csv')
    
    if method == 'stratified':
        return stratified_sample_dataset(df, 100000)
    elif method == 'random':
        return simple_random_sample(df, 100000)
    elif method == 'systematic':
        return systematic_sample(df, 100000)
    else:
        raise ValueError("Method must be 'stratified', 'random', or 'systematic'")

