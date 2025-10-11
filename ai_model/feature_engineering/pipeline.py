"""Feature Engineering Pipeline

This module provides the main orchestrator for all feature transformations.
The FeatureEngineeringPipeline class combines all modular transformations
into a single, reusable pipeline for both notebooks and production code.

Classes:
- FeatureEngineeringPipeline: Main pipeline orchestrator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings

from .utils.qualifications import process_qualifications, get_qualification_stats
from .utils.work_type import process_work_types, get_work_type_stats
from .utils.experience import process_experience, get_experience_stats
from .utils.salary import process_salary, get_salary_stats
from .utils.text_processing import process_text_features, get_text_stats


class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline for JobHunt dataset.
    
    This class orchestrates all feature transformations and provides
    a clean interface for both notebook exploration and production use.
    
    Features:
    - Qualification classification (Bachelor/Master/PhD)
    - Work type standardization (Contract/Part-time/Internship/Full-time)
    - Experience range parsing (min/max/mid values)
    - Salary range parsing (min/max/mid values)
    - Text processing (role + job title + skills)
    
    Example:
        >>> pipeline = FeatureEngineeringPipeline()
        >>> features = pipeline.transform(df)
        >>> stats = pipeline.get_transformation_stats()
    """
    
    def __init__(self, 
                 qualification_col: str = 'Qualifications',
                 work_type_col: str = 'Work_Type',
                 experience_col: str = 'Experience',
                 salary_col: str = 'Salary',
                 role_col: str = 'Role',
                 job_title_col: str = 'Job Title',
                 skills_col: str = 'Skills'):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            qualification_col (str): Name of qualifications column
            work_type_col (str): Name of work type column
            experience_col (str): Name of experience column
            salary_col (str): Name of salary column
            role_col (str): Name of role column
            job_title_col (str): Name of job title column
            skills_col (str): Name of skills column
        """
        self.qualification_col = qualification_col
        self.work_type_col = work_type_col
        self.experience_col = experience_col
        self.salary_col = salary_col
        self.role_col = role_col
        self.job_title_col = job_title_col
        self.skills_col = skills_col
        
        # Store transformation statistics
        self.transformation_stats = {}
        self.feature_columns = []
        
    def transform(self, df: pd.DataFrame, 
                 include_stats: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
        """
        Apply all feature transformations to the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw job data
            include_stats (bool): Whether to compute transformation statistics
            verbose (bool): Whether to print progress information
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        if verbose:
            print("Starting feature engineering pipeline...")
            print(f"Input shape: {df.shape}")
        
        # Initialize result with original DataFrame
        result = df.copy()
        
        # Track feature columns added
        original_columns = set(df.columns)
        
        # 1. Process Qualifications
        if self.qualification_col in df.columns:
            if verbose:
                print(f"Processing qualifications from '{self.qualification_col}'...")
            
            qual_features = process_qualifications(df[self.qualification_col])
            result = pd.concat([result, qual_features], axis=1)
            
            if include_stats:
                self.transformation_stats['qualifications'] = get_qualification_stats(df[self.qualification_col])
        else:
            if verbose:
                warnings.warn(f"Column '{self.qualification_col}' not found. Skipping qualification processing.")
        
        # 2. Process Work Types
        work_type_col = self.work_type_col if self.work_type_col in df.columns else ('Work Type' if 'Work Type' in df.columns else None)
        if work_type_col:
            if verbose:
                print(f"Processing work types from '{work_type_col}'...")
            
            work_features = process_work_types(df[work_type_col])
            result = pd.concat([result, work_features], axis=1)
            
            if include_stats:
                self.transformation_stats['work_types'] = get_work_type_stats(df[work_type_col])
        else:
            if verbose:
                warnings.warn(f"Column '{self.work_type_col}' not found. Skipping work type processing.")
        
        # 3. Process Experience
        if self.experience_col in df.columns:
            if verbose:
                print(f"Processing experience from '{self.experience_col}'...")
            
            exp_features = process_experience(df[self.experience_col])
            result = pd.concat([result, exp_features], axis=1)
            
            if include_stats:
                self.transformation_stats['experience'] = get_experience_stats(df[self.experience_col])
        else:
            if verbose:
                warnings.warn(f"Column '{self.experience_col}' not found. Skipping experience processing.")
        
        # 4. Process Salary
        if self.salary_col in df.columns:
            if verbose:
                print(f"Processing salary from '{self.salary_col}'...")
            
            salary_features = process_salary(df[self.salary_col])
            result = pd.concat([result, salary_features], axis=1)
            
            if include_stats:
                self.transformation_stats['salary'] = get_salary_stats(df[self.salary_col])
        else:
            if verbose:
                warnings.warn(f"Column '{self.salary_col}' not found. Skipping salary processing.")
        
        # 5. Process Text Features
        if verbose:
            print(f"Processing text features from '{self.role_col}', '{self.job_title_col}', '{self.skills_col}'...")
        
        text_features = process_text_features(df, self.role_col, self.job_title_col, self.skills_col)
        result = pd.concat([result, text_features], axis=1)
        
        if include_stats:
            self.transformation_stats['text'] = get_text_stats(text_features['combined_text'])
        
        # Store feature columns for reference
        self.feature_columns = [col for col in result.columns if col not in original_columns]
        
        if verbose:
            print(f"Feature engineering complete!")
            print(f"Output shape: {result.shape}")
            print(f"Added {len(self.feature_columns)} new features")
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of all engineered feature columns.
        
        Returns:
            List[str]: List of feature column names
        """
        return self.feature_columns.copy()
    
    def get_transformation_stats(self) -> Dict:
        """
        Get statistics from all transformations.
        
        Returns:
            Dict: Dictionary with transformation statistics
        """
        return self.transformation_stats.copy()
    
    def print_transformation_summary(self):
        """
        Print a summary of all transformations and their statistics.
        """
        print("Feature Engineering Pipeline Summary")
        print("=" * 50)
        
        if not self.transformation_stats:
            print("No transformations have been applied yet.")
            return
        
        for transform_name, stats in self.transformation_stats.items():
            print(f"\n{transform_name.upper()} TRANSFORMATION:")
            print("-" * 30)
            
            if transform_name == 'qualifications':
                print(f"Total records: {stats['total_records']:,}")
                print(f"Has qualification: {stats['has_qualification']:,} ({stats['has_qualification_pct']:.1f}%)")
                print(f"Bachelor: {stats['bachelor_count']:,} ({stats['bachelor_count_pct']:.1f}%)")
                print(f"Master: {stats['master_count']:,} ({stats['master_count_pct']:.1f}%)")
                print(f"PhD: {stats['phd_count']:,} ({stats['phd_count_pct']:.1f}%)")
            
            elif transform_name == 'work_types':
                print(f"Total records: {stats['total_records']:,}")
                print(f"Has work type: {stats['has_work_type']:,} ({stats['has_work_type_pct']:.1f}%)")
                print(f"Full-time: {stats['full_time_count']:,} ({stats['full_time_count_pct']:.1f}%)")
                print(f"Contract: {stats['contract_count']:,} ({stats['contract_count_pct']:.1f}%)")
                print(f"Part-time: {stats['part_time_count']:,} ({stats['part_time_count_pct']:.1f}%)")
                print(f"Internship: {stats['internship_count']:,} ({stats['internship_count_pct']:.1f}%)")
            
            elif transform_name == 'experience':
                print(f"Total records: {stats['total_records']:,}")
                print(f"Has experience: {stats['has_experience']:,} ({stats['has_experience_pct']:.1f}%)")
                print(f"Entry level: {stats['entry_level_count']:,} ({stats['entry_level_count_pct']:.1f}%)")
                print(f"Mid level: {stats['mid_level_count']:,} ({stats['mid_level_count_pct']:.1f}%)")
                print(f"Senior level: {stats['senior_level_count']:,} ({stats['senior_level_count_pct']:.1f}%)")
                if stats['mid_experience_avg']:
                    print(f"Avg experience: {stats['mid_experience_avg']:.1f} years")
            
            elif transform_name == 'salary':
                print(f"Total records: {stats['total_records']:,}")
                print(f"Has salary: {stats['has_salary']:,} ({stats['has_salary_pct']:.1f}%)")
                if stats['mid_salary_avg']:
                    print(f"Avg salary: ${stats['mid_salary_avg']:,.0f}")
                    print(f"Median salary: ${stats['mid_salary_median']:,.0f}")
                if 'salary_p25' in stats:
                    print(f"Salary range (25th-75th percentile): ${stats['salary_p25']:,.0f} - ${stats['salary_p75']:,.0f}")
            
            elif transform_name == 'text':
                print(f"Total records: {stats['total_records']:,}")
                print(f"Non-empty text: {stats['non_empty_text']:,} ({stats['non_empty_text_pct']:.1f}%)")
        
        print(f"\nTOTAL FEATURES CREATED: {len(self.feature_columns)}")
        print("Feature categories:")
        for category in ['qualification', 'work_type', 'experience', 'salary', 'text']:
            category_features = [col for col in self.feature_columns if col.startswith(category)]
            if category_features:
                print(f"  {category}: {len(category_features)} features")
    
    def save_features(self, df: pd.DataFrame, filepath: str, 
                     features_only: bool = False):
        """
        Save the transformed DataFrame to a file.
        
        Args:
            df (pd.DataFrame): Transformed DataFrame
            filepath (str): Output file path
            features_only (bool): If True, save only engineered features
        """
        if features_only:
            feature_df = df[self.feature_columns]
            feature_df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        print(f"Features saved to: {filepath}")

