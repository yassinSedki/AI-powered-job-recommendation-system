"""Text Processing Module

This module provides functions to combine and clean text features from job data:
- Combine role, job title, and skills into a single text string
- Clean and normalize text (lowercase, remove punctuation, normalize spaces)
- Extract text statistics and features

Functions:
- clean_and_combine_text: Combine and clean multiple text columns
- process_text_features: Process text data and extract features
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Union, Optional, List


def remove_redundant_words(text: str, preserve_order: bool = True) -> str:
    """
    Remove redundant/duplicate words from text with advanced filtering.
    
    Args:
        text (str): Input text string
        preserve_order (bool): Whether to preserve the original word order
        
    Returns:
        str: Text with redundant words removed
        
    Examples:
        >>> remove_redundant_words("python java python sql java")
        'python java sql'
        >>> remove_redundant_words("Software Engineer software developer")
        'software engineer developer'
    """
    if not text or not isinstance(text, str):
        return ""
    
    words = text.split()
    if not words:
        return ""
    
    if preserve_order:
        # Preserve order while removing duplicates (case-insensitive)
        unique_words = []
        seen = set()
        
        for word in words:
            word_lower = word.lower()
            # Skip very short words and duplicates
            if len(word) > 1 and word_lower not in seen:
                unique_words.append(word)
                seen.add(word_lower)
        
        return ' '.join(unique_words)
    else:
        # Remove duplicates without preserving order (faster)
        unique_words = list(dict.fromkeys([w for w in words if len(w) > 1]))
        return ' '.join(unique_words)


def clean_text(text: Union[str, None]) -> str:
    """
    Clean and normalize a single text string.
    
    Operations:
    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation (except hyphens and underscores)
    - Normalize spaces
    - Remove special characters
    
    Args:
        text (str or None): Text to clean
        
    Returns:
        str: Cleaned text
        
    Examples:
        >>> clean_text("Software Engineer - Python/Java")
        'software engineer python java'
        >>> clean_text("Data Scientist (ML/AI)")
        'data scientist ml ai'
    """
    if pd.isna(text) or text is None:
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Replace common separators with spaces
    separators = [',', ';', '|', '/', '\\', '(', ')', '[', ']', '{', '}', ':', '"', "'"]
    for sep in separators:
        text = text.replace(sep, ' ')
    
    # Remove punctuation except hyphens and underscores
    punctuation_to_remove = string.punctuation.replace('-', '').replace('_', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    
    # Replace multiple hyphens/underscores with single space
    text = re.sub(r'[-_]+', ' ', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_and_combine_text(role: Union[str, None] = None, 
                          job_title: Union[str, None] = None, 
                          skills: Union[str, None] = None,
                          separator: str = ' ') -> str:
    """
    Combine and clean multiple text fields into a single normalized string with duplicate removal.
    
    This function:
    - Cleans each text field (lowercase, remove punctuation, normalize spaces)
    - Combines all fields with the specified separator
    - Removes redundant/duplicate words (case-insensitive)
    - Preserves word order while eliminating duplicates
    
    Args:
        role (str or None): Job role text
        job_title (str or None): Job title text
        skills (str or None): Skills text
        separator (str): Separator to use between fields
        
    Returns:
        str: Combined and cleaned text with no duplicate words
        
    Examples:
        >>> clean_and_combine_text("Software Engineer", "Senior Software Developer", "Python, Java, Python")
        'software engineer senior developer python java'
        >>> clean_and_combine_text(role="Data Scientist", skills="ML, AI, Machine Learning")
        'data scientist ml ai machine learning'
    """
    # Clean each text field
    cleaned_texts = []
    
    for text in [role, job_title, skills]:
        cleaned = clean_text(text)
        if cleaned:  # Only add non-empty strings
            cleaned_texts.append(cleaned)
    
    # Combine with separator
    combined = separator.join(cleaned_texts)
    
    # Remove redundant words using the advanced function
    return remove_redundant_words(combined, preserve_order=True)


# Removed extract_text_features function as only combined_text is needed


def process_text_features(df: pd.DataFrame, 
                         role_col: str = 'Role',
                         job_title_col: str = 'Job Title', 
                         skills_col: str = 'skills') -> pd.DataFrame:
    """
    Process text features from a DataFrame with role, job title, and skills columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing text columns
        role_col (str): Name of the role column
        job_title_col (str): Name of the job title column
        skills_col (str): Name of the skills column
        
    Returns:
        pd.DataFrame: DataFrame with processed text features:
            - combined_text: Cleaned and combined text
            
    Examples:
        >>> df = pd.DataFrame({
        ...     'Role': ['Software Engineer', 'Data Scientist'],
        ...     'Job Title': ['Senior Developer', 'ML Engineer'],
        ...     'Skills': ['Python, Java', 'Python, ML, AI']
        ... })
        >>> result = process_text_features(df)
        >>> print(result.columns.tolist())
        ['combined_text']
    """
    # Get columns with fallback to None if column doesn't exist
    role_series = df[role_col] if role_col in df.columns else pd.Series([None] * len(df))
    job_title_series = df[job_title_col] if job_title_col in df.columns else pd.Series([None] * len(df))
    skills_series = df[skills_col] if skills_col in df.columns else pd.Series([None] * len(df))
    
    # Combine and clean text
    combined_text = []
    for i in range(len(df)):
        role = role_series.iloc[i] if i < len(role_series) else None
        job_title = job_title_series.iloc[i] if i < len(job_title_series) else None
        skills = skills_series.iloc[i] if i < len(skills_series) else None
        
        combined = clean_and_combine_text(role, job_title, skills)
        combined_text.append(combined)
    
    # Create result DataFrame with only combined_text
    result = pd.DataFrame({
        'combined_text': combined_text
    })
    
    return result


def get_text_stats(text_series: pd.Series) -> dict:
    """
    Get basic statistics about text data.
    
    Args:
        text_series (pd.Series): Series containing text data
        
    Returns:
        dict: Dictionary with basic text statistics
    """
    # Basic text statistics
    non_empty = text_series.dropna().astype(str)
    non_empty = non_empty[non_empty.str.len() > 0]
    
    stats = {
        'total_records': len(text_series),
        'non_empty_text': len(non_empty),
        'avg_text_length': non_empty.str.len().mean() if len(non_empty) > 0 else 0,
        'max_text_length': non_empty.str.len().max() if len(non_empty) > 0 else 0,
        'min_text_length': non_empty.str.len().min() if len(non_empty) > 0 else 0,
        'median_text_length': non_empty.str.len().median() if len(non_empty) > 0 else 0
    }
    
    # Add percentages
    total = stats['total_records']
    stats['non_empty_text_pct'] = (stats['non_empty_text'] / total * 100) if total > 0 else 0
    
    return stats
