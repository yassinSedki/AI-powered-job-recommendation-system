import os
import sys
import numpy as np
import pandas as pd

# Ensure ai_model is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.append(os.path.join(PROJECT_ROOT))

# Imports from project modules
from ai_model.feature_engineering.utils.text_processing import clean_and_combine_text
from ai_model.embedding import create_embedding
from ai_model.recommendation.core_recommendation import recommend_jobs


def build_user_profile(role: str, skills: str) -> dict:
    """Build a simple user profile dict for recommendation inputs."""
    return {
        "role": role,
        "skills": skills,
    }


def prepare_query_text(user_profile: dict) -> str:
    """Produce unified text from user role and skills using the same cleaner as jobs."""
    role = user_profile.get("role", "")
    skills = user_profile.get("skills", "")
    # Duplicate role into job_title to enrich the combined string (optional)
    query_text = clean_and_combine_text(role, role, skills)
    return query_text


def load_jobs_df(dataset_path: str) -> pd.DataFrame:
    """Load jobs DataFrame ensuring required columns exist and map experience fields."""
    df = pd.read_csv(dataset_path)

    # Verify basic required columns
    base_required_cols = [
        "Job_Id",
        "latitude",
        "longitude",
        "combined_text",
    ]
    missing_base = [c for c in base_required_cols if c not in df.columns]
    if missing_base:
        raise ValueError(f"Dataset at {dataset_path} is missing required columns: {missing_base}")

    # ExperienceScorer expects 'min_experience' and 'max_experience' but dataset uses 'experience_min'/'experience_max'
    # Map if necessary
    if "min_experience" not in df.columns and "experience_min" in df.columns:
        df["min_experience"] = df["experience_min"]
    if "max_experience" not in df.columns and "experience_max" in df.columns:
        df["max_experience"] = df["experience_max"]

    return df


def main():
    # Example user input
    example_role = "Data Scientist"
    example_skills = "Python, Machine Learning, SQL, Deep Learning, Statistics"

    # Build user profile and query text
    user_profile = build_user_profile(example_role, example_skills)
    query_text = prepare_query_text(user_profile)
    print("Prepared query text:\n", query_text)

    # Create embedding for the query text
    try:
        query_embedding = create_embedding(query_text)
        if isinstance(query_embedding, np.ndarray):
            print(f"Created query embedding with shape: {query_embedding.shape}")
        else:
            print("Warning: query_embedding is not a numpy array; type:", type(query_embedding))
    except Exception as e:
        print(f"Error creating embedding: {e}. Proceeding without embedding scorer.")
        query_embedding = None

    # Load jobs dataset
    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "complete_dataset10k.csv")
    jobs_df = load_jobs_df(dataset_path)
    print(f"Loaded jobs dataset with shape: {jobs_df.shape}")

    # Embeddings directory for precomputed job embeddings
    embeddings_dir = os.path.join(PROJECT_ROOT, "embeddings")

    # Scoring weights: adjust depending on whether we have a query embedding
    if query_embedding is None:
        scoring_weights = {
            "experience": 0.6,
            "location": 0.4,
        }
    else:
        scoring_weights = {
            "experience": 0.4,
            "location": 0.2,
            "embedding": 0.4,
        }

    # Example user criteria for scoring
    user_experience_years = 3  # adjust to test experience matching
    user_latitude = 30.0444    # Cairo example
    user_longitude = 31.2357

    # Run full recommendation
    recommendations = recommend_jobs(
        jobs_df=jobs_df,
        gender="male",               # or 'male'/'female' if you want to filter
        work_type="Internship",            # e.g., 'Full-time', 'Part-time', etc.
        education="Master",            # e.g., 'Bachelor', 'Master', 'PhD', etc.
        user_experience=user_experience_years,
        user_latitude=user_latitude,
        user_longitude=user_longitude,
        query_embedding=query_embedding,
        scoring_weights=scoring_weights,
        embeddings_dir=embeddings_dir,
        max_recommendations=20,
        filter_logic="AND",       # 'AND' or 'OR' for combining filters
    )

    # Display top recommendations
    print("\nTop recommendations (showing up to 10):")
    for i, rec in enumerate(recommendations[:10], start=1):
        print(f"{i}. Job ID: {rec.get('job_id')} | Title: {rec.get('title')} | Company: {rec.get('company')} | Work Type: {rec.get('work_type')} | Score: {rec.get('total_score'):.4f}")


if __name__ == "__main__":
    main()