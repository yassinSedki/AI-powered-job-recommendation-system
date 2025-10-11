import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Ensure project root is on sys.path for ai_model imports (JobHunt root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ..model.schemas import (
    PredictSalaryRequest,
    PredictSalaryResponse,
    RecommendJobsRequest,
    RecommendJobsResponse,
    PredictAndRecommendRequest,
    PredictAndRecommendResponse,
    JobRecommendation,
)
from ..core.services.salary_service import SalaryPredictionService
from ..core.services.recommendation_service import RecommendationService
from ..core.path_utils import get_embeddings_dir, get_dataset_path, get_salary_artifacts_dir

from ai_model.embedding import create_embedding
from ai_model.feature_engineering.utils.text_processing import clean_and_combine_text

app = FastAPI(title="JobHunt Backend", version="1.0.0")

# Enable CORS for frontend (Vite/Next dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

salary_service = SalaryPredictionService()
recommend_service = RecommendationService()

@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        embeddings_dir = get_embeddings_dir()
        dataset_path = get_dataset_path()
        salary_dir = get_salary_artifacts_dir()
        return {
            "status": "ok",
            "embeddings_dir": embeddings_dir,
            "dataset_exists": os.path.exists(dataset_path),
            "salary_artifacts_exists": os.path.exists(salary_dir),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/jobs/{job_id}", response_model=JobRecommendation)
async def get_job(job_id: Any) -> JobRecommendation:
    try:
        job = recommend_service.get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobRecommendation(**job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict_salary", response_model=PredictSalaryResponse)
async def predict_salary(req: PredictSalaryRequest) -> PredictSalaryResponse:
    try:
        # If embedding provided, use it; else compute from role+skills
        if req.embedding_1024 is not None and len(req.embedding_1024) == 1024:
            embedding = np.array(req.embedding_1024, dtype=float)
        else:
            if not (req.role or req.skills):
                raise HTTPException(status_code=400, detail="Provide either embedding_1024 or role/skills to compute embedding")
            combined = clean_and_combine_text(req.role, req.role, req.skills)
            try:
                embedding = create_embedding(combined)
            except Exception:
                raise HTTPException(status_code=400, detail="Embedding provider is not configured. Provide embedding_1024 directly or configure the embedding provider (e.g., set API key)")
        pred = salary_service.predict(
            embedding_1024=embedding,
            work_type=req.work_type,
            qualification=req.qualification,
            preference=req.preference,
            experience_mid=req.experience_mid,
            latitude=req.latitude,
            longitude=req.longitude,
        )
        return PredictSalaryResponse(predicted_salary=float(pred))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations", response_model=RecommendJobsResponse)
async def recommendations(req: RecommendJobsRequest) -> RecommendJobsResponse:
    try:
        recs = recommend_service.recommend(
            role=req.role,
            skills=req.skills,
            education=req.education,
            work_type=req.work_type,
            experience=req.experience,
            latitude=req.latitude,
            longitude=req.longitude,
            gender=req.gender,
            max_recommendations=req.max_recommendations,
        )
        # Return enriched recommendations directly
        return RecommendJobsResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict_and_recommend", response_model=PredictAndRecommendResponse)
async def predict_and_recommend(req: PredictAndRecommendRequest) -> PredictAndRecommendResponse:
    try:
        combined = clean_and_combine_text(req.role, req.role, req.skills)
        try:
            embedding = create_embedding(combined)
        except Exception:
            raise HTTPException(status_code=400, detail="Embedding provider is not configured. Provide embedding_1024 to the salary endpoint or configure the embedding provider (e.g., set API key)")
        pred = salary_service.predict(
            embedding_1024=embedding,
            work_type=req.work_type,
            qualification=req.qualification,
            preference=req.preference,
            experience_mid=req.experience,
            latitude=req.latitude,
            longitude=req.longitude,
        )
        recs = recommend_service.recommend(
            role=req.role,
            skills=req.skills,
            education=req.qualification,
            work_type=req.work_type,
            experience=req.experience,
            latitude=req.latitude,
            longitude=req.longitude,
            gender=req.gender,
            max_recommendations=req.max_recommendations,
        )
        return PredictAndRecommendResponse(
            predicted_salary=float(pred),
            recommendations=recs,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))