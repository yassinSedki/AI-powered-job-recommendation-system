# JobHunt

An end-to-end, AI-powered job recommendation and salary prediction system. It blends traditional feature engineering with modern text embeddings and explainable multi-scorer logic, exposed via a FastAPI backend and a Vite React frontend.

## Why this project matters
- Personalized, explainable recommendations: experience, location, and embedding-based signals are combined with clear, tunable weights.
- Practical salary prediction: uses engineered features and/or 1024-d embeddings to estimate compensation.
- Production-ready surfaces: REST API + modern frontend for quick demos and integration.
- Realistic scale: built and validated on a 10k job postings dataset for robust testing and iteration.

## Dataset (10k jobs)
- Location: `dataset/`
- Key files:
  - `complete_dataset10k.csv` — the core 10k job postings dataset
  - `complete_dataset_with_features.csv` — enriched dataset with engineered features
  - `engineered_features_only.csv` — features-only view
  - `feature_metadata.csv`, `transformation_statistics.txt` — feature and pipeline info
- Related scripts: `scripts/sample_10k_dataset.py`, `scripts/reduce_dataset.py`, `scripts/process_10k_embeddings.py` (for generating/processing embeddings)

## How it works (pipeline)
1) Ingest the 10k dataset from `dataset/`
2) Feature engineering via `ai_model/feature_engineering/pipeline.py`
3) Text embeddings from job descriptions via `ai_model/embedding/job_embeddings.py`
4) Recommendation scoring with Experience, Location, and Embedding scorers (`ai_model/recommendation/core_recommendation.py`)
5) Salary prediction model under `ai_model/salary_predection/`
6) Serve through FastAPI (`backend/app/api/main.py`) and visualize with React (`frontend/`)

## Project Structure
- `backend/app` — FastAPI app (API, core services, models, tests)
- `frontend` — Vite React UI (Tailwind CSS)
- `ai_model` — embeddings, recommendation scoring, feature engineering, notebooks, salary prediction
- `dataset` — input CSVs (10k jobs + engineered variants)
- `embeddings` — generated embeddings & metadata (`job_embeddings.npy`, `job_ids.pkl`, etc.)
- `scripts` — utilities: sampling/reduction, embeddings processing, tests

## Prerequisites
- Python 3.12 + Poetry
- Node.js (LTS) + npm

## Setup
1) Copy `.env.example` to `.env` and fill required values (e.g., `COHERE_API_KEY`).
2) Install Python deps: `poetry install`.
3) Frontend deps (optional for API-only): `cd frontend && npm install`.

## Environment
- Embedding (Cohere): `COHERE_API_KEY`, `COHERE_MODEL`, `COHERE_BATCH_SIZE`.
- Optional embedding vars: `DEFAULT_EMBEDDING_PROVIDER`, `DEFAULT_EMBEDDINGS_OUTPUT_DIR`, `DEFAULT_TEXT_COLUMN`, `DEFAULT_JOB_ID_COLUMN`, `LOG_LEVEL`, `EMBEDDING_DEBUG`.
- Optional app vars: `VITE_API_BASE_URL`, `BACKEND_HOST`, `BACKEND_PORT`.

## Run
- Backend: `poetry run uvicorn backend.app.api.main:app --reload` → http://127.0.0.1:8000/
- Frontend: `cd frontend && npm run dev` → http://127.0.0.1:5173/

## API Overview
- GET `/health` — status; checks for dataset/embeddings/salary artifacts.
- GET `/api/jobs/{job_id}` — job by ID.
- POST `/api/predict_salary` — predict salary using features and/or embeddings.
- POST `/api/recommendations` — recommendations for given criteria.
- POST `/api/predict_and_recommend` — compute embedding, predict salary, and return recommendations.

## Recommendation Logic (explainable)
- Scorers: Experience, Location, Embedding.
- Defaults: Experience=0.9, Location=0.9 (embedding complements these).
- Combine: per-scorer outputs min-max normalized to [0,1], then weighted average forms final score.
- Customize: pass `scoring_weights` to tune relative importance per signal.

## Notebooks & Scripts
- Notebooks: EDA (`ai_model/notebooks/eda_analysis.ipynb`), feature engineering (`feature_engineering_comprehensive.ipynb`), salary training.
- Scripts: sampling/reduction (`sample_10k_dataset.py`, `reduce_dataset.py`), embeddings (`process_10k_embeddings.py`), tests (`test_full_recommendation.py`, `test_embedding_sample.py`).

## Testing
- Run: `poetry run pytest -q`.

## Troubleshooting
- Embeddings: ensure `COHERE_API_KEY` is set and provider reachable.
- Data missing: confirm files exist under `dataset/`.
- CORS/ports: adjust `VITE_API_BASE_URL` and/or `BACKEND_PORT`, then restart dev servers.