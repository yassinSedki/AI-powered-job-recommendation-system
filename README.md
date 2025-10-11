# JobHunt

AI-powered job recommendation and salary prediction system. It combines feature engineering, text embeddings, multi-scorer recommendation logic, and a FastAPI backend with a Vite React frontend.

## Highlights
- End-to-end pipeline: datasets → embeddings → feature engineering → recommendation & salary prediction → REST API + frontend.
- Smart weighted scoring: experience and location scorers default to 0.9 each; embedding scorer complements them. Scores are min-max normalized and combined via weighted average for stable rankings.
- Provider-agnostic embeddings: Cohere supported out of the box via environment variables; designed to extend to other providers.
- Clear APIs: health check, salary prediction, recommendations, and a combo endpoint.
- Reproducible dev: notebooks for analysis, scripts for sampling/reduction, and a comprehensive .env.example + .gitignore.

## Project Structure
- backend/app: FastAPI app (API, core services, models)
- frontend: Vite React UI
- ai_model: embedding, recommendation scoring, feature engineering, notebooks, salary prediction
- dataset: input CSVs
- embeddings: generated embeddings & metadata
- scripts: utilities and tests

## Prerequisites
- Python 3.12, Poetry
- Node.js (LTS), npm

## Setup
1. Copy `.env.example` to `.env` and fill values (e.g., `COHERE_API_KEY`).
2. Install Python deps: `poetry install`.
3. (Optional) Frontend deps: `cd frontend && npm install`.

## Environment
- Embedding (Cohere): `COHERE_API_KEY`, `COHERE_MODEL`, `COHERE_BATCH_SIZE`.
- Optional general embedding vars (if enabled): `DEFAULT_EMBEDDING_PROVIDER`, `DEFAULT_EMBEDDINGS_OUTPUT_DIR`, `DEFAULT_TEXT_COLUMN`, `DEFAULT_JOB_ID_COLUMN`, `LOG_LEVEL`, `EMBEDDING_DEBUG`.
- Optional app: `VITE_API_BASE_URL`, `BACKEND_HOST`, `BACKEND_PORT`.

## Run
- Backend: `poetry run uvicorn backend.app.api.main:app --reload` → http://127.0.0.1:8000/
- Frontend: `cd frontend && npm run dev` → http://127.0.0.1:5173/

## API Overview
- GET `/health` — status + checks for dataset/embeddings/salary artifacts.
- GET `/api/jobs/{job_id}` — job by ID.
- POST `/api/predict_salary` — predicts salary using features and/or 1024-d embedding.
- POST `/api/recommendations` — returns recommendations for given criteria.
- POST `/api/predict_and_recommend` — computes embedding, predicts salary, and returns recommendations.

## Recommendation Logic
- Scorers: Experience, Location, Embedding.
- Defaults: Experience=0.9, Location=0.9 (embedding default defined in code).
- Combine: per-scorer outputs normalized to [0,1], then weighted average to form final score.
- Customize: pass `scoring_weights` in the recommendation layer if you want different weights.

## Data & Embeddings
- Put source CSVs in `dataset/`.
- Generated embeddings and metadata live in `embeddings/`.
- Ensure your embedding provider is configured via `.env`.

## Notebooks & Scripts
- Notebooks: EDA and feature engineering, salary model training.
- Scripts: sampling/reduction (`sample_10k_dataset.py`, `reduce_dataset.py`), embeddings (`process_10k_embeddings.py`), tests (`test_full_recommendation.py`, `test_embedding_sample.py`).

## Testing
- Run: `poetry run pytest -q`.

## Dev Notes
- `.gitignore` excludes env files, caches, build artifacts, datasets, embeddings, logs, and temp files.
- `.env.example` provided — never commit real secrets.
- Avoid committing large data/model artifacts unless required.

## Troubleshooting
- Embedding errors: set `COHERE_API_KEY` and confirm connectivity.
- Missing data: ensure files exist under `dataset/`.
- CORS/ports: adjust `VITE_API_BASE_URL`, `BACKEND_PORT`, then restart dev servers.