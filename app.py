"""
app.py — Real-Time Career Intelligence Platform
FastAPI local server with dashboard UI

Run:
    pip install fastapi uvicorn python-multipart pdfminer.six lightgbm scikit-learn pandas
    python app.py
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_ranker.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "dataset.csv")

FEATURE_COLS = [
    "semantic_similarity", "skill_overlap", "experience_match",
    "skill_trend_score",   "candidate_years", "job_min_years",
]

SKILL_KEYWORDS = [
    "python", "java", "scala", "sql", "r", "javascript", "typescript",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "spark", "hadoop", "kafka", "airflow", "dbt", "etl", "data pipeline",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git",
    "postgresql", "mysql", "mongodb", "redis", "dynamodb", "snowflake",
    "tableau", "power bi", "excel", "pandas", "numpy", "statistics",
    "communication", "leadership", "agile", "project management",
]

TRENDING = {
    "python": 0.95, "machine learning": 0.90, "deep learning": 0.88,
    "kubernetes": 0.85, "aws": 0.84, "pytorch": 0.83, "tensorflow": 0.81,
    "docker": 0.79, "spark": 0.76, "nlp": 0.75, "kafka": 0.73,
    "airflow": 0.72, "dbt": 0.71, "data pipeline": 0.70,
}

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
print("Model loaded.")

# Load job data once
print("Loading job data...")
_df = pd.read_csv(DATA_PATH)
JOBS_DF = _df[["job_id", "job_title", "job_min_years", "skill_trend_score"]].drop_duplicates("job_id").reset_index(drop=True)
print(f"Loaded {len(JOBS_DF)} unique jobs.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Career Intelligence Platform")
templates = Jinja2Templates(directory="templates")
# ── Helper functions ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def extract_skills(text: str) -> list:
    return [s for s in SKILL_KEYWORDS if re.search(r"\b" + re.escape(s) + r"\b", text)]


def extract_years(text: str) -> int:
    for pat in [r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
                r"experience\s+of\s+(\d+)\+?\s*years?"]:
        m = re.search(pat, text)
        if m:
            return min(int(m.group(1)), 20)
    return 0


def parse_pdf(pdf_bytes: bytes) -> dict:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(pdf_bytes))
    except Exception:
        text = pdf_bytes.decode("utf-8", errors="ignore")
    clean = clean_text(text)
    return {"raw_text": clean, "skills": extract_skills(clean), "years_exp": extract_years(clean)}


def skill_overlap(candidate_skills, job_skills) -> float:
    if not job_skills:
        return 0.0
    return round(len(set(candidate_skills) & set(job_skills)) / len(job_skills), 4)


def experience_match(cand_years: int, job_min: int) -> float:
    diff = cand_years - job_min
    if diff >= 0:
        return round(min(1.0, 0.7 + 0.1 * min(diff, 3)), 4)
    return round(max(0.0, 0.7 + 0.1 * diff), 4)


def get_job_skills(job_id: str) -> list:
    rows = _df[_df["job_id"] == job_id]
    if rows.empty:
        return []
    # skills stored in candidate_skills column as proxy
    return []


def build_features(candidate: dict) -> np.ndarray:
    rows = []
    candidate_words = set(candidate["raw_text"].split())
    for _, job in JOBS_DF.iterrows():
        job_words  = set(job["job_title"].lower().split())
        sem_sim    = round(len(candidate_words & job_words) / max(len(job_words), 1), 4)
        sem_sim    = min(sem_sim, 1.0)
        rows.append([
            sem_sim,
            0.3,                              # skill_overlap placeholder
            experience_match(candidate["years_exp"], int(job["job_min_years"])),
            float(job["skill_trend_score"]),
            float(candidate["years_exp"]),
            float(job["job_min_years"]),
        ])
    return np.array(rows)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded", "jobs": len(JOBS_DF)}


@app.post("/upload")
async def upload_cv(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        candidate = parse_pdf(pdf_bytes)
        features  = build_features(candidate)
        scores    = MODEL.predict(features)

        # Rank top 10
        top_idx   = np.argsort(scores)[::-1][:10]
        recommendations = []
        for rank, idx in enumerate(top_idx, 1):
            job = JOBS_DF.iloc[idx]
            recommendations.append({
                "rank":            rank,
                "job_id":          job["job_id"],
                "job_title":       job["job_title"],
                "relevance_score": round(float(scores[idx]), 4),
                "min_years":       int(job["job_min_years"]),
                "trend_score":     round(float(job["skill_trend_score"]), 4),
            })

        # Skill market demand for chart
        skill_demand = {s: round(TRENDING.get(s, 0.5), 2) for s in candidate["skills"]}

        return JSONResponse({
            "filename":        file.filename,
            "candidate_skills": candidate["skills"],
            "candidate_years":  candidate["years_exp"],
            "skill_demand":     skill_demand,
            "recommendations":  recommendations,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)