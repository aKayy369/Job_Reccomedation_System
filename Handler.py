"""
lambda_src/handler.py

AWS Lambda inference handler.
Triggered by API Gateway when a user uploads their CV.

Flow:
  POST /upload  →  parse CV  →  build features  →  rank jobs  →  return top-10
  GET  /health  →  return 200 OK
"""

import os
import json
import base64
import pickle
import re
import boto3
import tempfile
from io import BytesIO

# ── Config from environment variables (set in main.tf) ───────────────────────
S3_BUCKET      = os.environ["S3_BUCKET"]
MODEL_S3_KEY   = os.environ["MODEL_S3_KEY"]
MODEL_LOCAL    = os.environ["MODEL_LOCAL"]       # /tmp/lightgbm_ranker.pkl
DYNAMODB_TABLE = os.environ["DYNAMODB_TABLE"]

# ── AWS clients (created once per container, reused on warm invocations) ──────
s3       = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table(DYNAMODB_TABLE)

# ── Model (loaded once per container, cached for warm calls) ──────────────────
_model = None

FEATURE_COLS = [
    "semantic_similarity",
    "skill_overlap",
    "experience_match",
    "skill_trend_score",
    "candidate_years",
    "job_min_years",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    """Download model from S3 to /tmp if not already cached."""
    global _model
    if _model is not None:
        return _model                         # already loaded — warm call
    if not os.path.exists(MODEL_LOCAL):
        print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_S3_KEY}")
        s3.download_file(S3_BUCKET, MODEL_S3_KEY, MODEL_LOCAL)
    with open(MODEL_LOCAL, "rb") as f:
        _model = pickle.load(f)
    print("Model loaded successfully")
    return _model


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


def skill_overlap(candidate_skills, job_skills) -> float:
    if not job_skills:
        return 0.0
    return round(len(set(candidate_skills) & set(job_skills)) / len(job_skills), 4)


def experience_match(candidate_years: int, job_min_years: int) -> float:
    diff = candidate_years - job_min_years
    if diff >= 0:
        return round(min(1.0, 0.7 + 0.1 * min(diff, 3)), 4)
    return round(max(0.0, 0.7 + 0.1 * diff), 4)


def skill_trend(job_skills: list) -> float:
    if not job_skills:
        return 0.5
    return round(sum(TRENDING.get(s, 0.5) for s in job_skills) / len(job_skills), 4)


def parse_resume(pdf_bytes: bytes) -> dict:
    """Extract text, skills and years from raw PDF bytes."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(pdf_bytes))
    except Exception:
        text = pdf_bytes.decode("utf-8", errors="ignore")

    clean = clean_text(text)
    return {
        "raw_text":   text,
        "skills":     extract_skills(clean),
        "years_exp":  extract_years(clean),
    }


def fetch_all_jobs() -> list:
    """
    Scan DynamoDB for all job feature records.
    For 300 jobs this is fast. In production, use Query with filters.
    """
    response = table.scan()
    return response.get("Items", [])


def build_feature_vectors(candidate: dict, jobs: list) -> list:
    """
    Build one feature vector per job for this candidate.
    Returns list of dicts ready for model.predict().
    """
    import numpy as np

    # Simple TF-IDF-like proxy for semantic similarity
    # In production replace with Sentence-BERT embeddings
    candidate_words = set(candidate["raw_text"].lower().split())

    vectors = []
    job_ids = []
    job_titles = []

    for job in jobs:
        job_skills   = job.get("skills", [])
        job_words    = set(" ".join(job_skills).split())
        overlap_words = candidate_words & job_words
        sem_sim      = round(len(overlap_words) / max(len(job_words), 1), 4)
        sem_sim      = min(sem_sim, 1.0)

        vec = {
            "semantic_similarity": sem_sim,
            "skill_overlap":       skill_overlap(candidate["skills"], job_skills),
            "experience_match":    experience_match(
                                       candidate["years_exp"],
                                       int(job.get("min_years", 1))
                                   ),
            "skill_trend_score":   float(job.get("trend_score", 0.5)),
            "candidate_years":     float(candidate["years_exp"]),
            "job_min_years":       float(job.get("min_years", 1)),
        }
        vectors.append([vec[f] for f in FEATURE_COLS])
        job_ids.append(job["job_id"])
        job_titles.append(job.get("job_title", "Unknown"))

    return vectors, job_ids, job_titles


# ── Main handler ──────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Entry point called by API Gateway.

    event["routeKey"] tells us which route was hit:
      GET /health  → return 200
      POST /upload → process CV and return recommendations
    """
    route = event.get("routeKey", "")

    # ── Health check ──────────────────────────────────────────────────────────
    if route == "GET /health":
        return {
            "statusCode": 200,
            "body": json.dumps({"status": "ok", "message": "Lambda is running"})
        }

    # ── CV upload ─────────────────────────────────────────────────────────────
    if route == "POST /upload":
        try:
            # 1. Decode the uploaded PDF
            body = event.get("body", "")
            if event.get("isBase64Encoded", False):
                pdf_bytes = base64.b64decode(body)
            else:
                pdf_bytes = body.encode("utf-8")

            # 2. Parse resume → extract skills + years
            print("Parsing resume...")
            candidate = parse_resume(pdf_bytes)
            print(f"  Skills found : {candidate['skills']}")
            print(f"  Years exp    : {candidate['years_exp']}")

            # 3. Fetch all job features from DynamoDB
            print("Fetching job features from DynamoDB...")
            jobs = fetch_all_jobs()
            if not jobs:
                return {
                    "statusCode": 503,
                    "body": json.dumps({"error": "No jobs in feature store yet."
                                        " Run the job ingestion pipeline first."})
                }
            print(f"  Jobs fetched : {len(jobs)}")

            # 4. Build feature vectors
            vectors, job_ids, job_titles = build_feature_vectors(candidate, jobs)

            # 5. Load model + predict relevance scores
            print("Running model inference...")
            model   = load_model()
            import numpy as np
            scores  = model.predict(np.array(vectors))

            # 6. Rank and return top 10
            ranked  = sorted(zip(scores, job_ids, job_titles), reverse=True)[:10]
            results = [
                {
                    "rank":            i + 1,
                    "job_id":          jid,
                    "job_title":       title,
                    "relevance_score": round(float(score), 4),
                }
                for i, (score, jid, title) in enumerate(ranked)
            ]

            # 7. Save parsed resume to S3 (async in production — sync here for simplicity)
            resume_key = f"processed/resumes/{context.aws_request_id}.json"
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=resume_key,
                Body=json.dumps({
                    "skills":   candidate["skills"],
                    "years_exp": candidate["years_exp"],
                }),
                ContentType="application/json"
            )
            print(f"Resume saved to s3://{S3_BUCKET}/{resume_key}")

            return {
                "statusCode": 200,
                "headers":    {"Content-Type": "application/json"},
                "body":       json.dumps({
                    "candidate_skills": candidate["skills"],
                    "candidate_years":  candidate["years_exp"],
                    "recommendations":  results,
                })
            }

        except Exception as e:
            print(f"ERROR: {e}")
            return {
                "statusCode": 500,
                "body":       json.dumps({"error": str(e)})
            }

    # ── Unknown route ─────────────────────────────────────────────────────────
    return {
        "statusCode": 404,
        "body": json.dumps({"error": f"Route not found: {route}"})
    }