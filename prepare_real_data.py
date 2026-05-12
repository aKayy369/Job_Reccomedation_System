"""
prepare_real_data.py

Processes the two real Kaggle datasets into a training-ready CSV:

  Dataset 1 — Resumes
    https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
    File needed: Resume.csv
    Columns: ID, Resume_str, Resume_html, Category

  Dataset 2 — LinkedIn Job Postings
    https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
    File needed: job_postings.csv
    Columns: job_id, title, description, skills_desc, ...

Usage:
    python src/prepare_real_data.py \
        --resumes  data/raw/Resume.csv \
        --jobs     data/raw/job_postings.csv \
        --output   data/dataset.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Skill pool (ESCO-aligned, covers both datasets) ───────────────────────────
SKILL_KEYWORDS = [
    # Programming
    "python", "java", "scala", "r", "c++", "javascript", "typescript",
    "sql", "bash", "go", "ruby",
    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    # Data Engineering
    "spark", "hadoop", "kafka", "airflow", "dbt", "flink",
    "etl", "data pipeline", "data warehouse",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ci/cd", "git", "linux",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "dynamodb", "snowflake",
    # BI / Analytics
    "tableau", "power bi", "excel", "looker", "pandas", "numpy",
    # Soft / Other
    "communication", "leadership", "agile", "project management",
    "statistics", "probability", "data analysis",
]

# Map resume Category → normalised role name
CATEGORY_MAP = {
    "Data Science":           "Data Scientist",
    "Information-Technology": "Software Engineer",
    "Business-Development":   "Business Analyst",
    "Finance":                "Business Analyst",
    "Banking":                "Business Analyst",
    "Accountant":             "Business Analyst",
    "Engineering":            "Data Engineer",
    "HR":                     "Business Analyst",
    "Healthcare":             "Other",
    "Teacher":                "Other",
    "Advocate":               "Other",
    "Designer":               "Other",
    "Digital-Media":          "Other",
    "Sales":                  "Business Analyst",
    "Consultant":             "Business Analyst",
    "Fitness":                "Other",
    "Agriculture":            "Other",
    "BPO":                    "Other",
    "Automobile":             "Other",
    "Chef":                   "Other",
    "Apparel":                "Other",
    "Construction":           "Other",
    "Public-Relations":       "Other",
    "Arts":                   "Other",
    "Aviation":               "Other",
}

# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_skills(text: str) -> list[str]:
    """Keyword scan against SKILL_KEYWORDS list."""
    text = clean_text(text)
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.append(skill)
    return found


def extract_years_experience(text: str) -> int:
    """Best-effort extraction of years of experience from resume text."""
    text = clean_text(text)
    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+)\+?\s*years?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return min(int(m.group(1)), 20)   # cap at 20
    return 0   # unknown → assume entry-level


def extract_min_years_job(text: str) -> int:
    """Extract minimum required experience from job description."""
    text = clean_text(text)
    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience",
        r"minimum\s+(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return min(int(m.group(1)), 15)
    return 1   # default minimum


# ── Feature engineering ───────────────────────────────────────────────────────

def skill_overlap(candidate_skills: list, job_skills: list) -> float:
    if not job_skills:
        return 0.0
    c, j = set(candidate_skills), set(job_skills)
    return round(len(c & j) / len(j), 4)


def experience_match_score(candidate_years: int, job_min_years: int) -> float:
    diff = candidate_years - job_min_years
    if diff >= 0:
        return round(min(1.0, 0.7 + 0.1 * min(diff, 3)), 4)
    return round(max(0.0, 0.7 + 0.1 * diff), 4)


def skill_trend_score(job_skills: list) -> float:
    """Market-demand weight for job's skills (proxy for real Adzuna trend data)."""
    TRENDING = {
        "python": 0.95, "machine learning": 0.90, "deep learning": 0.88,
        "kubernetes": 0.85, "aws": 0.84, "pytorch": 0.83, "tensorflow": 0.81,
        "docker": 0.79, "spark": 0.76, "nlp": 0.75, "kafka": 0.73,
        "airflow": 0.72, "dbt": 0.71, "data pipeline": 0.70,
    }
    if not job_skills:
        return 0.5
    return round(np.mean([TRENDING.get(s, 0.50) for s in job_skills]), 4)


# ── TF-IDF semantic similarity (proxy for Sentence-BERT) ─────────────────────

def build_tfidf_similarity(resume_texts: list[str],
                            job_texts: list[str]) -> np.ndarray:
    """
    Returns (n_resumes × n_jobs) cosine similarity matrix using TF-IDF.
    In production this is replaced by Sentence-BERT embeddings + cosine.
    """
    print("  Computing TF-IDF cosine similarity matrix...")
    combined   = resume_texts + job_texts
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  stop_words="english")
    tfidf      = vectorizer.fit_transform(combined)

    resume_vecs = tfidf[:len(resume_texts)]
    job_vecs    = tfidf[len(resume_texts):]

    sim_matrix  = cosine_similarity(resume_vecs, job_vecs)   # shape (R, J)
    return sim_matrix


# ── Relevance label (graded 0-3 for LambdaRank) ──────────────────────────────

def make_label(sem_sim: float, overlap: float,
               exp_match: float, trend: float) -> int:
    score = (overlap * 0.40) + (sem_sim * 0.35) + (exp_match * 0.15) + (trend * 0.10)
    if score >= 0.65:
        return 3
    elif score >= 0.45:
        return 2
    elif score >= 0.25:
        return 1
    return 0


# ── Load & preprocess resumes ─────────────────────────────────────────────────

def load_resumes(path: str, max_resumes: int = 500) -> pd.DataFrame:
    print(f"\n[Resumes] Loading {path}...")
    df = pd.read_csv(path)

    # Handle both possible column name casings
    df.columns = [c.strip() for c in df.columns]
    if "Resume_str" not in df.columns and "resume_str" in df.columns:
        df.rename(columns={"resume_str": "Resume_str",
                            "category": "Category", "id": "ID"}, inplace=True)

    df = df.dropna(subset=["Resume_str", "Category"])
    df = df.head(max_resumes).reset_index(drop=True)

    df["clean_text"]  = df["Resume_str"].apply(clean_text)
    df["skills"]      = df["clean_text"].apply(extract_skills)
    df["years_exp"]   = df["clean_text"].apply(extract_years_experience)
    df["role"]        = df["Category"].map(CATEGORY_MAP).fillna("Other")
    df["candidate_id"] = ["C" + str(i).zfill(4) for i in range(len(df))]

    print(f"  Loaded {len(df)} resumes | roles: {df['role'].value_counts().to_dict()}")
    print(f"  Avg skills per resume: {df['skills'].apply(len).mean():.1f}")
    print(f"  Avg years exp: {df['years_exp'].mean():.1f}")
    return df[["candidate_id", "clean_text", "skills", "years_exp", "role"]]


# ── Load & preprocess job postings ────────────────────────────────────────────

def load_jobs(path: str, max_jobs: int = 300) -> pd.DataFrame:
    print(f"\n[Jobs] Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # LinkedIn dataset columns: job_id, title, description, skills_desc
    # Fallback: look for any description-like column
    desc_col = next(
        (c for c in ["description", "job_description", "job description",
                     "skills_desc", "requirements"] if c in df.columns),
        None
    )
    title_col = next(
        (c for c in ["title", "job_title", "position"] if c in df.columns),
        None
    )
    if desc_col is None or title_col is None:
        raise ValueError(
            f"Could not find title/description columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[desc_col]).head(max_jobs).reset_index(drop=True)
    df["clean_desc"]  = df[desc_col].apply(clean_text)
    df["job_title"]   = df[title_col].fillna("Unknown").str.strip()
    df["skills"]      = df["clean_desc"].apply(extract_skills)
    df["min_years"]   = df["clean_desc"].apply(extract_min_years_job)
    df["trend_score"] = df["skills"].apply(skill_trend_score)
    df["job_id"]      = ["J" + str(i).zfill(4) for i in range(len(df))]

    print(f"  Loaded {len(df)} jobs")
    print(f"  Avg skills per job: {df['skills'].apply(len).mean():.1f}")
    return df[["job_id", "job_title", "clean_desc", "skills", "min_years", "trend_score"]]


# ── Build candidate-job feature dataset ──────────────────────────────────────

def build_feature_dataset(resumes: pd.DataFrame,
                           jobs: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[Features] Building {len(resumes)} × {len(jobs)} = "
          f"{len(resumes) * len(jobs):,} candidate-job pairs...")

    # TF-IDF similarity matrix (R × J)
    sim_matrix = build_tfidf_similarity(
        resumes["clean_text"].tolist(),
        jobs["clean_desc"].tolist()
    )

    rows = []
    for r_idx, resume in resumes.iterrows():
        for j_idx, job in jobs.iterrows():
            sem_sim   = round(float(sim_matrix[r_idx, j_idx]), 4)
            overlap   = skill_overlap(resume["skills"], job["skills"])
            exp_match = experience_match_score(resume["years_exp"], job["min_years"])
            trend     = job["trend_score"]
            label     = make_label(sem_sim, overlap, exp_match, trend)

            rows.append({
                "candidate_id":        resume["candidate_id"],
                "job_id":              job["job_id"],
                "job_title":           job["job_title"],
                "semantic_similarity": sem_sim,
                "skill_overlap":       overlap,
                "experience_match":    exp_match,
                "skill_trend_score":   trend,
                "candidate_years":     resume["years_exp"],
                "job_min_years":       job["min_years"],
                "label":               label,
            })

    df = pd.DataFrame(rows)
    print(f"  Done. Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare real Kaggle datasets for training")
    parser.add_argument("--resumes",  default="data/raw/Resume.csv",
                        help="Path to Resume.csv from Kaggle resume dataset")
    parser.add_argument("--jobs",     default="data/raw/job_postings.csv",
                        help="Path to job_postings.csv from LinkedIn Kaggle dataset")
    parser.add_argument("--output",   default="data/dataset.csv",
                        help="Output path for feature dataset")
    parser.add_argument("--max-resumes", type=int, default=500)
    parser.add_argument("--max-jobs",    type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("  Real-Time Career Intelligence Platform")
    print("  Real Dataset Preparation Pipeline")
    print("=" * 60)

    resumes = load_resumes(args.resumes,  max_resumes=args.max_resumes)
    jobs    = load_jobs(args.jobs,         max_jobs=args.max_jobs)
    df      = build_feature_dataset(resumes, jobs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n  Saved → {args.output}  ({len(df):,} rows)")
    print("\n  Next step: python src/train.py")


if __name__ == "__main__":
    main()