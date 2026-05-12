"""
inference.py
Simulates the AWS Lambda inference handler locally.
Load the pickle, pass a feature vector, get top-10 job recommendations.

Usage:
    python inference.py
"""

import os
import pickle
import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")   # suppress sklearn/lgbm version warnings

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_ranker.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "dataset.csv")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"  Model loaded   | best_iter={meta['best_iteration']}")
    print(f"  Features       : {meta['feature_cols']}")
    print(f"  Val  NDCG@10   : {meta['metrics'].get('Val_NDCG@10', 'n/a')}")
    print(f"  Test NDCG@10   : {meta['metrics'].get('Test_NDCG@10', 'n/a')}")
    return model, meta


def recommend(model, meta, candidate_id: str, top_k: int = 10) -> pd.DataFrame:
    df           = pd.read_csv(DATA_PATH)
    feature_cols = meta["feature_cols"]

    cand_df = df[df["candidate_id"] == candidate_id].copy()
    if cand_df.empty:
        print(f"  Candidate {candidate_id} not found.")
        return pd.DataFrame()

    # Pass DataFrame (not .values) so feature names match → no sklearn warning
    cand_df["relevance_score"] = model.predict(cand_df[feature_cols])

    top_jobs = (
        cand_df[["job_id", "job_title", "relevance_score", "label"]]
        .sort_values("relevance_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    top_jobs.index     += 1
    top_jobs.index.name = "rank"
    return top_jobs


def main():
    print("=" * 50)
    print("  Inference Test — simulating Lambda handler")
    print("=" * 50)

    model, meta = load_model()

    test_candidates = ["C0001", "C0025", "C0099"]
    for cid in test_candidates:
        print(f"\n  Top-10 recommendations for {cid}:")
        result = recommend(model, meta, cid, top_k=10)
        if not result.empty:
            print(result.to_string())


if __name__ == "__main__":
    main()