"""
train.py
Trains a LightGBM LambdaRank model on the real career dataset.

Pipeline:
  1. Load dataset
  2. Split by candidate (query) — train / val / test
  3. Build LightGBM Dataset with group sizes
  4. Train with LambdaRank objective
  5. Evaluate with NDCG@10, Precision@10, Recall@10
  6. Save model as pickle for AWS Lambda inference
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths (relative to wherever train.py lives) ───────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_ranker.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_metadata.json")

FEATURE_COLS = [
    "semantic_similarity",
    "skill_overlap",
    "experience_match",
    "skill_trend_score",
    "candidate_years",
    "job_min_years",
]
LABEL_COL = "label"
QUERY_COL = "candidate_id"

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [5, 10],
    "boosting_type":     "gbdt",
    "num_leaves":        31,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      300,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbose":           -1,
}

# ── Evaluation helpers ────────────────────────────────────────────────────────

def precision_at_k(labels_sorted: np.ndarray, k: int = 10, threshold: int = 2) -> float:
    top_k = labels_sorted[:k]
    return float(np.sum(top_k >= threshold) / k)


def recall_at_k(labels_sorted: np.ndarray, all_labels: np.ndarray,
                k: int = 10, threshold: int = 2) -> float:
    total_relevant = np.sum(all_labels >= threshold)
    if total_relevant == 0:
        return 0.0
    return float(np.sum(labels_sorted[:k] >= threshold) / total_relevant)


def ndcg_at_k(labels_sorted: np.ndarray, k: int = 10) -> float:
    top_k   = labels_sorted[:k]
    ideal_k = np.sort(labels_sorted)[::-1][:k]
    gains   = (2 ** top_k   - 1) / np.log2(np.arange(2, len(top_k)   + 2))
    ideal_g = (2 ** ideal_k - 1) / np.log2(np.arange(2, len(ideal_k) + 2))
    dcg, idcg = np.sum(gains), np.sum(ideal_g)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(model, df: pd.DataFrame, split_name: str = "Test") -> dict:
    scores       = model.predict(df[FEATURE_COLS])
    df           = df.copy()
    df["pred_score"] = scores
    p10, r10, n10 = [], [], []

    for _, grp in df.groupby(QUERY_COL):
        grp_sorted = grp.sort_values("pred_score", ascending=False)
        labels     = grp_sorted[LABEL_COL].values
        p10.append(precision_at_k(labels))
        r10.append(recall_at_k(labels, grp[LABEL_COL].values))
        n10.append(ndcg_at_k(labels))

    metrics = {
        f"{split_name}_Precision@10": round(np.mean(p10), 4),
        f"{split_name}_Recall@10":    round(np.mean(r10), 4),
        f"{split_name}_NDCG@10":      round(np.mean(n10), 4),
    }
    print(f"\n  [{split_name} Evaluation]")
    for k, v in metrics.items():
        print(f"    {k:<30} {v:.4f}")
    return metrics

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"  Reading: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows | "
          f"{df[QUERY_COL].nunique()} candidates | "
          f"{df['job_id'].nunique()} jobs")
    print(f"  Label distribution:\n{df[LABEL_COL].value_counts().sort_index().to_string()}")
    return df

# ── Train / Val / Test split by candidate (no leakage) ───────────────────────

def split_by_query(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    all_qids = df[QUERY_COL].unique()
    train_q, temp_q = train_test_split(all_qids,
                                        test_size=val_frac + test_frac,
                                        random_state=42)
    val_q, test_q   = train_test_split(temp_q,
                                        test_size=test_frac / (val_frac + test_frac),
                                        random_state=42)
    df_train = df[df[QUERY_COL].isin(train_q)].copy()
    df_val   = df[df[QUERY_COL].isin(val_q)].copy()
    df_test  = df[df[QUERY_COL].isin(test_q)].copy()

    print(f"\n  Train : {len(train_q)} candidates  ({len(df_train):,} pairs)")
    print(f"  Val   : {len(val_q)} candidates  ({len(df_val):,} pairs)")
    print(f"  Test  : {len(test_q)} candidates  ({len(df_test):,} pairs)")
    return df_train, df_val, df_test

# ── Build LightGBM Dataset ────────────────────────────────────────────────────

def build_lgbm_dataset(df: pd.DataFrame, reference=None):
    df_s        = df.sort_values(QUERY_COL)
    group_sizes = df_s.groupby(QUERY_COL, sort=True).size().tolist()
    X = df_s[FEATURE_COLS].values
    y = df_s[LABEL_COL].values.astype(int)
    dataset = lgb.Dataset(X, label=y, group=group_sizes,
                          feature_name=FEATURE_COLS,
                          free_raw_data=False,
                          reference=reference)
    return dataset, df_s

# ── Training ──────────────────────────────────────────────────────────────────

def train(df_train: pd.DataFrame, df_val: pd.DataFrame) -> lgb.LGBMRanker:
    print("\nBuilding LightGBM datasets...")
    train_set, df_train_s = build_lgbm_dataset(df_train)
    val_set,   df_val_s   = build_lgbm_dataset(df_val, reference=train_set)

    model = lgb.LGBMRanker(**LGBM_PARAMS)

    print(f"\nTraining LightGBM LambdaRank  "
          f"(n_estimators={LGBM_PARAMS['n_estimators']})...")

    model.fit(
        df_train_s[FEATURE_COLS].values,
        df_train_s[LABEL_COL].values.astype(int),
        group=train_set.get_group(),
        eval_set=[(df_val_s[FEATURE_COLS].values,
                   df_val_s[LABEL_COL].values.astype(int))],
        eval_group=[val_set.get_group()],
        eval_metric=["ndcg"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"\n  Best iteration : {model.best_iteration_}")
    return model

# ── Save model + metadata ─────────────────────────────────────────────────────

def save_model(model: lgb.LGBMRanker, metrics: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved    → {MODEL_PATH}")

    metadata = {
        "model_type":     "LightGBM LambdaRank",
        "feature_cols":   FEATURE_COLS,
        "label_col":      LABEL_COL,
        "hyperparameters": LGBM_PARAMS,
        "best_iteration": int(model.best_iteration_),
        "metrics":        metrics,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {META_PATH}")

# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(model: lgb.LGBMRanker):
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    max_imp    = max(importance.values()) or 1
    print("\n  [Feature Importance]")
    for feat, imp in importance.items():
        bar = "█" * int(imp / max_imp * 20)
        print(f"    {feat:<25} {imp:>6}  {bar}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Real-Time Career Intelligence Platform")
    print("  LightGBM LambdaRank — Local Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data(DATA_PATH)

    print("\n[2/5] Splitting dataset by candidate (query)...")
    df_train, df_val, df_test = split_by_query(df)

    print("\n[3/5] Training...")
    model = train(df_train, df_val)

    print("\n[4/5] Evaluating...")
    val_metrics  = evaluate(model, df_val,  split_name="Val")
    test_metrics = evaluate(model, df_test, split_name="Test")

    print("\n[5/5] Saving model...")
    save_model(model, {**val_metrics, **test_metrics})

    print_feature_importance(model)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("  Next: upload models/lightgbm_ranker.pkl to AWS S3")
    print("=" * 60)


if __name__ == "__main__":
    main()