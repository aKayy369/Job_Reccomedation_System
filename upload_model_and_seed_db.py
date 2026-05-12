"""
upload_model_and_seed_db.py

Run this AFTER terraform apply to:
  1. Upload your trained model pickle to S3
  2. Seed DynamoDB with job features from dataset.csv

Usage:
    python upload_model_and_seed_db.py \
        --bucket  career-intel-123456789012 \
        --table   career-intel-job-features \
        --dataset data/dataset.csv \
        --model   models/lightgbm_ranker.pkl
"""

import argparse
import json
import ast
import boto3
import pandas as pd
from decimal import Decimal

def upload_model(bucket: str, model_path: str):
    print(f"\n[1/2] Uploading model to S3...")
    s3 = boto3.client("s3")
    s3.upload_file(model_path, bucket, "models/lightgbm_ranker.pkl")
    print(f"  Done → s3://{bucket}/models/lightgbm_ranker.pkl")


def seed_dynamodb(table_name: str, dataset_path: str):
    print(f"\n[2/2] Seeding DynamoDB with job features...")
    df    = pd.read_csv(dataset_path)
    table = boto3.resource("dynamodb").Table(table_name)

    # Get unique jobs with their features
    job_cols = ["job_id", "job_title", "job_min_years", "skill_trend_score"]
    jobs_df  = df[job_cols].drop_duplicates(subset="job_id")

    # Also need skills per job — rebuild from dataset
    # (skills were stored as strings in CSV, parse them back)
    skill_map = {}
    if "job_skills" in df.columns:
        for _, row in df[["job_id", "job_skills"]].drop_duplicates("job_id").iterrows():
            try:
                skill_map[row["job_id"]] = ast.literal_eval(row["job_skills"])
            except Exception:
                skill_map[row["job_id"]] = []

    count = 0
    with table.batch_writer() as batch:
        for _, row in jobs_df.iterrows():
            item = {
                "job_id":       row["job_id"],
                "job_title":    str(row.get("job_title", "Unknown")),
                "min_years":    Decimal(str(int(row["job_min_years"]))),
                "trend_score":  Decimal(str(round(float(row["skill_trend_score"]), 4))),
                "skills":       skill_map.get(row["job_id"], []),
            }
            batch.put_item(Item=item)
            count += 1

    print(f"  Done → {count} jobs written to DynamoDB table '{table_name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",  required=True, help="S3 bucket name from terraform output")
    parser.add_argument("--table",   required=True, help="DynamoDB table name from terraform output")
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--model",   default="models/lightgbm_ranker.pkl")
    args = parser.parse_args()

    upload_model(args.bucket, args.model)
    seed_dynamodb(args.table, args.dataset)

    print("\nAll done! Test your API with:")
    print(f"  python test_api.py --url <your_api_url>")


if __name__ == "__main__":
    main()