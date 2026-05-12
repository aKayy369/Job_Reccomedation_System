"""
test_api.py
Tests your deployed API Gateway endpoint.

Usage:
    python test_api.py --url https://abc123.execute-api.us-east-1.amazonaws.com
"""

import argparse
import requests
import json

def test_health(base_url: str):
    print("\n[1] Health check...")
    r = requests.get(f"{base_url}/health")
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {r.json()}")


def test_upload(base_url: str, cv_path: str):
    print(f"\n[2] Uploading CV: {cv_path}")
    with open(cv_path, "rb") as f:
        r = requests.post(
            f"{base_url}/upload",
            data=f.read(),
            headers={"Content-Type": "application/pdf"},
        )
    print(f"  Status : {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        print(f"  Skills found   : {body.get('candidate_skills', [])}")
        print(f"  Years exp      : {body.get('candidate_years', 0)}")
        print(f"\n  Top-10 recommendations:")
        for job in body.get("recommendations", []):
            print(f"    {job['rank']:>2}. {job['job_title']:<40} score={job['relevance_score']}")
    else:
        print(f"  Error: {r.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="API Gateway base URL from terraform output")
    parser.add_argument("--cv",  default=None,  help="Path to a PDF resume to test upload")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    test_health(base_url)

    if args.cv:
        test_upload(base_url, args.cv)
    else:
        print("\n  Tip: pass --cv path/to/resume.pdf to test the full upload flow")


if __name__ == "__main__":
    main()