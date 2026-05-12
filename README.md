# Real-Time Career Intelligence Platform
### End-to-End Data Engineering and MLOps Pipeline for Resume–Job Matching

**Authors:** Anubhav Kharel (st125999) · Samir Pokharel (st125989)  
**Course:** Selected Topic: Data Engineering and MLOps (2026)  
**Instructor:** Dr. Chantri Polpasert  
**Institution:** Asian Institute of Technology, Thailand

---

## Project Overview

This project implements a Real-Time Career Intelligence Platform that performs automated resume–job matching using a scalable data engineering and MLOps pipeline. Users upload a PDF resume through a web dashboard and receive top-10 ranked job recommendations instantly, based on semantic similarity, skill overlap, experience compatibility, and market demand signals.

---

## System Architecture

```
User (browser)
     │
     ▼
EC2 t2.micro ── FastAPI Dashboard (app.py + templates/index.html)
     │                    │
     │                    ▼
     │           Lambda (Handler.py) ──── DynamoDB (300 job features)
     │                    │
     ▼                    ▼
S3 Bucket (career-intel-bitwise-v2)
├── raw/
│   ├── resumes/          ← Kaggle resume dataset
│   └── jobs/             ← LinkedIn job postings
├── processed/
│   └── resumes/          ← parsed CVs written by Lambda
├── features/
│   └── dataset.csv       ← 150,000 candidate-job pairs
├── models/
│   └── lightgbm_ranker.pkl ← trained LightGBM model
├── app/
│   ├── app.py            ← FastAPI application
│   └── index.html        ← dashboard UI
└── metrics/
    ├── skill_trends/     ← current skill distribution
    └── skill_distribution_baseline/ ← PSI baseline

CloudWatch ── 3 alarms (Lambda errors, API latency, EC2 CPU)
     │
     ▼
SNS ── email alerts → st125989@ait.asia

Drift Detection (cron every Sunday)
     └── drift_detector.py ── PSI threshold 0.25
```

---

## Datasets

### Resume Dataset
- **Source:** [Kaggle — snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Size:** 2,484 resumes across 25 job categories
- **Used:** 500 resumes for training
- **Key column:** `Resume_str` — full resume text

### Job Postings Dataset
- **Source:** [Kaggle — arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **Size:** 504 MB of real LinkedIn job postings
- **Used:** 300 job postings
- **Key columns:** `title`, `description`

### Skill Taxonomy
- 50+ ESCO-aligned skill keywords
- Market demand weights assigned per skill (Python: 0.95, ML: 0.90, etc.)

---

## Local Setup

### Prerequisites
```
Python 3.11+
pip
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Prepare data
```bash
python prepare_real_data.py \
    --resumes data/raw/Resume.csv \
    --jobs    data/raw/postings.csv \
    --output  data/dataset.csv
```

**Output:** `data/dataset.csv` — 150,000 candidate-job pairs with labels

### Step 2 — Train model
```bash
python train.py
```

**Output:** `models/lightgbm_ranker.pkl` + `models/model_metadata.json`

**Results:**
| Metric | Score |
|---|---|
| Val NDCG@10 | 0.9011 |
| Test NDCG@10 | 0.8886 |
| Best iteration | 20 |

### Step 3 — Test inference locally
```bash
python inference.py
```

### Step 4 — Run web dashboard locally
```bash
python app.py
```
Open `http://127.0.0.1:8000`

---

## Feature Engineering

Each candidate-job pair is represented by a 6-dimensional feature vector:

| Feature | Description | Source |
|---|---|---|
| `semantic_similarity` | TF-IDF cosine similarity between resume and job text | prepare_real_data.py |
| `skill_overlap` | Fraction of job's required skills in candidate resume | keyword extraction |
| `experience_match` | Score based on candidate years vs job minimum | regex extraction |
| `skill_trend_score` | Market demand weight for job's required skills | ESCO-aligned weights |
| `candidate_years` | Years of experience from resume | regex extraction |
| `job_min_years` | Minimum years required from job description | regex extraction |

### Relevance Label Definition
```
Score = 0.40 × skill_overlap
      + 0.35 × semantic_similarity
      + 0.15 × experience_match
      + 0.10 × skill_trend_score

Label 3 (Highly Relevant) : score ≥ 0.65
Label 2 (Relevant)        : score ≥ 0.45
Label 1 (Marginal)        : score ≥ 0.25
Label 0 (Irrelevant)      : score < 0.25
```

---

## Model Training

**Algorithm:** LightGBM LambdaRank  
**Objective:** Directly optimise NDCG ranking metric

### Key Hyperparameters
| Parameter | Value |
|---|---|
| objective | lambdarank |
| metric | ndcg |
| ndcg_eval_at | [5, 10] |
| learning_rate | 0.05 |
| n_estimators | 300 |
| num_leaves | 31 |
| subsample | 0.8 |
| early_stopping | 30 rounds |

### Training Strategy
- Split by candidate ID to prevent data leakage
- 70% train / 15% val / 15% test
- Model saved as pickle for AWS inference

---

## AWS Cloud Deployment

### AWS Resources

| Resource | Name | Purpose |
|---|---|---|
| S3 Bucket | career-intel-bitwise-v2 | Data lake + model storage |
| DynamoDB | career-intel-job-features | Online feature store (300 jobs) |
| EC2 | career-intel-web (t2.micro) | FastAPI web dashboard |
| Lambda | career-intel-inference | Serverless inference API |
| API Gateway | career-intel-api | Public HTTPS endpoint |
| CloudWatch | 3 alarms | Monitoring |
| SNS | career-intel-alerts | Email notifications |

### Deployment Steps

#### 1. Upload files to S3
```bash
# Create folder structure
aws s3api put-object --bucket career-intel-bitwise-v2 --key raw/resumes/
aws s3api put-object --bucket career-intel-bitwise-v2 --key raw/jobs/
aws s3api put-object --bucket career-intel-bitwise-v2 --key processed/resumes/
aws s3api put-object --bucket career-intel-bitwise-v2 --key features/
aws s3api put-object --bucket career-intel-bitwise-v2 --key models/
aws s3api put-object --bucket career-intel-bitwise-v2 --key app/
aws s3api put-object --bucket career-intel-bitwise-v2 --key metrics/skill_trends/
aws s3api put-object --bucket career-intel-bitwise-v2 --key metrics/skill_distribution_baseline/

# Upload files
aws s3 cp models/lightgbm_ranker.pkl s3://career-intel-bitwise-v2/models/
aws s3 cp data/dataset.csv           s3://career-intel-bitwise-v2/features/
aws s3 cp app.py                     s3://career-intel-bitwise-v2/app/
aws s3 cp templates/index.html       s3://career-intel-bitwise-v2/app/
aws s3 cp data/raw/Resume.csv        s3://career-intel-bitwise-v2/raw/resumes/
aws s3 cp data/raw/postings.csv      s3://career-intel-bitwise-v2/raw/jobs/
```

#### 2. Create DynamoDB table and seed jobs
```bash
aws dynamodb create-table \
    --table-name career-intel-job-features \
    --attribute-definitions AttributeName=job_id,AttributeType=S \
    --key-schema AttributeName=job_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

python3 -c "
import boto3, pandas as pd
from decimal import Decimal
df = pd.read_csv('dataset.csv')
table = boto3.resource('dynamodb', region_name='us-east-1').Table('career-intel-job-features')
jobs = df[['job_id','job_title','job_min_years','skill_trend_score']].drop_duplicates('job_id')
with table.batch_writer() as batch:
    for _, row in jobs.iterrows():
        batch.put_item(Item={
            'job_id':      row['job_id'],
            'job_title':   str(row.get('job_title','Unknown')),
            'min_years':   Decimal(str(int(row['job_min_years']))),
            'trend_score': Decimal(str(round(float(row['skill_trend_score']),4))),
        })
print('Done → 300 jobs written')
"
```

#### 3. Launch EC2 and run app
```bash
# On EC2 after connecting via EC2 Instance Connect
sudo yum update -y
sudo yum install python3-pip cronie -y
pip3 install fastapi uvicorn python-multipart pdfminer.six lightgbm scikit-learn pandas numpy jinja2 boto3

mkdir -p ~/career_app/templates ~/career_app/models ~/career_app/data
aws s3 cp s3://career-intel-bitwise-v2/app/app.py          ~/career_app/
aws s3 cp s3://career-intel-bitwise-v2/app/index.html      ~/career_app/templates/
aws s3 cp s3://career-intel-bitwise-v2/models/lightgbm_ranker.pkl ~/career_app/models/
aws s3 cp s3://career-intel-bitwise-v2/features/dataset.csv ~/career_app/data/

# Run with screen
screen -S careerapp
cd ~/career_app && python3 app.py
# Ctrl+A then D to detach
```

#### 4. Set up monitoring
```bash
# Create SNS topic
aws sns create-topic --name career-intel-alerts --region us-east-1
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:ACCOUNT_ID:career-intel-alerts \
    --protocol email \
    --notification-endpoint YOUR_EMAIL

# Create CloudWatch alarms
aws cloudwatch put-metric-alarm \
    --alarm-name "Lambda-Errors" \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --dimensions Name=FunctionName,Value=career-intel-inference \
    --statistic Sum --period 300 --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:career-intel-alerts \
    --region us-east-1

aws cloudwatch put-metric-alarm \
    --alarm-name "API-Latency-High" \
    --metric-name Latency \
    --namespace AWS/ApiGateway \
    --statistic Average --period 300 --threshold 25000 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:career-intel-alerts \
    --region us-east-1

aws cloudwatch put-metric-alarm \
    --alarm-name "EC2-CPU-High" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --dimensions Name=InstanceId,Value=YOUR_INSTANCE_ID \
    --statistic Average --period 300 --threshold 80 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:career-intel-alerts \
    --region us-east-1
```

#### 5. Set up drift detection
```bash
# Run once to create baseline
python3 drift_detector.py

# Add weekly cron job (every Sunday midnight)
sudo systemctl start crond && sudo systemctl enable crond
(crontab -l 2>/dev/null; echo "0 0 * * 0 python3 /home/ec2-user/career_app/drift_detector.py >> /home/ec2-user/career_app/drift.log 2>&1") | crontab -
```

---

## Monitoring

### CloudWatch Alarms
| Alarm | Metric | Threshold | Action |
|---|---|---|---|
| Lambda-Errors | Lambda Errors | ≥ 1 | SNS email |
| API-Latency-High | API Gateway Latency | ≥ 25,000ms | SNS email |
| EC2-CPU-High | EC2 CPUUtilization | ≥ 80% | SNS email |

### Drift Detection (PSI)
- Runs every Sunday via cron
- Computes Population Stability Index between baseline and current skill distributions
- Sends SNS alert if PSI > 0.25
- Baseline stored at: `s3://career-intel-bitwise-v2/metrics/skill_distribution_baseline/baseline.json`
- Current stored at: `s3://career-intel-bitwise-v2/metrics/skill_trends/current.json`

---

## Demo Instructions

### Starting the demo
1. Start Learner Lab session (wait for green dot)
2. Go to EC2 → Start instance if stopped
3. Connect via EC2 Instance Connect
4. Check app: `screen -r careerapp`
5. If gone, restart: `cd ~/career_app && screen -S careerapp && python3 app.py`
6. Get IP: `curl ifconfig.me`
7. Open `http://EC2_IP:8000`

### Demo flow
1. Show architecture diagram
2. Show S3 bucket structure in AWS Console
3. Show DynamoDB table with job features
4. Show CloudWatch alarms
5. Open dashboard → upload a PDF resume
6. Show detected skills + market demand chart
7. Show top-10 ranked job recommendations

---

## Project File Structure

```
mlops_project/
├── app.py                      ← FastAPI web application
├── templates/
│   └── index.html              ← dashboard UI
├── prepare_real_data.py        ← ETL pipeline (local)
├── train.py                    ← LightGBM training (local)
├── inference.py                ← local inference test
├── Handler.py                  ← AWS Lambda handler
├── drift_detector.py           ← PSI drift detection (EC2)
├── upload_model_and_seed_db.py ← S3 + DynamoDB seeding
├── models/
│   ├── lightgbm_ranker.pkl     ← trained model
│   └── model_metadata.json     ← training metrics
├── data/
│   ├── raw/
│   │   ├── Resume.csv          ← Kaggle resume dataset
│   │   └── postings.csv        ← LinkedIn job postings
│   └── dataset.csv             ← generated feature dataset
└── requirements.txt
```

---

## References

- Khelkhal, K., & Lanasri, D. (2025). Smart-hiring: End-to-end CV information extraction and job matching pipeline. arXiv preprint.
- Jiang, J., et al. (2020). Learning effective representations for person-job fit. arXiv.
- Bian, S., et al. (2020). Learning to match jobs with resumes from sparse interaction data. Proceedings of the Web Conference.
- Alonso, R., et al. (2025). Job matching and skill identification using deep learning. Expert Systems with Applications.
- Baylor, D., et al. (2017). TFX: A TensorFlow-based production-scale machine learning platform. KDD.