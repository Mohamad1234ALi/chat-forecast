from pathlib import Path
import os, json, datetime
import boto3
from dotenv import load_dotenv, dotenv_values
import streamlit as st

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET  = os.getenv("S3_BUCKET")                
LOCAL_PATH = Path("data/embeddings.json")
KEY_PREFIX = "talk2data/embeddings"               

assert S3_BUCKET, "S3_Bucket missed in .env"
assert LOCAL_PATH.exists(), f"{LOCAL_PATH} not found"


stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
key = f"{KEY_PREFIX}/embeddings.json" #key = f"{KEY_PREFIX}/embeddings-{stamp}.json"
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=AWS_REGION
)

#print("Werte aus der .env-Datei:", dotenv_values(".env"))  # Zeigt alle Werte aus der .env-Datei
#print("Geladener AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))  # Zeigt den geladenen Wert

s3.upload_file(
    Filename=str(LOCAL_PATH),
    Bucket=S3_BUCKET,    
    Key=key,
    ExtraArgs={"ContentType": "application/json"}
)

print(f"✅ Hochgeladen: s3://{S3_BUCKET}/{key}")