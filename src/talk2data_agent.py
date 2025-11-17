import streamlit as st
import requests, os, json
import re
from dotenv import load_dotenv, dotenv_values
import os
import json
import numpy as np
import boto3
from openai import OpenAI
import logging
import pathlib
from sklearn.metrics.pairwise import cosine_similarity


# Path configuration
SCRIPT_DIR = pathlib.Path(__file__).parent.parent
QUERIES_PATH = SCRIPT_DIR / "data" / "queries.json"
PROMPT_PATH = SCRIPT_DIR / "prompts" / "query_selector.txt"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
S3_EMBEDDINGS_KEY = os.getenv(
    "S3_EMBEDDINGS_KEY", 
    "talk2data/embeddings/embeddings.json"
)
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_s3_client = None
_embeddings_cache = None

# setup
load_dotenv()

# Force load values from .env file to override system environment
env_values = dotenv_values(".env")
for key, value in env_values.items():
    if value:
        os.environ[key] = value

# Validate and initialize OpenAI client
#_openai_api_key = os.getenv("OPENAI_API_KEY")
_openai_api_key = st.secrets["OPENAI_API_KEY"]
# You can also put this in st.secrets["TALK2DATA_API_URL"]
TALK2DATA_API_URL= os.getenv("TALK2DATA_API_URL")
if not _openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=_openai_api_key)
logger.info("✅ OpenAI client initialized")


def get_s3_client():
    """
    Get or create a cached S3 client instance.
    
    Returns:
        boto3.client: Configured S3 client.
        
    Raises:
        ValueError: If required AWS credentials are missing.
    """
    global _s3_client
    if _s3_client is None:
        # Validate required credentials
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required AWS credentials: {', '.join(missing_vars)}")
        
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        logger.info("✅ S3 client initialized")
    return _s3_client
    

def summarize_results(user_question, sql_query, rows):
     
    """
    Summarize query results into a human-readable answer using LLM.
    
    Args:
        user_question (str): The user's question text.
        sql_query (str): The SQL query that was executed.
        rows (list[dict]): List of result rows from the query.
        
    Returns:
        str: Human-readable summary of the results.
        
    Raises:
        ValueError: If rows is empty.
    """
     
    prompt = f"""
You are a helpful analyst. User asked: "{user_question}"
Query {sql_query} returned {len(rows)} rows:
{json.dumps(rows, default=str, indent=2)}

Write a clear, human-readable answer that:
- Mentions *every store in the result list* (do not skip any rows)
- Includes a headline summarizing what the table shows
- Highlights main insights or trends (if inferable)
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content.strip()


def call_sql_api(sql_query: str):
    """
    Call the Athena query API via API Gateway.
    Args:
        sql_query (str): The SQL query to execute.
    Returns:
        dict: Parsed JSON response from the API.
    """

    payload = {"sql_query": sql_query}
    # st.write("📤 Sending payload:", json.dumps(payload, indent=2))

    resp = requests.post(
        API_GATEWAY_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    # st.write("🔍 Full API response text:", resp.text)

    resp.raise_for_status()
    resp_data = resp.json()
    if "body" in resp_data:
        body = json.loads(resp_data["body"]) if isinstance(resp_data["body"], str) else resp_data["body"]
    else:
        body = resp_data

    st.write("🧩 Query :", body.get("sql_query"))
    st.write("📊 Row Count:", body.get("meta", {}).get("row_count"))
    st.write("🪄 Query ID:", body.get("meta", {}).get("query_id"))
    st.write("📈 Rows:")
    st.dataframe(body.get("rows"))
    
    return body

def fix_sql(sql: str) -> str:
    
    """
    Fix SQL query by replacing standalone "store" with "store_id" and adjusting date comparisons.
    Args:
        sql (str): The original SQL query.
    Returns:
        str: The fixed SQL query.
    # Use regex to ensure we only replace standalone "store"
    """
    # 1) Replace standalone "store" → "store_id"
    sql = re.sub(r"\bstore\b", "store_id", sql, flags=re.IGNORECASE)

    # 2) Replace comparisons like:
    #    date >= DATE('2013-03-25')
    #    date <= DATE('2013-03-31')
    sql = re.sub(
        r"(date)\s*([<>]=?|=)\s*DATE\('(\d{4}-\d{2}-\d{2})'\)",
        r"CAST(\1 AS DATE) \2 DATE '\3'",
        sql,
        flags=re.IGNORECASE,
    )

    # 3) Replace comparisons like:
    #    date >= '2013-03-25'
    #    date <= '2013-03-31'
    sql = re.sub(
        r"(date)\s*([<>]=?|=)\s*'(\d{4}-\d{2}-\d{2})'",
        r"CAST(\1 AS DATE) \2 DATE '\3'",
        sql,
        flags=re.IGNORECASE,
    )

    # 4) Optionally handle BETWEEN with DATE('...')
    sql = re.sub(
        r"(date)\s+BETWEEN\s+DATE\('(\d{4}-\d{2}-\d{2})'\)\s+AND\s+DATE\('(\d{4}-\d{2}-\d{2})'\)",
        r"CAST(\1 AS DATE) BETWEEN DATE '\2' AND DATE '\3'",
        sql,
        flags=re.IGNORECASE,
    )

    # 5) Optionally handle BETWEEN with plain strings
    sql = re.sub(
        r"(date)\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'",
        r"CAST(\1 AS DATE) BETWEEN DATE '\2' AND DATE '\3'",
        sql,
        flags=re.IGNORECASE,

    )

    # NEW: Fix DATE_TRUNC('week', date)
    sql = re.sub(
        r"date_trunc\(\s*'(\w+)'\s*,\s*date\s*\)",
        r"date_trunc('\1', CAST(date AS DATE))",
        sql,
        flags=re.IGNORECASE
    )

    return sql

def generate_sql_from_question(question: str) -> dict:
    """
    Generate SQL query from a natural language question using Talk2Data API.
    Args:
        question (str): The natural language question.
    Returns:
        dict: Parsed JSON response containing the SQL query.
    """ 

    payload = {
        "question": question,
        "max_retries": 3,
        "confidence_threshold": 0.7,
    }

    resp = requests.post(
        TALK2DATA_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    raw_sql = data.get("sql_query", "")

    # Replace standalone "store" with "store_id"
    # \b ensures we don't touch "store_id", "store_type", etc.
    fixed_sql = fix_sql(raw_sql)

    # Put the fixed SQL back into the response dict
    data["sql_query"] = fixed_sql
    return data


# ---------- Main Streamlit App ----------
try:

    # ---------- Streamlit UI ----------
    st.title("Rossmann Forecast QA")
    # user_q = st.text_input("Ask a question (e.g., 'Which stores had the biggest forecast errors last week?', 'Give me the weekly forecast vs actual sales for store 105 during May 2014 ?', 'Which stores had the biggest forecast errors last week of month 3 year 2013 ?')")
    user_q = st.text_input(
        "Ask a question:",
        placeholder="e.g.,'Which stores had the biggest forecast errors last week?' or 'Weekly sales for store 105 in May 2014'",
        key="user_q",
    )

    # Get user question

    if st.button("Ask") and user_q.strip():
        
        try:
            # ---------- Generate SQL via Talk2Data API ----------
            st.info("Generating SQL query...")
            api_result = generate_sql_from_question(user_q)
            sql_query = api_result.get("sql_query")

        except requests.HTTPError as e:
            st.error(f"❌ API HTTP error: {e} - {getattr(e.response, 'text', '')}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

        # ---------- Call Athena via API Gateway ----------
        with st.spinner("Running Athena query..."):
            try:
                api_resp = call_sql_api(sql_query)
                rows = api_resp.get("rows", [])
            except Exception as e:
                st.error(f"❌ {e}\n\nPlease try again or change your query")
                st.stop()


        # ---------- Human-readable summary ----------
        # st.write(rows)
        if rows:
            with st.spinner("Summarizing results..."):
                try:
                    summary = summarize_results(user_q, sql_query, rows)
                    st.markdown("**Answer:**")
                    st.write(summary)
                except Exception as e:
                    st.error("❌ Failed to summarize results.")
                    
        else:
            st.info("No results found for this query.")
        
        logger.info("✅ Processing completed successfully")
        
    else:
        st.stop()

except KeyboardInterrupt:
    logger.info("Processing aborted by user")
    st.error("Processing aborted by user")
    st.stop()

except Exception as e:
    logger.exception("❌ Processing failed")
    st.error(f"\n❌ Failed to process question: {e}")
    st.stop()


        
    
    
    