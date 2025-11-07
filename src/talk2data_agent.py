import streamlit as st
import requests, os, json
from dotenv import load_dotenv, dotenv_values
import os
import json
import numpy as np
import boto3
from openai import OpenAI
import logging
import pathlib
from sklearn.metrics.pairwise import cosine_similarity
from llm_query_selector import (
    load_prompt,
    build_prompt,
    validate_and_retry
)

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
    

def get_embeddings() -> dict:
    """
    Get or load cached embeddings from S3 or local file.
    
    Returns:
        dict: Dictionary mapping query keys to embedding vectors.
        
    Raises:
        RuntimeError: If embeddings cannot be loaded from any source.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = load_embeddings()
        logger.info(f"Cached {len(_embeddings_cache)} embeddings")
    return _embeddings_cache

# load Embeddings from s3 or local fallback
def load_embeddings() -> dict:
    """
    Load embeddings from S3 or fallback to local file.
    
    Returns:
        dict: Dictionary mapping query keys to embedding vectors.
        
    Raises:
        RuntimeError: If embeddings cannot be loaded from any source.
    """
    # Try S3 first
    try:
        s3 = get_s3_client()
        bucket = os.getenv("S3_BUCKET")
        if not bucket:
            raise ValueError("S3_BUCKET environment variable not set")
            
        response = s3.get_object(
            Bucket=bucket,
            Key=S3_EMBEDDINGS_KEY
        )
        content = response["Body"].read().decode("utf-8")
        embeddings = json.loads(content)
        logger.info(f"✅ Loaded {len(embeddings)} embeddings from S3: {S3_EMBEDDINGS_KEY}")
        return embeddings
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load embeddings from S3: {e}")
        logger.info("🔄 Falling back to local embeddings file...")
        
        # Fallback to local file
        try:
            local_embeddings_path = SCRIPT_DIR / "data" / "embeddings.json"
            with open(local_embeddings_path, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
                logger.info(f"✅ Loaded {len(embeddings)} embeddings from local file: {local_embeddings_path}")
                return embeddings
                
        except FileNotFoundError:
            logger.error(f"❌ Local embeddings file not found: {local_embeddings_path}")
            raise RuntimeError(
                f"Could not load embeddings from S3 or local file.\n"
                f"S3 error: {e}\n"
                f"Local file not found: {local_embeddings_path}"
            ) from e
            
        except Exception as local_error:
            logger.error(f"❌ Failed to load local embeddings: {local_error}")
            raise RuntimeError(
                f"Could not load embeddings from S3 or local file.\n"
                f"S3 error: {e}\n"
                f"Local error: {local_error}"
            ) from local_error

# Embed User-input
def embed_text(text: str) -> list[float]:
    """
    Generate embedding vector for input text using OpenAI API.
    
    Args:
        text (str): Input text to embed.
        
    Returns:
        list[float]: Embedding vector.
        
    Raises:
        ValueError: If text is empty.
        RuntimeError: If OpenAI API call fails.
    """
    if not text or not text.strip():
        raise ValueError("Input text for embedding cannot be empty")
    
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e

# --- Cosine Similarity ---
def find_top_matches(user_text: str, top_k: int = 2) -> list[tuple[str, float]]:
    """
    Find the top-k most similar queries to the user's question.
    
    Args:
        user_text (str): The user's question text.
        top_k (int): Number of top matches to return. Defaults to 2.
        
    Returns:
        list[tuple[str, float]]: List of (query_key, similarity_score) tuples.
        
    Raises:
        ValueError: If top_k is less than 1 or user_text is empty.
    """
    if not user_text or not user_text.strip():
        raise ValueError("user_text cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    
    user_vec = embed_text(user_text)
    db = get_embeddings()

    keys = list(db.keys())
    db_matrix = np.stack([db[k] for k in keys])  # (N x D)
    user_matrix = np.array(user_vec).reshape(1, -1)  # (1 x D)

    sims = cosine_similarity(user_matrix, db_matrix)[0]  

    top_k = min(top_k, len(keys))  # Ensure we don't exceed available keys
    top_indices = sims.argsort()[::-1][:top_k]
    return [(keys[i], sims[i]) for i in top_indices]

def summarize_results(user_question, query_key, rows):
    prompt = f"""
You are a helpful analyst. User asked: "{user_question}"
Query {query_key} returned {len(rows)} rows:
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


def call_query_api(query_key, params):
    # Default limit if missing
    if params.get("limit") is None:
        params["limit"] = 5

    payload = {"query_key": query_key, "params": params}
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

    st.write("🧩 Query Key:", body.get("query_key"))
    st.write("📊 Row Count:", body.get("meta", {}).get("row_count"))
    st.write("🪄 Query ID:", body.get("meta", {}).get("query_id"))
    st.write("📈 Rows:")
    st.dataframe(body.get("rows"))
    
    return body



def main():
    """
    Main execution function for the Talk2Data agent.
    Processes user questions and maps them to database queries.
    """
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
        user_text = user_q.strip()
        if not user_text:
            logger.error("No question provided")
            print("❌ No question provided")
            return 1
        
        if st.button("Ask") and user_text:
        
            # Find top matching queries
            logger.info(f"Processing question: {user_text}")
            top_matches = find_top_matches(user_text)
            print("\n🔍 Top Matches:")
            for k, score in top_matches:
                print(f"  • {k}: {score:.3f}")
            
            # Load queries description
            try:
                with open(QUERIES_PATH, "r", encoding="utf-8") as f:
                    queries_dict = json.load(f)
            except FileNotFoundError:
                logger.error(f"queries.json not found at {QUERIES_PATH}")
                raise RuntimeError(f"Configuration file not found: {QUERIES_PATH}")
            
            # Prepare prompt
            raw_prompt = load_prompt(str(PROMPT_PATH))
            prompt = build_prompt(
                prompt_template=raw_prompt,
                user_text=user_text,
                top_matches=top_matches,
                queries_dict=queries_dict
            )
            
            # LLM Call + validation + retry
            mapping = validate_and_retry(prompt)
            print("\n✅ LLM-Mapping erfolgreich:")
            print(json.dumps(mapping, indent=2, ensure_ascii=False))

            
            # st.write("Executing query:", mapping.query_key)
            with st.spinner("Running Athena query..."):
                try:
                    api_resp = call_query_api(mapping.query_key, mapping.params)
                    rows = api_resp.get("rows", [])
                except Exception as e:
                    st.error(f"❌ {e}\n\nPlease try again or change your query")
                    st.stop()


            # ---------- Human-readable summary ----------
            # st.write(rows)
            if rows:
                with st.spinner("Summarizing results..."):
                    try:
                        summary = summarize_results(user_q, mapping.query_key, rows)
                        st.markdown("**Answer:**")
                        st.write(summary)
                    except Exception as e:
                        st.error("❌ Failed to summarize results.")
                        
            else:
                st.info("No results found for this query.")


            
            logger.info("✅ Processing completed successfully")
            return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Aborted by user")
        logger.info("Processing aborted by user")
        return 130
        
    except Exception as e:
        logger.exception("❌ Processing failed")
        print(f"\n❌ Failed to process question: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
        
        
    
    
    