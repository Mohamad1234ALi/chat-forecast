import streamlit as st
import requests, os, json
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from datetime import date, timedelta
import openai
from openai import OpenAI

# ---------- Config ----------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
API_GATEWAY_URL = st.secrets["API_GATEWAY_URL"]


client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- Pydantic ----------
class BaseParams(BaseModel):
    start_date: str
    end_date: str
    limit: Optional[int] = Field(5, ge=1, le=100)

class BiggestForecastErrorsParams(BaseParams):
    pass

class StoreWeekSummaryParams(BaseParams):
    store_id: int

class MappingOutput(BaseModel):
    query_key: str
    params: dict

# ---------- Helper functions ----------
MIN_DATE = date(2013,1,1)
MAX_DATE = date(2015,12,31)

def latest_week_in_dataset():
    end = MAX_DATE
    start = end - timedelta(days=6)
    return start.isoformat(), end.isoformat()

def map_user_to_query(user_text: str) -> MappingOutput:
    prompt = f"""
You are an assistant that MUST map user questions about Rossmann store sales to one of the safe queries:
1) biggest_forecast_errors -> params: start_date, end_date, limit
2) store_week_summary -> params: store_id, start_date, end_date, limit
Return ONLY JSON like {{"query_key":"...","params":{{...}}}}.
If dates are relative (like "last week") map them to **2013–2015 only**.
If the question cannot be mapped to a query, return:
{{"query_key":"unsupported","params":{{}}}}
User question: "{user_text}"
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except:
        import re
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            # Return unsupported if JSON not found
            parsed = {"query_key": "unsupported", "params": {}}
        else:
            parsed = json.loads(m.group(0))
    
    # Ensure both keys exist
    if "query_key" not in parsed:
        parsed["query_key"] = "unsupported"
    if "params" not in parsed:
        parsed["params"] = {}

    # Fill defaults if missing
    if parsed["query_key"] != "unsupported":
        if "start_date" not in parsed["params"]:
            parsed["params"]["start_date"], parsed["params"]["end_date"] = latest_week_in_dataset()
        if "limit" not in parsed["params"]:
            parsed["params"]["limit"] = 5

    return MappingOutput(**parsed)
    
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

# ---------- Streamlit UI ----------
st.title("Rossmann Forecast QA")
user_q = st.text_input("Ask a question (e.g., 'Which stores had the biggest forecast errors last week?', 'Give me the weekly forecast vs actual sales for store 105 during May 2014 ?', 'Which stores had the biggest forecast errors last week of month 3 year 2013 ?')")

if st.button("Ask") and user_q.strip():
    with st.spinner("Mapping question..."):
        try:
            mapping = map_user_to_query(user_q)
        except Exception as e:
            st.error(f"Mapping failed: {e}")
            st.stop()

    if mapping.query_key == "unsupported":
        st.warning("Question not recognized or supported. Try asking about store sales or forecast errors.")
        st.stop()

    # Validate params client-side
    try:
        if mapping.query_key == "biggest_forecast_errors":
            BiggestForecastErrorsParams(**mapping.params)
        elif mapping.query_key == "store_week_summary":
            StoreWeekSummaryParams(**mapping.params)
    except ValidationError as e:
        st.error(f"Param validation failed: {e}")
        st.stop()

    # st.write("Executing query:", mapping.query_key)
    with st.spinner("Running Athena query..."):
        try:
            api_resp = call_query_api(mapping.query_key, mapping.params)
            rows = api_resp.get("rows", [])
        except Exception as e:
            st.error(f"API call failed: {e}")
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
                st.warning("Failed to summarize results, showing raw table instead.")
                st.table(rows)
    else:
        st.info("No data returned for the given date range.")
