import streamlit as st
import requests

LAMBDA_API_URL = "https://03w3ig64hl.execute-api.us-east-1.amazonaws.com/default/chat-forecast-insight"

st.title("📊 Forecast Insight Chatbot")

query = st.text_input("Ask about sales forecast performance:")
if st.button("Ask"):
    response = requests.post(LAMBDA_API_URL, json={"query": query})
    answer = response.json().get("response", "No response.")
    st.write(answer)
