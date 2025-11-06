import os
import json
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


with open("data/queries.json") as f:
    queries = json.load(f)
    
texts = [q['description'] for q in queries.values()]
keys = list(queries.keys())

embeddings = []

for text in texts:
    resp = client.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )
    embeddings.append(resp.data[0].embedding)
    
out = { k:v for k,v in zip(keys, embeddings)}

with open("data/embeddings.json", "w") as f:
    json.dump(out, f)