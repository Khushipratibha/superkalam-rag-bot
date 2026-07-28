import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Current Affairs AI Tutor", page_icon="📰")
st.title("📰 Current Affairs AI Tutor")
st.caption("Powered by today's NewsOnAir articles | Your current affairs help")

@st.cache_resource
def load_data():
    if not os.path.exists("data/articles.json"):
        import subprocess
        st.info("Scraping today's news, please wait...")
        subprocess.run(["python", "setup.py"])
    
    with open("data/articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
    embeddings = np.load("data/embeddings.npy")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return articles, embeddings, embedder

articles, embeddings, embedder = load_data()
st.success(f"Loaded {len(articles)} today's news articles. Ask me anything!")

query = st.text_input("Ask about today's current affairs:", placeholder="e.g. What happened with fuel prices today?")

if query:
    with st.spinner("Finding relevant news..."):
        query_embedding = embedder.encode([query])
        scores = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        context = "\n\n".join([articles[i]["text"] for i in top_indices])

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful current affairs tutor for UPSC aspirants. Answer clearly based on the context provided. If the answer is not in the context, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        st.write(response.choices[0].message.content)

        with st.expander("Sources"):
            for i in top_indices:
                st.write(articles[i]["url"])