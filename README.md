# 📰 Current Affairs AI Tutor — SuperKalam Prototype

A RAG-based chatbot that scrapes today's news from NewsOnAir every day and answers UPSC current affairs questions in real time — built as a prototype of what SuperKalam could ship.

![App Screenshot](screenshot.png)

## How it works

1. `setup.py` scrapes today's articles from NewsOnAir and builds a vector index using sentence-transformers
2. `app.py` runs a Streamlit chatbot — when a user asks a question, it finds the most relevant articles using cosine similarity and passes them as context to Llama 3.1 via Groq API
3. The LLM answers only from today's scraped content, keeping answers grounded and current

## Why this matters for UPSC prep

Current affairs is one of the hardest parts of UPSC preparation — information is scattered, changes daily, and students struggle to retain and query it. This prototype shows how an AI tutor could let aspirants ask natural language questions about the day's news instead of passively reading articles.

## Run locally

```bash
pip install -r requirements.txt
python setup.py        # run once daily to scrape fresh news
streamlit run app.py   # start the chatbot
```

## Tech Stack

- Scraping: requests + BeautifulSoup
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector search: NumPy cosine similarity
- LLM: Llama 3.1 via Groq API
- UI: Streamlit