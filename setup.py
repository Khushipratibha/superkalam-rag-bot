import requests
from bs4 import BeautifulSoup
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_URL = "https://newsonair.gov.in/category/national/"
headers = {"User-Agent": "Mozilla/5.0"}

def get_article_links():
    response = requests.get(BASE_URL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "newsonair.gov.in" in href and href not in [BASE_URL, "https://newsonair.gov.in/"]:
            if "/category/" not in href and "/audio" not in href and "/bulletin" not in href:
                links.append(href)
    return list(set(links))

def get_article_text(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        article = soup.find("div", class_="entry-content")
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else ""
        if article:
            return title_text + ". " + article.get_text(strip=True)
    except:
        return None

print("Fetching article links...")
links = get_article_links()
print(f"Found {len(links)} articles")

articles = []
for link in links:
    text = get_article_text(link)
    if text and len(text) > 100:
        articles.append({"url": link, "text": text})
        print(f"Scraped: {link}")

print(f"\nTotal articles scraped: {len(articles)}")

print("Embedding articles...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [a["text"] for a in articles]
embeddings = embedder.encode(texts)

os.makedirs("data", exist_ok=True)
np.save("data/embeddings.npy", embeddings)
with open("data/articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print("Done! Data saved to data/ folder.")