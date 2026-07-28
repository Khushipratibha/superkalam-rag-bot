import requests
from bs4 import BeautifulSoup
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

CATEGORY_URLS = [
    "https://newsonair.gov.in/category/national/",
    "https://newsonair.gov.in/category/international/",
    "https://newsonair.gov.in/category/business/",
    "https://newsonair.gov.in/category/sports/",
    "https://newsonair.gov.in/category/miscellaneous/",
]
headers = {"User-Agent": "Mozilla/5.0"}

def get_article_links(base_url):
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "newsonair.gov.in" in href and href not in [base_url, "https://newsonair.gov.in/"]:
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

print("Fetching article links from all categories...")
all_links = set()
for url in CATEGORY_URLS:
    category_links = get_article_links(url)
    print(f"  {url} -> {len(category_links)} links")
    all_links.update(category_links)

links = list(all_links)
print(f"\nTotal unique articles found across all categories: {len(links)}")

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