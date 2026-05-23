import requests
from bs4 import BeautifulSoup

url = "https://newsonair.gov.in/fuel-prices-rise-by-up-to-91-paise-per-litre/"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Try to find article content
article = soup.find("div", class_="entry-content")
if article:
    print(article.get_text(strip=True))
else:
    print("entry-content not found, printing all text:")
    print(soup.get_text(strip=True)[:2000])