import requests
from bs4 import BeautifulSoup

def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

# Add more pages if needed
urls = [
    "https://www.mindgate.solutions/",
    "https://www.mindgate.solutions/offerings/retail-payments/",
    "https://www.mindgate.solutions/offerings/reconciliation/",
    "https://www.mindgate.solutions/about-us/"
    "https://www.mindgate.solutions/career/"

]

documents = []
for url in urls:
    text = fetch_text_from_url(url)
    documents.append({"content": text, "metadata": {"source": url}})

    print("documents===",documents)


