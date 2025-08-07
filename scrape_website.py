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








# import requests
# from bs4 import BeautifulSoup
# import os
# from urllib.parse import urljoin, urlparse

# def fetch_structured_content(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")

#     for tag in soup(["script", "style", "nav", "footer"]):
#         tag.decompose()

#     # Structured content with headings and paragraphs
#     structured_data = []

#     current_section = {"heading": None, "subheading": None, "paragraphs": []}

#     for elem in soup.find_all(['h1', 'h2', 'h3', 'p']):
#         if elem.name == "h1":
#             # Save previous section if exists
#             if current_section["paragraphs"]:
#                 structured_data.append(current_section)
#                 current_section = {"heading": None, "subheading": None, "paragraphs": []}
#             current_section["heading"] = elem.get_text(strip=True)
#         elif elem.name == "h2":
#             current_section["subheading"] = elem.get_text(strip=True)
#         elif elem.name == "h3":
#             current_section["subheading"] = elem.get_text(strip=True)
#         elif elem.name == "p":
#             text = elem.get_text(strip=True)
#             if text:
#                 current_section["paragraphs"].append(text)

#     # Add the last section
#     if current_section["paragraphs"]:
#         structured_data.append(current_section)

#     return structured_data

# # List of Mindgate pages
# urls = [
#     "https://www.mindgate.solutions/",
#     "https://www.mindgate.solutions/offerings/retail-payments/",
#     "https://www.mindgate.solutions/offerings/reconciliation/",
#     "https://www.mindgate.solutions/about-us/",
#     "https://www.mindgate.solutions/career/"
# ]

# # Scrape all pages
# all_structured_content = {}
# for url in urls:
#     data = fetch_structured_content(url)
#     all_structured_content[url] = data

# # Preview result
# for url, sections in all_structured_content.items():
#     print(f"\n🔗 URL: {url}")
#     for section in sections:
#         print(f"\n🧩 Heading: {section['heading']}")
#         if section["subheading"]:
#             print(f"   ➤ Subheading: {section['subheading']}")
#         for para in section["paragraphs"]:
#             print(f"   ▪ {para}")
