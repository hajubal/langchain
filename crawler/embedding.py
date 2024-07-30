import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import KonlpyTextSplitter

import os


def crawl_website(domain):
    visited = set()
    to_visit = [domain]
    pages = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            pages.append((url, text))
            visited.add(url)

            for link in soup.find_all('a'):
                href = link.get('href')

                if href and href.startswith("/service/board/cm_wp/"):
                    full_url = urljoin(url, href)
                    if urlparse(full_url).netloc == urlparse(domain).netloc:
                        to_visit.append(full_url)
                        print(f"Add url: {full_url}")
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

    return pages


def vectorize_and_store(pages):
    text_splitter = KonlpyTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = []

    for url, content in pages:
        chunks = text_splitter.split_text(content)
        texts.extend([(chunk, {"source": url}) for chunk in chunks])

    embeddings = HuggingFaceEmbeddings()

    return FAISS.from_texts([text for text, metadata in texts], embeddings,
                                    metadatas=[metadata for text, metadata in texts])


def save_vector_store(vector_store, path):
    vector_store.save_local(path)


if __name__ == "__main__":
    domain = "https://www.clien.net/service/board/cm_wp"  # 크롤링할 도메인 주소
    vector_store_path = "faiss_index"

    # 1. 웹사이트 크롤링 및 벡터화
    print("Crawling website...")
    pages = crawl_website(domain)

    print("Vectorizing and storing data...")
    vector_store = vectorize_and_store(pages)
    save_vector_store(vector_store, vector_store_path)

    print("Crawling complete.")
