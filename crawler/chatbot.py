import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
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
                if href:
                    full_url = urljoin(url, href)
                    if urlparse(full_url).netloc == urlparse(domain).netloc:
                        to_visit.append(full_url)
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

    return pages


def vectorize_and_store(pages):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = []
    for url, content in pages:
        chunks = text_splitter.split_text(content)
        texts.extend([(chunk, {"source": url}) for chunk in chunks])

    embeddings = HuggingFaceEmbeddings()

    print(embeddings)

    vector_store = FAISS.from_texts([text for text, metadata in texts], embeddings,
                                    metadatas=[metadata for text, metadata in texts])

    return vector_store


def save_vector_store(vector_store, path):
    vector_store.save_local(path)


def load_vector_store(path):
    embeddings = HuggingFaceEmbeddings()
    # FAISS 벡터 스토어를 로드할 때, 시스템은 기본적으로 pickle 파일의 역직렬화를 허용하지 않습니다.
    # 이는 악의적인 pickle 파일이 실행될 수 있는 위험을 방지하기 위함입니다.
    # 이를 무시하기 위해서는 allow_dangerous_deserialization=True 옵션을 줘야 한다.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def query_vector_store(vector_store, query):
    llm = Ollama(model="llama3:latest")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    result = qa_chain({"query": query})
    return result["result"]


if __name__ == "__main__":
    domain = "https://hajubal.hashnode.dev/"  # 크롤링할 도메인 주소
    vector_store_path = "faiss_index"

    # 1. 웹사이트 크롤링 및 벡터화
    if not os.path.exists(vector_store_path):
        print("Crawling website...")
        pages = crawl_website(domain)
        print("Vectorizing and storing data...")
        vector_store = vectorize_and_store(pages)
        save_vector_store(vector_store, vector_store_path)
    else:
        print("Loading existing vector store...")
        vector_store = load_vector_store(vector_store_path)

    # 2. 벡터 데이터 조회
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        result = query_vector_store(vector_store, query)
        print("Answer:", result)