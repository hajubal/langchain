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


def load_vector_store(path):
    embeddings = HuggingFaceEmbeddings()
    # FAISS 벡터 스토어를 로드할 때, 시스템은 기본적으로 pickle 파일의 역직렬화를 허용하지 않습니다.
    # 이는 악의적인 pickle 파일이 실행될 수 있는 위험을 방지하기 위함입니다.
    # 이를 무시하기 위해서는 allow_dangerous_deserialization=True 옵션을 줘야 한다.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def query_vector_store(vector_store, query):
    llm = Ollama(model="llama3.1")

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 2,
                       # "filter": lambda metadata: not set(metadata.get("role", [])).isdisjoint([1, 2])
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    result = qa_chain({"query": query})
    return result["result"]


if __name__ == "__main__":
    domain = "https://www.clien.net/service/board/cm_wp"  # 크롤링할 도메인 주소
    vector_store_path = "faiss_index"

    print("Loading existing vector store...")
    vector_store = load_vector_store(vector_store_path)

    # 2. 벡터 데이터 조회
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        result = query_vector_store(vector_store, query)
        print("Answer:", result)