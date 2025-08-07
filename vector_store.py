from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scrape_website import documents as website_docs
from user_upload_handler import load_user_documents, split_documents

def build_vector_store(user_file_paths=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for doc in website_docs:
        chunks = text_splitter.create_documents([doc["content"]], [doc["metadata"]])
        all_chunks.extend(chunks)

    if user_file_paths:
        user_docs = load_user_documents(user_file_paths)
        user_chunks = split_documents(user_docs)
        all_chunks.extend(user_chunks)

    # embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    print("user_file_paths==")
    vector_store = FAISS.from_documents(all_chunks, embedding)
    vector_store.save_local("faiss_index")
    return vector_store
