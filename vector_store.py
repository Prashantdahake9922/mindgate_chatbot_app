from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scrape_website import documents as website_docs
from user_upload_handler import load_user_documents, split_documents

def build_vector_store(user_file_paths=None):
    # Step 1: Process scraped website documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for doc in website_docs:
        chunks = text_splitter.create_documents([doc["content"]], [doc["metadata"]])
        all_chunks.extend(chunks)

    # Step 2: If user files are given, process and split them
    if user_file_paths:
        user_docs = load_user_documents(user_file_paths)
        user_chunks = split_documents(user_docs)
        all_chunks.extend(user_chunks)

    # Step 3: Create and save vector store
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("user_file_paths==")
    vector_store = FAISS.from_documents(all_chunks, embedding)
    vector_store.save_local("faiss_index")
    return vector_store









# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# # from scrape_website import documents

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# all_chunks = []
# for doc in documents:
#     chunks = splitter.create_documents([doc["content"]], [doc["metadata"]])
#     all_chunks.extend(chunks)
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_store = FAISS.from_documents(all_chunks, embedding)
# vector_store.save_local("faiss_index")
