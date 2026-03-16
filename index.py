import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()


print("Loading PDF...")

loader = PyPDFLoader("dsa.pdf")
documents = loader.load()

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print("Creating embeddings...")

embedding_model = SentenceTransformersEmbeddings()

print("Creating vector DB...")

vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="./vectordb"
)

vectorstore.persist()

print("✅ Vector database created successfully.")
