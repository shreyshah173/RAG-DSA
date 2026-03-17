import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):   # Uses a pretrained embedding model to transform the words into specific embedding understood by model
        self.model = SentenceTransformer(model_name)  # assigns the model as the SentenceTransformer with the model to use.

    def embed_documents(self, texts):  
        return self.model.encode(texts, convert_to_numpy=True).tolist()   #used to embed the documents basically convert them into vectors at vectorstore

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()


print("Loading PDF...")

files = ['dsa.pdf']

# loader = PyPDFLoader("dsa.pdf")
# documents = loader.load()

all_documents = [ ]

for file in files:
    loader = PyPDFLoader(file)   # it creates a PDF loader object which will read the file in the arguments `file`  
    docs = loader.load()      # now the load() function will load the data from the loader and save it in docs  
    all_documents.extend(docs)  # and the docs will be saved to common array all_documents

documents = all_documents

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(   # splits the complete data into chunks for saving in the model
    chunk_size=500, # chunk_size, size on one individual chunk for the data
    chunk_overlap=50 # size to overlap the current chunk with the previous chunk
)

docs = splitter.split_documents(documents)  # using the splitter object the documents in the argument will be splitted into chunks.

print("Creating embeddings...")

embedding_model = SentenceTransformersEmbeddings()

print("Creating vector DB...")

vectorstore = Chroma.from_documents(   # create vectors from the docs using embedding_model and saving them in persist_directory
    docs,    # contains the vector of the documents which we need to use for data
    embedding_model,  # sentence transformer contains model for creating embedding from the data
    persist_directory="./vectordb" # location for saving the vector embeddings
)

vectorstore.persist() # the persist function saves the vectorstore in local disk from memory earlier it was in the memeory 

print("✅ Vector database created successfully.")
