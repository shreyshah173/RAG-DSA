import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist() # convert the user query to embeddings for matching with vectorstore data


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", "APIKEY")

# Groq setup
client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.1-8b-instant"

# Load embeddings
embedding_model = SentenceTransformersEmbeddings()  # object for creating the embeddings

# Load existing vector DB
vectorstore = Chroma(   # load the existing vector db in the vectorstore 
    persist_directory="./vectordb",      # the directory where the vectors exist
    embedding_function=embedding_model   # the model to use for embedding the query
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # This is used to search from the vectorstore and give the top 3 chunks with the similarity

prompt = PromptTemplate(   # this template forces llm to answer from the context and to answer the question
    template="""
Answer the question based only on the context.

Context:
{context}

Question:  
{question}  
""",
    input_variables=["context", "question"]  # variables for the final prompt
)


def rag_query(question):

    docs = retriever.invoke(question) # fetches the top three results from the vectorstore

    context = "\n\n".join([doc.page_content for doc in docs])   # join the top 3 results fetched from the vector store

    final_prompt = prompt.format( # using prompt template the creates the final prompt for the llm
        context=context,
        question=question
    )

    response = client.chat.completions.create(   # calling the groq ai for the response of the final prompt which will be the answer based on the user input
        model=MODEL,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message.content


while True:
    question = input("\nAsk question (type 'exit' to quit): ")

    if question == "exit":
        break

    print("\nAnswer:\n")
    print(rag_query(question))
