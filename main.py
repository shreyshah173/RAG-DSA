import os

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # SentenceTransformers returns a numpy array; convert it to nested lists.
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


# load env here
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY", "APIKEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY must be set before calling the Groq API.")

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

loader = PyPDFLoader("dsa.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embedding_model = SentenceTransformersEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="./vectordb"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

prompt = PromptTemplate(
    template="""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

SYSTEM_MESSAGE = (
    "You are an assistant that only uses the context provided by the user. "
    "If there is not enough context, say that you need more information."
)


def rag_query(question):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": final_prompt},
        ],
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_tokens=512,
    )

    if not response.choices:
        raise RuntimeError("Groq returned an empty response for the question.")

    return response.choices[0].message.content.strip()


question = "What is the content in the document tells about what topic it works on "

answer = rag_query(question)

print(answer)
