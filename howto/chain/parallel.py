from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf = Path("data/basak.pdf")

loader = PyPDFLoader(str(pdf))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = splitter.split_documents(documents)

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embed,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
)

template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}

    You must use Korean.
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(
    model="llama3.1:latest",
)

retrieval_chain = (
    {"context": retriever,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(retrieval_chain.invoke("바삭국의 경제에 대해 말해줘"))