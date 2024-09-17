
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core import print_documents


pdf = Path("data/basak.pdf")
loader = PyPDFLoader(str(pdf))
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = splitter.split_documents(data)

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embed,
)

retriever = vectorstore.as_retriever()

question = "바삭국의 통치자는 누구야?"

docs = retriever.invoke(question)
print("Vanila retriever")
print_documents(docs)

llm = ChatOllama(
    model="llama3.1:latest",
)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(
    question
)
print("LLMChainExtractor")
print_documents(compressed_docs)

filter_ = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=filter_,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(
    question
)
print("LLMChainFilter")
print_documents(compressed_docs)
