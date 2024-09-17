import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import (OpenAIEmbeddings, ChatOpenAI)

from langchain.retrievers.document_compressors import (
    LLMListwiseRerank,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from cenai_core import get_value, print_documents


os.environ["OPENAI_API_KEY"] = get_value("OPENAI_API_KEY")

pdf = Path("data/basak.pdf")
loader = PyPDFLoader(str(pdf))
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = splitter.split_documents(data)
embed = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embed,
)

retriever = vectorstore.as_retriever()

question = "바삭국의 통치자는 누구야?"

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

filter_ = LLMListwiseRerank.from_llm(llm, top_n=1)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=filter_,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(question)
print("LLMListRerank")
print_documents(compressed_docs)

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76,
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(question)
print("EmbeddingsFilter")
print_documents(compressed_docs)

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=". "
)

redundant_filter = EmbeddingsRedundantFilter(
    embeddings=embed,
)

relevant_filter = EmbeddingsFilter(
    embeddings=embed,
    similarity_threshold=0.76,
)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(question)
print("DocumentCompressionPipeline")
print_documents(compressed_docs)

