from pathlib import Path

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core import get_value


dalloway = Path("data/mrs_dalloway.txt")
sound_and_fury = Path("data/sound_and_fury.txt")

loaders = [
    TextLoader(str(dalloway)),
    TextLoader(str(sound_and_fury)),
]
documents = []
for loader in loaders:
    documents.extend(loader.load())

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
)

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=embed,
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(documents, ids=None)

sub_docs = vectorstore.similarity_search("Luster")

print(sub_docs[0].page_content)

print("====================================")

retrieved_docs = retriever.invoke("Luster")

print(retrieved_docs[0].page_content)

print("====================================")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embed,
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

print(len(list(store.yield_keys())))

sub_docs = vectorstore.similarity_search("Dalloway")
print(sub_docs[0].page_content)

print("====================================")

retrieved_docs = retriever.invoke("Dalloway")
print(len(retrieved_docs[0].page_content))

print(retrieved_docs[0].page_content)