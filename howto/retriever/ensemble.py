from langchain_core.runnables import ConfigurableField
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever


doc_list_1 = [
    "I like apples",
    "I like oranges",
    "Apples and oringes are fruits",
]

bm25_retriever = BM25Retriever.from_texts(
    doc_list_1,
    metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

doc_list_2 = [
    "You like apples",
    "You like oranages",
]

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_texts(
    doc_list_2,
    embed,
    metadatas=[{"source": 2}] * len(doc_list_2)
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weight=[0.5, 0.5],
)

docs = ensemble_retriever.invoke("apples")
print(docs)

faiss_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2},
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_faiss",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

enemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
)

config = {"configurable": {"search_kwargs_faiss": {"k": 1}}}
docs = ensemble_retriever.invoke("apples", config=config)
print(docs)