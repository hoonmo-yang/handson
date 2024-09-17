from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_huggingface import HuggingFaceEmbeddings

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my fovorite songs.",
    "The Celtics are my favorite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points.",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larray Bird was an iconic NBA player.",
]

retriever = Chroma.from_texts(texts, embedding=embed).as_retriever(
    search_kwargs={"k": 10}
)

query = "What can you tell me about the Celtics?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"{i + 1}: PAGE_CONTENT:{doc.page_content} METADATA:{doc.metadata}")

reordering = LongContextReorder()
docs = reordering.transform_documents(docs)

for i, doc in enumerate(docs):
    print(f"{i + 1}: PAGE_CONTENT:{doc.page_content} METADATA:{doc.metadata}")

llm = ChatOllama(
    model="llama3.1:latest",
)

prompt_template = """
Given these texts:
----
{context}
----
Pleas answer the following question:
{query}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"]
)

chain = create_stuff_documents_chain(llm, prompt)
answer = chain.invoke({"context": docs, "query": query})
print(answer)
