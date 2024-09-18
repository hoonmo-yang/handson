from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama


pdf = Path("data/basak.pdf")

loader = PyPDFLoader(str(pdf))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
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
    search_kwargs={"k": 3},
)

def concat_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


message = '''
아래에 주어진 RAG를 기반으로 답변을 해야 합니다. 이미 알고 있는 내용과 RAG의 내용을 최대한
서로 반영하되 서로 일치하지 않으면 반드시 RAG의 내용을 채택해야 합니다.
모든 답변은 반드시 한국어로 이루어져야 합니다.

{question}

RAG:
{context}
'''

prompt = ChatPromptTemplate.from_messages([("human", message)])

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = (
    {
        "context": retriever | concat_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("바삭국의 역사를 설명해봐")

print(answer)
