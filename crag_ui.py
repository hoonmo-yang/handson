from fastapi import FastAPI
from operator import itemgetter
from pathlib import Path
from pydantic import BaseModel, Field
import uvicorn

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import (
    create_retrieval_chain, create_history_aware_retriever
)

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes


class InputType(BaseModel):
    input: str = Field(
        ...,
        description="user input"
    )


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


print("retriever ...")

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
    search_kwargs={"k": 3},
)

print("retriever complted")

llm = ChatOllama(
    model="llama3.1:latest",
)

contextualize_q_system_prompt = '''
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history.
    Do NOT answer the question, just reformulate it if needed
    and otherwise return it as is.
    You must use Korean.
'''

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = '''
You are an assistant for question-answering tasks.
Use the following pieces of retrived context to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise.
You must use Korean.

{context}
'''

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

chain = (
    RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).with_config(
        configurable={
            "session_id": "abc123",
        }
    )
    | itemgetter("answer")
    | StrOutputParser()
)

app = FastAPI()

add_routes(
    app,
    chain.with_types(input_type=InputType),
    path="/crag",
)

uvicorn.run(
    app=app,
    host="0.0.0.0",
    port=8000,
)
