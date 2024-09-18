import os

from langchain.chains import create_history_aware_retriever
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_openai import (OpenAIEmbeddings, ChatOpenAI)

from cenai_core import get_value, print_documents

os.environ["OPENAI_API_KEY"] = get_value("OPENAI_API_KEY")

documents = [
    Document(
        page_content="군포시는 경기도에 있다."
    ),
]


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return documents


llm = ChatOllama(
    model="llama3.1:latest",
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

contextualize_q_system_prompt = """
Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
Just reformulate it if needed and otherwise return it
as is.
You must use Korean.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=FakeRetriever(),
    prompt=prompt,
)

response = history_aware_retriever.invoke(
    {
        "input": "내 이름이 뭐지?",
        "chat_history": [
            ("human", "내 이름은 양훈모야."),
            ("ai", "예 훈모님 만나서 반갑습니다."),
        ]
    },
)

print(response)

response = prompt.invoke(
    {
        "input": "내 이름이 뭐지?",
        "chat_history": [
            ("human", "내 이름은 양훈모야."),
            ("ai", "예 훈모님 만나서 반갑습니다."),
        ]
    },
)

print(response)