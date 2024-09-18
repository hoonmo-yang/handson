from operator import itemgetter

from langchain_community.chat_models import ChatOllama

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.runnables.history import RunnableWithMessageHistory


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You're an assistant who speaks in {language}. "
        "Respond in 20 words or fewer."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = (
    prompt
    | llm
    | {"answer": RunnablePassthrough()}
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
) | itemgetter("answer") | StrOutputParser()

print("response")
print(with_message_history.invoke(
    {
        "input": "Hi - I am Hoonmo",
        "language": "English",
    },
    config={"configurable": {"session_id": "1"}}
))

print("In history")
print(store["1"])

print("response")
print(with_message_history.invoke(
    {
        "input": "Do you know my name?",
        "language": "English",
    },
    config={"configurable": {"session_id": "1"}}
))

print("In history")
print(store["1"])