import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자가 입력한 문장을 간단하게 요약해야 합니다. "
        "그외는 아무런 말도 해서는 안됩니다. "
        "반드시 한국어를 사용해야 합니다."
    ),
    (
        "user",
        "'''{input}'''"
    ),
])

chain = prompt | llm | StrOutputParser()

app = FastAPI()

add_routes(
    app,
    chain,
    path="/summary",
)

uvicorn.run(
    app=app,
    host="localhost",
    port=8000,
)
