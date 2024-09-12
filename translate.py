import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_template(
    "다른 것은 대답하지 말고 다음의 문장을 영어로 번역하세요:\n {input}"
)

chain = prompt | llm | StrOutputParser()

app = FastAPI()

add_routes(
    app,
    chain,
    path="/translate",
)

uvicorn.run(
    app=app,
    host="localhost",
    port=8000,
)
