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
        "당신은 유능하고 친절한 AI 도우미입니다. "
        "사용자의 질문에 항상 충실하게 답변해야 합니다. "
        "오직 한국어로만 대답하세요."
    ),
    (
        "user",
        "{user_input}"
    )
])

chain = prompt | llm | StrOutputParser()

app = FastAPI()

add_routes(
    app,
    chain,
    path="/hi",
)

uvicorn.run(
    app=app,
    host="localhost",
    port=8000,
)
