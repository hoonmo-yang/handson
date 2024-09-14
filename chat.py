import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
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
        "반드시 한국어를 사용해야 합니다."
    ),
    MessagesPlaceholder(variable_name="messages1")
])

chain = prompt | llm | StrOutputParser()

app = FastAPI()

add_routes(
    app,
    chain,
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

uvicorn.run(
    app=app,
    host="0.0.0.0",
    port=8000,
)
