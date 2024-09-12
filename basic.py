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
        "당신은 항상 한국어로 답변해야 합니다. "
    ),
    (
        "user",
        "{user_input}"
    )
])

chain = prompt | llm | StrOutputParser()

question = "한국의 수도는 어디니?"

answer = chain.invoke(
    {"user_input": question},
)

print(answer)
