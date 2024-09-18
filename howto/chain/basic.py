from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_template(
    "tell me a joke about {topic}"
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "bears"}))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You must answer in Korean"),
    ("user", "tell me an advice about {topic}"),
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "감기"}))
