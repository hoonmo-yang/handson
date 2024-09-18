from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write out the following equation using algebraic symbols then solve it."
    ),
    ("human", "{equation_statement}"),
])

chain = (
    {"equation_statement": RunnablePassthrough()}
    | prompt
    | llm.bind(stop=["To"])
    | StrOutputParser()
)

print(chain.invoke("x raised to the third plus seven equals 12"))