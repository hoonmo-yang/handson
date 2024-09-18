from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_template(
    "tell me a joke about {topic}"
)

chain = prompt | llm | StrOutputParser()

analysis_prompt = ChatPromptTemplate.from_template(
    "Is this a funny joke? {joke}"
)

composed_chain = (
    {"joke": chain}
    | analysis_prompt
    | llm
    | StrOutputParser()
)

answer = composed_chain.invoke({"topic": "bears"})
print(answer)

composed_chain_with_lambda = (
    chain
    | (lambda input: {"joke": input})
    | analysis_prompt
    | llm
    | StrOutputParser()
)

answer = composed_chain.invoke({"topic": "beets"})
print(answer)

composed_chain_with_pipe = (
    RunnableParallel({"joke": chain})
    .pipe(analysis_prompt)
    .pipe(llm)
    .pipe(StrOutputParser())
)

answer = composed_chain.invoke({"topic": "beets"})
print(answer)

composed_chain_with_pipe = (
    RunnableParallel({"joke": chain})
    .pipe(analysis_prompt, llm, StrOutputParser())
)

answer = composed_chain.invoke({"topic": "beets"})
print(answer)