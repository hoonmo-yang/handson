from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda


llm = ChatOllama(
    model="llama3.1:latest",
)

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

langchain_chain = PromptTemplate.from_template(
    """You are an expert in langchain. 
Always answer questions starting with "As Harrison Chase told me".
Respond to the following question:

Question: {question}
Answer:"""
) | llm | StrOutputParser()

llama3_chain = PromptTemplate.from_template(
    """You are an expert in llama3. 
Always answer questions starting with "As Dario Amodei told me".
Respond to the following question:

Question: {question}
Answer:"""
) | llm | StrOutputParser()

general_chain = PromptTemplate.from_template(
    """Respond to the following question:

Question: {question}
Answer:"""
) | llm | StrOutputParser()


def route(info: dict[str, str]) -> Runnable:
    if "llama3" in info["topic"].lower():
        return llama3_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain


full_chain = {
    "topic": chain,
    "question": lambda x: x["question"]
} | RunnableLambda(route)

print(full_chain.invoke({"question": "how do I use langchain?"}))