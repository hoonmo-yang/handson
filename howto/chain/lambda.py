from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain


def length_function(text: str) -> int:
    return len(text)


def _multiple_length_function(text1: str, text2: str) -> int:
    return len(text1) * len(text2)


def multiple_length_function(dict_: dict[str, str]) -> int:
    return _multiple_length_function(
        dict_["text1"], dict_["text2"]
    )

llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_template(
    "What is {a} + {b}?"
)

chain1 = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {
            "text1": itemgetter("foo"),
            "text2": itemgetter("bar")
        } | RunnableLambda(multiple_length_function),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(chain1.invoke({"foo": "bar", "bar": "gah"}))

prompt1 = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

prompt2 = ChatPromptTemplate.from_template(
    "What is the subject of this joke: {joke}"
)


@chain
def custom_chain(text: str) -> Document:
    prompt_val1 = prompt1.invoke({"topic": text})
    output1 = llm.invoke(prompt_val1)
    parsed_output1 = StrOutputParser().invoke(output1)
    chain2 = prompt2 | llm | StrOutputParser()
    return chain2.invoke({"joke": parsed_output1})


print(custom_chain.invoke("bears"))