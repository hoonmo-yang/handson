from langchain_community.chat_models import ChatOllama
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_huggingface import HuggingFaceEmbeddings


physics_template = """You are a very smart physics professor. 
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

llm = ChatOllama(
    model="llama3.1:latest",
)

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

prompt_templates = [physics_template, math_template]
prompt_embeddings = embed.embed_documents(prompt_templates)


def prompt_router(input):
    query_embedding = embed.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

print(chain.invoke("What is blackhole?"))

print(chain.invoke("Solve the equation: x**2 - 2x - 3 = 0"))
