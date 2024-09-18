import asyncio

from langchain_community.chat_models import ChatOllama

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
)

from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(
    model="llama3.1:latest",
)

prompt = ChatPromptTemplate.from_template(
    "tell me a joke about {topic}"
)
parser = StrOutputParser()
chain = prompt | llm | parser


async def stream():
    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, end="|", flush=True)


asyncio.run(stream())

chain = (
    llm
    | JsonOutputParser()
)


async def stream2():
    async for text in chain.astream(
        "output a list of the contries france, spain and japan "
        "and their populations in JSON format. "
        "Use a dict with an outer key of 'countries' which "
        "contains a list of countries. "
        "Each country should have the key `name` and `population`"
    ):
        print(text, flush=True)


asyncio.run(stream2())


def _extract_country_names(inputs: dict[str, str]) -> list[str]:
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries
        if isinstance(country, dict)
    ]
    return country_names


chain = llm | JsonOutputParser() | _extract_country_names

for text in chain.stream(
    "output a list of the contries france, spain and japan "
    "and their populations in JSON format. "
    "Use a dict with an outer key of 'countries' which "
    "contains a list of countries. "
    "Each country should have the key `name` and `population`"
):
    print(text, end="|", flush=True)