import logging
from pathlib import Path

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pretty_print(documents):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + document.page_content
             for i, document in enumerate(documents)]
        )
    )


pdf = Path("data/basak.pdf")
loader = PyPDFLoader(str(pdf))
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = splitter.split_documents(data)

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embed,
)

retriever = vectorstore.as_retriever()

llm = ChatOllama(
    model="llama3.1:latest",
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

logging.basicConfig()
logger = logging.getLogger("langchain.retrievers.multi_query")
logger.setLevel(logging.INFO)

question = "바삭국의 정치 상황에 대해 말해줘."
unique_docs = retriever_from_llm.invoke(question)

print(len(unique_docs))
pretty_print(unique_docs)


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text:str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) # Remove empty lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your goal
    is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Your generated questions must be in Korean.
    Original question: {question}""",
)

llm_chain = QUERY_PROMPT | llm | output_parser

retriever = MultiQueryRetriever(
    retriever=retriever,
    llm_chain=llm_chain,
    parser_key="lines",
)

question = "바삭국의 정치 상황에 대해 말해줘."
unique_docs = retriever.invoke(question)

print(len(unique_docs))
pretty_print(unique_docs)