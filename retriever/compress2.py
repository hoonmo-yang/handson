from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
)