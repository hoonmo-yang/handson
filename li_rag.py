from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


documents = SimpleDirectoryReader("data").load_data()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda",
    normalize=True,
)

Settings.llm = Ollama(
    model="llama3.1:latest",
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("바삭국의 수도는 어디야?")
print(response)
