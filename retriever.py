import asyncio

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma


documents = [
    Document(
        page_content="개는 대단한 반려동물이다. 충성도와 친밀도가 매우 높다.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="고양이는 야행성 동물이다.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="고양이는 개와 다르다.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="개와 고양이",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="고양이와 쥐",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="고양이는 독립적인 동물로 자기만의 공간을 선호한다.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="개는 인간한테 매우 외향적인 친밀도를 비치는 "
                     "반면 고양이는 내향적이기 때문에 사교성이 떨어져 보인다.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="금붕어는 초심자가 선호하는 인기 많은 애완종이다. 돌보는데 별로 손이 많이 안간다.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="마티스 그림 중에 고양이가 어항 속의 금붕어를 잡으려는 장면을 "
                     "그린 그림이 있다.",
        metadata={"source": "matisse-picture-doc"},
    ),
    Document(
        page_content="앵무새는 사람의 대화를 흉내낼만큼 영리한 새이다.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="토끼는 뛰어 돌아다닐 수 있는 넓은 공간이 필요한 사회성이 많은 동물이다.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

embed = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    documents,
    embedding=embed
)

targets = asyncio.run(vectorstore.asimilarity_search("고양이"))

for target in targets:
    print(target.page_content)


targets = asyncio.run(vectorstore.asimilarity_search_with_score("고양이"))

for page, score in targets:
    print(page.page_content, score)


embedding = embed.embed_query("고양이")
print(embedding)

targets = vectorstore.similarity_search_by_vector(embedding)
for target in targets:
    print(target.page_content)

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)

keywords = ["고양이", "앵무새"]
results = retriever.batch(keywords)


print("")

for result, keyword in zip(results, keywords):
    print(f"keyword: {keyword}")

    for page in result:
        print(page.page_content)

    print("")
