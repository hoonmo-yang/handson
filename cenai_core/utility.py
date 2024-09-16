from typing import Sequence

from langchain_core.documents import Document


def print_documents(documents: Sequence[Document]) -> None:
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + document.page_content
             for i, document in enumerate(documents)]
        )
    )
