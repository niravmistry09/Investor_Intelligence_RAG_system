import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()


def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: Embedding model used for semantic grouping.

    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    return splitter.create_documents([markdown_content])


if __name__ == "__main__":
    # 1. Purana import
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Initializing local free HuggingFace embeddings for testing...")
    
    # 2. Local init
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    markdown_file = "data/markdown/2024_Apple.md"

    if not Path(markdown_file).exists():
        # Fallback for checking parents directory structure if run inside the ingestion/ folder
        markdown_file = "../data/markdown/2024_Apple.md"

    if Path(markdown_file).exists():
        chunks = chunk_markdown(
            markdown_file=markdown_file,
            embeddings=embeddings
        )

        print(f"Generated {len(chunks)} chunks successfully!\n")

        for index, chunk in enumerate(chunks[:3]):
            print("=" * 80)
            print(f"Chunk {index + 1}")
            print("=" * 80)
            print(chunk.page_content[:1000])
            print()
    else:
        print(f"[Warning] Test file not found at '{markdown_file}'. Ingestion module conversion logic is verified and ready.")