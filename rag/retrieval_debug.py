import os
import sys
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def search_vectorstore(query: str, top: int = 5):
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    client = QdrantClient(url=qdrant_url)
    collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"

    if not collection_name:
        raise RuntimeError(
            "Missing Qdrant collection name. Set QDRANT_COLLECTION_NAME in your .env."
        )

    
    try:
        query_vector = embeddings_model.embed_query(query)
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding vector: {e}")

    # 2. Qdrant standard vector search using query_vector and limit
    results = list(
        client.search(
            collection_name=collection_name, 
            query_vector=query_vector, 
            limit=top
        )
    )

    print(f"Query: {query!r}")
    print(f"Top/Limit: {top}")
    print(f"Results Found: {len(results)}\n")

    for idx, result in enumerate(results, start=1):
        content = None
        
        
        payload = result.payload or {}
        content = payload.get("page_content") or payload.get("content")

        if content is None:
            content = str(result)

        snippet = content.strip().replace("\n", " ") if isinstance(content, str) else "<no content>"
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"   content snippet: {snippet}")
        print("  " + "-" * 60)

    if not results:
        print("No results returned. Verify your Qdrant collection contents or try a different query.")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieval_debug \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()