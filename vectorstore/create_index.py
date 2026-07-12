import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()


def create_index(
    endpoint: str | None = None,
    api_key: str | None = None,
    index_name: str = "investor_intelligence",
    embedding_dimensions: int = 1024
) -> None:
    
    qdrant_url = endpoint or os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
    
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    print(f"Connecting to Qdrant at {qdrant_url}...")

    # 2. Check if collection already exists, if not, create it
    try:
        client.get_collection(collection_name=index_name)
        print(f"Collection '{index_name}' already exists. Re-creating to apply new configurations...")
    except Exception:
        print(f"Collection '{index_name}' does not exist. Creating new one...")

    # 3. Create or Recreate collection with Cosine distance metric and 1024 dimensions
    client.recreate_collection(
        collection_name=index_name,
        vectors_config=models.VectorParams(
            size=embedding_dimensions, 
            distance=models.Distance.COSINE
        ),
    )

    print(f"Qdrant collection '{index_name}' created successfully with {embedding_dimensions} dimensions.")


if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"

    create_index(
        endpoint=qdrant_url,
        index_name=collection_name,
        embedding_dimensions=1024  
    )