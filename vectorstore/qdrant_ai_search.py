import os
import uuid
from types import SimpleNamespace
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings

# Local embedding model initialization 
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


class AISearchVectorStore:
    """Qdrant Vector Store wrapper matching original interface names to prevent breakage."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str = "investor_intelligence"
    ) -> None:
        # Cloud URL aur API Key ko environment se uthayega
        qdrant_url = os.getenv("QDRANT_URL") or endpoint or "http://localhost:6333"
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or api_key
        
        
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME") or index_name

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks directly to local Qdrant collection.
        """
        # Ensure collection exists with correct dimensions (1024 for BAAI/bge-large-en-v1.5)
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            print(f"Creating new Qdrant collection: {self.collection_name}")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )

        points = []

        for idx, chunk in enumerate(chunks):
            
            text_content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            
        
            vector = embeddings.embed_query(text_content)

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Qdrant accepts valid UUID strings
                    vector=vector,
                    payload={
                        "page_content": text_content,
                        "metadata": {
                            "company": company,
                            "year": int(year) if str(year).isdigit() else year,
                            "source_file": source_file,
                        }
                    }
                )
            )

        # Batch upload documents
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Uploaded {len(points)} chunks successfully to Qdrant collection '{self.collection_name}'.")


class Retriever:
    """Wrapper around Qdrant client for retrieving relevant chunks.
    Mirrors the architecture used in the ingestion and RAG extractor.
    """
    def __init__(self, client, collection_name: str = "investor_intelligence"):
        self.client = client
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME") or collection_name

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Retrieve relevant chunks from local Qdrant Vector Store."""
        
        # 1. Convert text query into vector embeddings
        query_vector = embeddings_model.embed_query(query)

        # 2. Build metadata filter expressions
        qdrant_filter = None
        filter_conditions = []

        if company:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.company",
                    match=models.MatchValue(value=company)
                )
            )
        if year:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.year",
                    match=models.MatchValue(value=int(year) if str(year).isdigit() else year)
                )
            )

        if filter_conditions:
            qdrant_filter = models.Filter(must=filter_conditions)

        # 3. Vector search execution
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k
            )
        except Exception as e:
            print(f"[Error] Qdrant search failed: {e}. Returning empty list.")
            return []

        documents = []
        
        # 4. Extract payload text content using SimpleNamespace glue code
        for result in results:
            payload = result.payload or {}
            content = payload.get("page_content") or payload.get("content") or ""
            documents.append(SimpleNamespace(page_content=content))
            
        return documents