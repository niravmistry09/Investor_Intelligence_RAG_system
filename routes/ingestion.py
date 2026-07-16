import os
import shutil
from fastapi import APIRouter, File, UploadFile
from pathlib import Path
# 1. Wapas purana normal import lao
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from ingestion.ingest_documents import ingest_document

router = APIRouter()
_embeddings_model = None

def get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        print("Loading Embedding Model into Local Memory safely...")
        # 2. Wapas local initialization bina kisi API key ke
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
    return _embeddings_model

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    try:
        upload_dir = Path("data/raw_pdfs")
        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = upload_dir / file.filename

        # Save the file locally first
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # 2. Fetch Qdrant Credentials 
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"

        if not qdrant_url:
            qdrant_url = "http://localhost:6333"

        # 3. Initialize Qdrant Client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            check_compatibility=False
        )

        embeddings_model = get_embeddings_model()

        # 4. Process the document using our updated ingestion logic
        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings_model,
            client=client,
            collection_name=collection_name
        )

        return {
            "message": "Document uploaded and ingested successfully!",
            "file_name": file.filename
        }
        
    except Exception as e:
        print(f"Error in ingestion endpoint: {e}")
        
        return {
            "message": f"Ingestion failed gracefully: {str(e)}",
            "file_name": file.filename
        }