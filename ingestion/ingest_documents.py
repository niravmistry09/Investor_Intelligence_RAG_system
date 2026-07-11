import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from rag.kpi_extractor_rag import extract_financial_metrics, Retriever
from database.save_metrics import save_metrics

load_dotenv()

# High-accuracy 1024-dimension model
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from a PDF filename."""
    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        year = parts[0]
        company = parts[-1]
    elif len(parts) >= 2:
        company = parts[0]
        year = parts[1]
    else:
        company = stem
        year = ""

    return company, year


def upload_chunks_to_qdrant(collection_name, points=None, client=None, chunks=None, **kwargs):
    """
    Converts LangChain Documents to Qdrant PointStructs with vector embeddings 
    and uploads them seamlessly, ensuring required payload indexes exist.
    """
    final_chunks = chunks if chunks is not None else points
    if final_chunks is None:
        raise ValueError("No data provided to upload. Both 'points' and 'chunks' are None.")

    # 1. Fetch credentials from environment
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # 2. Build cloud client
    cloud_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        check_compatibility=False
    )

    # 3. Handle collection creation gracefully
    try:
        if not cloud_client.collection_exists(collection_name=collection_name):
            print(f"Creating new Qdrant Cloud collection: {collection_name}")
            cloud_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
    except Exception:
        try:
            cloud_client.get_collection(collection_name=collection_name)
        except Exception:
            cloud_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )

    # 🔥 Fix: Added Index for 'source' to avoid the 400 Bad Request Warning
    try:
        cloud_client.create_payload_index(
            collection_name=collection_name,
            field_name="company",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        cloud_client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        cloud_client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as index_err:
        pass  # Silent pass if indexes already exist

    # 4. Convert LangChain Documents into proper Qdrant points with vector embeddings
    qdrant_points = []
    print(f"Generating vectors for {len(final_chunks)} chunks...")
    
    for chunk in final_chunks:
        text_content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        
        if 'company' in kwargs: metadata['company'] = kwargs['company']
        if 'year' in kwargs: metadata['year'] = int(kwargs['year']) if str(kwargs['year']).isdigit() else kwargs['year']
        if 'source_file' in kwargs: metadata['source'] = kwargs['source_file']

        # Generate the vector embedding using the embedding model
        vector = embeddings_model.embed_query(text_content)

        point_id = str(uuid.uuid4())
        qdrant_points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": text_content,
                    **metadata
                }
            )
        )

    # 5. Upload data points
    cloud_client.upload_points(
        collection_name=collection_name,
        points=qdrant_points
    )
    print(f"Uploaded chunks successfully to Qdrant Cloud collection '{collection_name}'.")


def check_if_already_ingested(client: QdrantClient, collection_name: str, source_filename: str) -> bool:
    """
    🔥 Smart Check: Verifies if vectors for this EXACT file already exist in Qdrant.
    """
    try:
        if not client.collection_exists(collection_name=collection_name):
            return False
            
        # Ab hum company/year nahi, direct EXACT filename check kar rahe hain
        filter_conditions = [
            models.FieldCondition(key="source", match=models.MatchValue(value=source_filename))
        ]

        count_result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(must=filter_conditions)
        )
        return count_result.count > 0
    except Exception as e:
        print(f"Warning during duplicate check: {e}")
        return False


def ingest_document(
    pdf_path: str,
    embeddings,
    client: QdrantClient,
    collection_name: str
) -> None:
    """Ingest a single PDF document into local free setup."""
    pdf_file = Path(pdf_path)
    company, year = parse_company_year(pdf_file)
    
    # 🛑 SMART FILTER: Exact Filename Check
    if check_if_already_ingested(client, collection_name, pdf_file.name):
        print(f"⏩ SKIPPED: {pdf_file.name} (This exact file is already ingested in Qdrant!)")
        return

    print(f"🚀 INGESTING: {pdf_file.name} as company={company!r}, year={year!r}")

    converter = PDFToMarkdownConverter()
    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir="data/markdown"
    )

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )
    print(f"Generated {len(chunks)} chunks for {pdf_file.name}")

    # Upload via updated handler
    upload_chunks_to_qdrant(
        client=client,
        collection_name=collection_name,
        chunks=chunks,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name
    )

    # Initialize Retriever
    retriever = Retriever(
        client=QdrantClient(
            url=os.getenv("QDRANT_URL") or "http://localhost:6333",
            api_key=os.getenv("QDRANT_API_KEY"),
            check_compatibility=False
        ),
        collection_name=collection_name
    )

    # Extract metrics via Groq
    metrics = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=int(year) if str(year).isdigit() else None
    )

    # Persist metrics to PostgreSQL
    if metrics:
        save_metrics(company=company, year=int(year) if str(year).isdigit() else None, metrics=metrics)


def ingest_directory(input_dir: str) -> None:
    """Ingest all PDFs from a directory."""
    embeddings = embeddings_model
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            client=client,
            collection_name=collection_name
        )


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")