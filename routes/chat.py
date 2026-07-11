import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi  # Asli Keyword Search Library

# Core imports
from rag.kpi_extractor_rag import Retriever
from llm.groq_llm import get_groq_client

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Initialize Qdrant Client
        qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            check_compatibility=False
        )
        
        # 2. Setup retriever
        retriever = Retriever(client=client, collection_name=collection_name)

        # 3. Step A: Semantic Search (Vector Match)
        # Groq limit safe rakhne ke liye hum locally 10 chunks fetch karenge
        if request.company and request.year:
            semantic_docs = retriever.invoke(
                query=request.question,
                company=request.company,
                year=request.year,
                top_k=10
            )
        else:
            semantic_docs = retriever.invoke(
                query=request.question,
                top_k=10
            )
            
        # 4. Step B: Local BM25 Keyword Search & Strict De-duplication
        final_docs = []
        
        if semantic_docs:
            corpus = [doc.page_content for doc in semantic_docs]
            tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
            
            # BM25 Algorithm
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = request.question.lower().split(" ")
            
            # Filter top 4 keyword matches
            bm25_top_docs = bm25.get_top_n(tokenized_query, semantic_docs, n=4)
            
            # 🔥 FIX: Unhashable type error se bachne ke liye text-content unique check lagaya
            seen_content = set()
            
            # Pehle BM25 keyword matching chunks ko priority do
            for doc in bm25_top_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    final_docs.append(doc)
            
            # Fir top 2 pure semantic chunks ko add karo (agar unique hain)
            for doc in semantic_docs[:2]:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    final_docs.append(doc)
                    
        context = "\n\n".join(doc.page_content for doc in final_docs)

        # 5. Build chat prompt
        prompt = (
            "You are an expert financial analyst. Use the following context from corporate reports to answer the user's question. "
            "Pay close attention to exact numbers, financial tables, and revenue statements within the context.\n"
            "If the context does not contain explicit details to answer the question accurately, politely state that you do not have enough data.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {request.question}\n\n"
            "Answer:"
        )

        # 6. Connect to Groq Cloud API
        groq_client = get_groq_client()
        chat_model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
        
        response = groq_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # Financial accuracy ke liye low temperature
        )
        
        answer = response.choices[0].message.content
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))