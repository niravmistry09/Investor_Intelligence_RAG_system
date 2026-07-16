
# Hybrid-RAG Investor Intelligence Platform

A high-performance, local-cloud hybrid Retrieval-Augmented Generation (RAG) platform designed to automate the extraction, indexing, and analysis of complex financial KPIs from corporate PDF reports (e.g., 10-K, annual reports). 

The system utilizes a custom **HTML/CSS/JavaScript** frontend coupled with a **FastAPI** backend, enforcing strict hallucination controls and executing a dual-retrieval strategy to capture critical financial statements and data tables flawlessly.

---

## 🚀 Key Features

* **Hybrid Retrieval Architecture:** Integrates Dense Vector Semantic Search (`BAAI/bge-large-en-v1.5`) with a local BM25 Keyword Matching algorithm (`rank_bm25`) to ensure financial tables (Revenue, EBITDA, Margins) are never omitted from the context.
* **Dual-Database Framework:** Combines a cloud-based **Qdrant Vector Database** for unstructured text chunk indexing with a structured **PostgreSQL** data layer for persistent storage of historical financial metrics.
* **Deterministic Accuracy Controls:** Calibrated LLM pipeline architecture (`temperature=0.2`) combined with dynamic `top_k` scaling to strictly adhere to official disclosures and eradicate hallucinated data.
* **Smart Ingestion Shield:** Implemented an automated deduplication layer that cross-checks PDF structural hashes prior to chunking, bypassing redundant embedding calculations and optimizing workflow speeds.
* **Automated CI Pipeline:** Integrated **GitHub Actions** for continuous integration, automatically validating Python code formatting/syntax (`flake8`) and checking frontend stability on every push.

---

## 📊 Platform Architecture Matrix (Local vs. Cloud)

| Component | Technology Used | Environment | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3, JavaScript | **Local / Edge** | Interactive, dynamic interface for dashboard metrics and chat system. |
| **Backend Framework** | FastAPI (Python) | **Local Server** | Handles routing, multi-stage data pipelines, and request processing. |
| **Embedding Engine** | `BAAI/bge-large-en-v1.5` | **Local CPU/RAM** | Generates high-fidelity 1024-dimensional semantic text representations locally. |
| **Keyword Search** | BM25 (`rank_bm25`) | **Local Memory** | Performs line-by-line exact keyword boosting on incoming text candidates. |
| **Vector Store** | Qdrant Client | **Cloud Cluster** | Manages remote persistent index pipelines for similarity search matching. |
| **Structured DB** | PostgreSQL | **Cloud / Local** | Tracks structured tables of historical KPI extractions across fiscal years. |
| **LLM Inference** | Groq Cloud API (`Llama-3.3-70b`) | **Cloud Inference** | Delivers ultra-fast financial analysis reasoning over the combined context. |

---

## 📁 Directory Layout

```text
├── .github/workflows/   # CI pipeline configuration (ci.yaml)
├── database/            # PostgreSQL schemas, migrations, and KPI tracking queries
├── ingestion/           # Document parsing pipelines, chunking, and metadata mapping
├── llm/                 # Groq cloud API configurations and inference clients
├── rag/                 # Local BM25 ranking engine and customized retriever components
├── routes/              # FastAPI endpoints (e.g., chat.py handles query processing)
├── static/              # Custom design frontend styles (CSS) and script assets (JS)
├── templates/           # Custom UI pages (HTML templates)
├── .env.example         # System environment configurations blueprint
├── main.py              # Application driver initialization script (FastAPI start)
└── requirements.txt     # Standardized application dependencies list

```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.10+ installed on your workspace.

### 2. Clone and Setup Environment

```bash
git clone <your-repository-url>
cd hybrid-rag-investor-platform

```

Create a virtual environment and update packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Environment Configuration

Create a `.env` file in the root folder using the structure below:

```env
# Database Connections
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_cloud_api_key
QDRANT_COLLECTION_NAME=investor_intelligence

POSTGRES_URL=postgresql://user:password@localhost:5432/dbname

# Cloud Inference Credentials
GROQ_API_KEY=your_groq_cloud_api_key
GROQ_MODEL=llama-3.3-70b-versatile

```

### 4. Running the Application

Launch the backend application driver:

```bash
uvicorn main:app --reload

```

Open your standard web browser and access the interactive application dashboard at `http://127.0.0.1:8000`.

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

The repository includes a GitHub Actions configuration that automatically triggers on every push or pull request to `main` or `master` branches. It performs:

* **Linting & Code Formatting:** Uses `flake8` to scan Python scripts for syntax errors or bad alignment.
* **Frontend Validation:** Verifies that HTML structures and custom CSS bundles load correctly without deployment runtime failures.

---

## 💬 API Endpoints Summary

### `POST /chat`

Processes questions targeting indexed corporate disclosures.

* **Payload:**
```json
{
  "question": "What was the total revenue and EBITDA margin for Apple Inc in 2024?",
  "company": "Apple Inc",
  "year": 2024
}

```


* **Response:** Returns high-fidelity, deterministic answers cross-referenced through the local BM25 + Vector architecture pipeline.

---
