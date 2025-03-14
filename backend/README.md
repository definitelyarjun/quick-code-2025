# RAG-Powered Study Assistant API

This is a Retrieval Augmented Generation (RAG) system for study materials. It allows users to upload PDF documents, query them using natural language, and receive intelligent responses based on the document contents.

## Features

- **PDF Document Ingestion**: Upload and process PDF documents for knowledge extraction
- **Retrieval Augmented Generation (RAG)**: Ask questions about your documents and get context-aware answers
- **Vector Search**: Fast and accurate semantic search over document content
- **Study Planning**: Create study schedules based on your progress and remaining chapters

## Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **FastAPI**: Modern, high-performance web framework for building APIs
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for serving the FastAPI application
- **PyMuPDF (fitz)**: PDF processing and text extraction
- **Sentence-Transformers**: State-of-the-art text embeddings for semantic search
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **Transformers**: Hugging Face's library for state-of-the-art NLP models
- **PyTorch**: Deep learning framework powering the language models
- **Accelerate & BitsAndBytes**: Optimization libraries for efficient model inference

### Frontend
- **React 18**: Modern JavaScript library for building user interfaces
- **TailwindCSS**: Utility-first CSS framework for rapid UI development
- **Lucide React**: Beautiful, consistent icon set
- **React Router**: Navigation and routing for React applications
- **date-fns**: Modern JavaScript date utility library
- **clsx & tailwind-merge**: Utilities for conditional class name construction

## Technical Implementation

This implementation uses a modern RAG architecture with the following components:

1. **Document Processing**
   - PDF text extraction using PyMuPDF
   - Intelligent text chunking with overlap for better context retention
   - Document metadata tracking

2. **Embedding & Indexing**
   - Sentence-Transformers for high-quality embeddings
   - FAISS vector database for efficient similarity search
   - Cosine similarity for semantic matching

3. **Generation**
   - Local language model for answer generation (Microsoft Phi-2)
   - Context-aware prompt construction
   - Source attribution for transparency

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js and npm

### Backend Installation

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Installation

```bash
cd frontend
npm install
```

### Running the API

```bash
cd backend
python start.py
```

The API will be available at http://localhost:8000

### Running the Frontend

```bash
cd frontend
npm start
```

The frontend will be available at http://localhost:3000

### API Documentation

Once running, visit http://localhost:8000/docs for the interactive API documentation.

## API Endpoints

### Document Management

- `POST /ingest`: Upload a PDF document for processing

### Query

- `GET /query`: Query your documents (simple GET method)
- `POST /query`: Query your documents with advanced options

### Schedule

- `POST /schedule`: Generate a study schedule based on your progress

### System

- `GET /model-status`: Check the status of the embedding and LLM models
- `POST /feedback`: Provide feedback on query responses

## Using the Default Document

The system will automatically try to use `Module_1.pdf` as the default document. Place this file in the backend directory or in the `uploaded_pdfs` directory.

## Example Usage

### Querying Documents

```python
import requests

# Simple query
response = requests.get(
    "http://localhost:8000/query",
    params={"query": "What are the main topics covered in this document?"}
)
print(response.json())

# Advanced query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Explain the concept of RAG in detail",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_k": 5
    }
)
print(response.json())
```

### Uploading a Document

```python
import requests

files = {'file': open('your_document.pdf', 'rb')}
response = requests.post('http://localhost:8000/ingest', files=files)
print(response.json())
```

## Frontend Integration

To integrate with the frontend:

1. The frontend should call the `/query` endpoint for RAG functionality
2. Use `/ingest` for document uploads
3. Display both the answer and relevant chunks for better user understanding

## Performance Considerations

- The system uses a small but efficient language model that can run on CPU
- For better performance, a CUDA-compatible GPU is recommended
- Embeddings are cached for faster subsequent queries
