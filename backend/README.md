# Study Assistant Backend

This is the backend implementation for a study assistant application that uses RAG (Retrieval-Augmented Generation) with Hugging Face's transformers library, specifically the "meta-llama/Llama-3.2-3B-Instruct" model.

## Features

- PDF ingestion and processing using pdfplumber
- Text chunking with overlapping windows
- Embedding generation using sentence-transformers ('all-MiniLM-L6-v2')
- Vector similarity search using FAISS
- RAG-powered question answering using Llama-3.2-3B-Instruct
- Study schedule generation based on user feedback

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
```
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:
```
python app.py
```

The server will run at `http://localhost:8000`.

## API Endpoints

- `/ingest`: Upload and process PDF files
- `/schedule`: Generate adaptive study plans
- `/query`: RAG-based question answering
- `/feedback`: Record study progress (to be implemented)

## Hardware Requirements

- At least 8GB of RAM (16GB+ recommended)
- GPU with at least 8GB VRAM for optimal performance (CPU-only mode will be significantly slower)

## Note on Model Usage

The Llama-3.2-3B-Instruct model requires acceptance of Meta's license agreement. Make sure you have the appropriate permissions to use this model for your specific use case.
