# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import numpy as np
import logging
import time
import re
import faiss
import torch
import json
import datetime
from pathlib import Path
import os
import fitz  # PyMuPDF for better PDF handling
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Study Assistant API",
    description="API for managing and querying study materials with RAG capabilities",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Query",
            "description": "Query operations for retrieving information from study materials"
        },
        {
            "name": "Document Management",
            "description": "Operations for managing study documents (PDF upload, etc.)"
        },
        {
            "name": "Schedule",
            "description": "Operations for managing study schedules"
        },
        {
            "name": "System",
            "description": "System management operations"
        }
    ]
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS headers directly to responses
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization"
    return response

# Global variables for models and vector storage
embedding_model = None
llm_model = None
tokenizer = None
vector_index = None
document_chunks = []
document_embeddings = None
model_loading_status = "not_started"  # Possible values: not_started, loading, ready, failed

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence-transformers model
LLM_MODEL_NAME = "microsoft/phi-2"  # Small but capable model for local use
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_PDF_PATH = "Module_1.pdf"  # Default PDF to use
PDF_CACHE_DIR = "uploaded_pdfs"
EMBEDDING_CACHE_DIR = "model_cache/embeddings"

# Create necessary directories
Path(PDF_CACHE_DIR).mkdir(exist_ok=True)
Path(EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Initialize models at startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts."""
    global model_loading_status
    
    # Start loading the embedding model
    try:
        logger.info("Loading embedding model...")
        get_embedding_model()
        logger.info("Embedding model loaded successfully")
        
        # Process default PDF if it exists
        default_pdf_path = os.path.join(PDF_CACHE_DIR, DEFAULT_PDF_PATH)
        module_pdf_path = DEFAULT_PDF_PATH  # Check if it's in the root directory
        
        if os.path.exists(default_pdf_path):
            logger.info(f"Processing default PDF: {default_pdf_path}")
            process_pdf(default_pdf_path)
        elif os.path.exists(module_pdf_path):
            logger.info(f"Processing default PDF from root: {module_pdf_path}")
            process_pdf(module_pdf_path)
        else:
            logger.warning(f"Default PDF not found at {default_pdf_path} or {module_pdf_path}")
            
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
    
    # Start loading the LLM model in the background
    if model_loading_status == "not_started":
        model_loading_status = "loading"
        try:
            load_llm_model()
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {str(e)}")
            model_loading_status = "failed"

class StudyProgress(BaseModel):
    chapter: str
    time_taken: float  # in hours
    completion_percentage: float
    difficulty_rating: Optional[int] = None

class ScheduleRequest(BaseModel):
    test_date: str
    completed_chapters: List[StudyProgress]
    remaining_chapters: List[str]

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    sources: Optional[List[Dict[str, Any]]] = None

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    relevance: int  # 1-5, where 1 is "not relevant at all" and 5 is "very relevant"
    helpfulness: int  # 1-5, where 1 is "not helpful at all" and 5 is "very helpful"

class ModelStatus(BaseModel):
    embedding_model: str
    llm_model: str
    status: str
    document_chunks: int
    message: str

def get_embedding_model():
    """Load or retrieve the embedding model."""
    global embedding_model
    if embedding_model is None:
        try:
            # Use sentence-transformers for better embeddings
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Successfully loaded {EMBEDDING_MODEL_NAME} embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            
            # Fallback to a simple embedding approach if model loading fails
            from collections import defaultdict
            
            class BasicWordEmbedding:
                def __init__(self, embedding_dim=384):
                    self.embedding_dim = embedding_dim
                    self.word_vectors = defaultdict(lambda: self._random_vector())
                    self.device = 'cpu'
                
                def _random_vector(self):
                    return np.random.normal(0, 0.1, (self.embedding_dim,))
                
                def _preprocess_text(self, text):
                    text = text.lower()
                    text = re.sub(r'[^\w\s]', '', text)
                    return text.split()
                
                def encode(self, sentences, batch_size=32):
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    
                    embeddings = []
                    for sentence in sentences:
                        words = self._preprocess_text(sentence)
                        if not words:
                            vec = np.zeros(self.embedding_dim)
                        else:
                            word_vectors = [self.word_vectors[word] for word in words]
                            vec = np.mean(word_vectors, axis=0)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        embeddings.append(vec)
                    
                    return np.array(embeddings)
            
            embedding_model = BasicWordEmbedding()
            logger.warning("Using fallback basic word embedding model")
            
    return embedding_model

def load_llm_model():
    """Load the LLM model."""
    global model_loading_status, tokenizer, llm_model
    
    try:
        logger.info(f"Loading {LLM_MODEL_NAME} model...")
        
        # Try to load the HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        # For phi-2, we can load with 8-bit quantization for lower memory usage
        if torch.cuda.is_available():
            logger.info("Using GPU for model loading")
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        else:
            logger.info("Using CPU for model loading")
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map="auto",
                load_in_8bit=True
            )
            
        model_loading_status = "ready"
        logger.info(f"{LLM_MODEL_NAME} model loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load LLM model: {str(e)}")
        
        # Fallback to a simple LLM
        class BasicLLM:
            def __init__(self):
                self.device = 'cpu'
                self.generate_text = self._generate_text
                
            def _generate_text(self, prompt, max_length=100, temperature=0.7):
                # Extract query from prompt
                query = prompt.split("Question:")[-1].split("Answer:")[0].strip()
                context = prompt.split("Context:")[1].split("Question:")[0].strip()
                
                relevant_segments = [seg for seg in context.split("\n\n") if len(seg) > 50]
                
                if not relevant_segments:
                    return "I don't have enough information to answer that question based on the provided context."
                
                # Simple response based on keyword matching
                keywords = query.lower().split()
                response_segments = []
                
                for segment in relevant_segments:
                    segment_keywords = set(segment.lower().split())
                    if any(kw in segment_keywords for kw in keywords):
                        response_segments.append(segment)
                
                if response_segments:
                    return f"Based on the information provided, I found the following relevant details: {' '.join(response_segments[:3])}"
                else:
                    return "I don't have specific information to answer that question based on the context provided."
                
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        llm_model = BasicLLM()
        model_loading_status = "ready"
        logger.warning("Using fallback basic LLM model")
    
    return llm_model

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text() + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        # Fallback to pdfplumber if fitz fails
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e2:
            logger.error(f"Error extracting text with fallback method: {str(e2)}")
            raise e2

def create_text_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks for better context retention."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to find a natural break point like a paragraph
        if end < text_length:
            # Look for paragraph break
            next_para = text.find("\n\n", end - chunk_overlap, end + 100)
            if next_para != -1:
                end = next_para
            else:
                # Look for sentence break (period followed by space)
                next_sentence = text.find(". ", end - 50, end + 50)
                if next_sentence != -1:
                    end = next_sentence + 1  # Include the period
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position for next chunk, ensuring overlap
        if end >= text_length:
            break
        start = end - chunk_overlap
    
    return chunks

def initialize_vector_store():
    """Initialize FAISS vector store with document chunks."""
    global vector_index, document_embeddings
    
    if len(document_chunks) == 0:
        logger.warning("No document chunks to index")
        return
    
    try:
        logger.info(f"Creating embeddings for {len(document_chunks)} chunks...")
        embedding_model = get_embedding_model()
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i+batch_size]
            batch_embeddings = embedding_model.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        document_embeddings = np.vstack(all_embeddings)
        
        # Initialize FAISS index - using cosine similarity
        vector_dimension = document_embeddings.shape[1]
        vector_index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(document_embeddings)
        
        # Add embeddings to the index
        vector_index.add(document_embeddings)
        logger.info(f"Vector store initialized with {len(document_chunks)} chunks")
        
        # Save embeddings to file for persistence
        embedding_file = os.path.join(EMBEDDING_CACHE_DIR, "document_embeddings.npy")
        chunks_file = os.path.join(EMBEDDING_CACHE_DIR, "document_chunks.json")
        
        np.save(embedding_file, document_embeddings)
        with open(chunks_file, 'w') as f:
            json.dump(document_chunks, f)
            
        logger.info(f"Saved embeddings and chunks to {EMBEDDING_CACHE_DIR}")
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")

def search_similar_chunks(query, top_k=5):
    """Search for document chunks similar to the query."""
    global vector_index, document_chunks, document_embeddings
    
    # Return empty results if no documents have been ingested
    if not document_chunks:
        logger.warning("No documents have been ingested yet")
        return []
    
    # Initialize vector store if needed
    if vector_index is None and document_chunks:
        try:
            initialize_vector_store()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            return []
    
    # Double-check vector index is initialized
    if vector_index is None:
        logger.warning("Vector index is not initialized")
        return []
    
    try:
        # Encode the query
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode([query])
        
        # Normalize the query vector for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        distances, indices = vector_index.search(query_embedding, top_k)
        
        # Return the similar chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(document_chunks):
                results.append({
                    "chunk": document_chunks[idx],
                    "score": float(distances[0][i]),  # Higher is better for inner product
                    "index": int(idx)
                })
        
        # Sort by score in descending order (higher is better)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results
    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        return []

def generate_rag_response(query, context_chunks, max_tokens=512, temperature=0.7):
    """Generate a response using the LLM with context from retrieved chunks."""
    global model_loading_status, tokenizer, llm_model
    
    if model_loading_status != "ready":
        # If the model isn't loaded, provide a placeholder response
        return f"I'm still preparing my knowledge base. Model status is '{model_loading_status}'. Please try again shortly."
    
    try:
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["chunk"] for chunk in context_chunks])
        
        # Create a RAG prompt
        prompt = f"""Answer the following question based ONLY on the context provided below. 
If you can't answer the question based on the context, just say "I don't have enough information to answer this question."
Do not make up information that is not in the context.

Context:
{context}

Question: {query}

Answer:"""

        try:
            # For HuggingFace models
            inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
            
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to extract just the answer part
            try:
                answer_part = response.split("Answer:")[-1].strip()
                return answer_part
            except:
                return response
        
        except AttributeError:
            # For our BasicLLM fallback
            response = llm_model.generate_text(prompt, max_length=max_tokens, temperature=temperature)
            return response
            
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        
        # Fallback response based on the retrieved chunks
        response = "I found some information that might help answer your question:\n\n"
        for i, chunk in enumerate(context_chunks[:3]):
            response += f"Excerpt {i+1}:\n{chunk['chunk'][:200]}...\n\n"
        
        return response

def process_pdf(pdf_path):
    """Process a PDF file and create vector embeddings."""
    global document_chunks
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        document_chunks = create_text_chunks(text)
        logger.info(f"Created {len(document_chunks)} chunks from PDF")
        
        # Initialize vector store
        initialize_vector_store()
        
        return True
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False

@app.post("/ingest", tags=["Document Management"])
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest a PDF document for RAG processing."""
    try:
        # Create uploads directory if it doesn't exist
        Path(PDF_CACHE_DIR).mkdir(exist_ok=True)
        
        # Save the file
        file_path = f"{PDF_CACHE_DIR}/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        success = process_pdf(file_path)
        
        if success:
            return {"message": "PDF processed successfully", "chunks": len(document_chunks)}
        else:
            return {"error": "Failed to process PDF"}, 500
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        return {"error": str(e)}, 500

@app.post("/schedule", tags=["Schedule"])
async def generate_schedule(request: ScheduleRequest):
    """Generate a study schedule based on completed and remaining chapters."""
    try:
        # Parse test date
        test_date = datetime.datetime.strptime(request.test_date, "%Y-%m-%d")
        days_until_test = (test_date - datetime.datetime.now()).days
        
        if days_until_test <= 0:
            return {"error": "Test date must be in the future"}
        
        # Calculate average time per chapter based on completed chapters
        if request.completed_chapters:
            avg_time = sum(p.time_taken for p in request.completed_chapters) / len(request.completed_chapters)
        else:
            avg_time = 2  # Default assumption: 2 hours per chapter
        
        # Generate schedule
        schedule = []
        remaining_days = days_until_test
        chapters_per_day = len(request.remaining_chapters) / remaining_days if remaining_days > 0 else 1
        
        # Adjust if we have very few chapters relative to days
        if chapters_per_day < 0.5 and len(request.remaining_chapters) > 0:
            days_needed = min(len(request.remaining_chapters) * 2, days_until_test)
            days_between_sessions = days_until_test / days_needed
            
            current_date = datetime.datetime.now()
            for i, chapter in enumerate(request.remaining_chapters):
                session_date = current_date + datetime.timedelta(days=i * days_between_sessions)
                schedule.append({
                    "date": session_date.strftime("%Y-%m-%d"),
                    "chapter": chapter,
                    "estimated_hours": avg_time
                })
        else:
            # Distribute chapters evenly
            chapters_remaining = list(request.remaining_chapters)
            for day in range(min(days_until_test, len(chapters_remaining))):
                num_chapters_today = max(1, round(chapters_per_day))
                if not chapters_remaining:
                    break
                    
                chapters_today = chapters_remaining[:num_chapters_today]
                chapters_remaining = chapters_remaining[num_chapters_today:]
                
                schedule.append({
                    "date": (datetime.datetime.now() + datetime.timedelta(days=day)).strftime("%Y-%m-%d"),
                    "chapters": chapters_today,
                    "estimated_hours": avg_time * len(chapters_today)
                })
        
        return {"schedule": schedule, "days_until_test": days_until_test}
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        return {"error": str(e)}

@app.get("/test-cors", tags=["System"])
async def test_cors():
    """Test endpoint for CORS configuration."""
    return {"message": "CORS is working properly"}

@app.get("/api/query", tags=["Query"])
async def query_simple(query: str = "default query"):
    """Simple query endpoint that returns a basic response."""
    return {"answer": f"You asked: {query}. Use the POST /query endpoint for full RAG functionality."}

@app.get("/query", tags=["Query"])
async def query_documents_get(
    query: str = Query(
        ..., 
        title="Query String",
        description="The question to ask about your documents",
        example="What is the main topic of this document?"
    ),
    top_k: int = Query(5, description="Number of relevant chunks to retrieve"),
    max_tokens: int = Query(512, description="Maximum tokens in the generated response"),
    temperature: float = Query(0.7, description="Temperature for response generation")
):
    """Query the documents using RAG and return an answer (GET method)."""
    
    # Check if documents have been ingested
    if not document_chunks:
        # Try to load default document if available
        if os.path.exists(DEFAULT_PDF_PATH):
            success = process_pdf(DEFAULT_PDF_PATH)
            if not success:
                return {"answer": "No documents have been ingested yet. Please upload a PDF document first.", "relevant_chunks": []}
        else:
            return {"answer": "No documents have been ingested yet. Please upload a PDF document first.", "relevant_chunks": []}
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(query, top_k=top_k)
    
    if not similar_chunks:
        return {"answer": "I couldn't find any relevant information in the documents. Try a different question or upload relevant documents.", "relevant_chunks": []}
    
    # Generate a response using RAG
    answer = generate_rag_response(query, similar_chunks, max_tokens=max_tokens, temperature=temperature)
    
    # Return the answer and relevant chunks
    return {
        "answer": answer,
        "relevant_chunks": [chunk["chunk"] for chunk in similar_chunks],
        "sources": [{"chunk_index": chunk["index"], "score": chunk["score"]} for chunk in similar_chunks]
    }

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents_post(request: QueryRequest):
    """Query the documents using RAG and return an answer (POST method)."""
    
    # Check if documents have been ingested
    if not document_chunks:
        # Try to load default document if available
        if os.path.exists(DEFAULT_PDF_PATH):
            success = process_pdf(DEFAULT_PDF_PATH)
            if not success:
                return {"answer": "No documents have been ingested yet. Please upload a PDF document first.", "relevant_chunks": []}
        else:
            return {"answer": "No documents have been ingested yet. Please upload a PDF document first.", "relevant_chunks": []}
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(request.query, top_k=request.top_k)
    
    if not similar_chunks:
        return {"answer": "I couldn't find any relevant information in the documents. Try a different question or upload relevant documents.", "relevant_chunks": []}
    
    # Generate a response using RAG
    answer = generate_rag_response(
        request.query, 
        similar_chunks, 
        max_tokens=request.max_tokens, 
        temperature=request.temperature
    )
    
    # Return the answer and relevant chunks
    return {
        "answer": answer,
        "relevant_chunks": [chunk["chunk"] for chunk in similar_chunks],
        "sources": [{"chunk_index": chunk["index"], "score": chunk["score"]} for chunk in similar_chunks]
    }

@app.post("/feedback", tags=["System"])
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback about a query response."""
    try:
        # Store feedback in a JSON file
        feedback_dir = Path("feedback")
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / "feedback.jsonl"
        
        feedback_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": request.query,
            "answer": request.answer,
            "relevance": request.relevance,
            "helpfulness": request.helpfulness
        }
        
        # Append to JSONL file
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        
        return {"message": "Feedback received successfully"}
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        return {"error": str(e)}

@app.get("/model-status", tags=["System"])
async def get_model_status(background_tasks: BackgroundTasks):
    """Get the status of the models."""
    # If the models are not loaded, start loading them
    if model_loading_status == "not_started":
        load_llm_model_in_background(background_tasks)
    
    # Return the current status
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "status": model_loading_status,
        "document_chunks": len(document_chunks),
        "message": f"Embedding model: {EMBEDDING_MODEL_NAME}, LLM model: {LLM_MODEL_NAME}, Status: {model_loading_status}"
    }

# Try to mount the PDF and cache directories as static files
try:
    app.mount("/pdfs", StaticFiles(directory=PDF_CACHE_DIR), name="pdfs")
except Exception as e:
    logger.warning(f"Could not mount PDF directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
