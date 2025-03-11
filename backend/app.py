# -*- coding: utf-8 -*-
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Study Assistant API",
    description="API for managing and querying study materials",
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

# Configure CORS - simpler configuration
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
vector_index = None
document_chunks = []
document_embeddings = None
model_loading_status = "not_started"  # Possible values: not_started, loading, ready, failed
tokenizer = None
model = None

# Model configuration
EMBEDDING_MODEL_NAME = "basic_embedding"  # Using a simple embedding approach
LLM_MODEL_NAME = "basic_llm"  # Using a simple LLM approach
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize models at startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts."""
    global model_loading_status
    
    # Start loading the embedding model
    try:
        get_embedding_model()
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
    
    # Start loading the LLM model
    if model_loading_status == "not_started":
        model_loading_status = "loading"
        try:
            _load_llm_model()
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
    vector_answer: Optional[str] = None
    vector_chunks: Optional[List[str]] = None

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
        logger.info("Initializing basic word embedding model")
        try:
            import numpy as np
            from collections import defaultdict
            import re
            
            # Create a simple word embedding model
            class BasicWordEmbedding:
                def __init__(self, embedding_dim=384):
                    self.embedding_dim = embedding_dim
                    self.word_vectors = defaultdict(lambda: self._random_vector())
                    self.device = 'cpu'
                
                def _random_vector(self):
                    return np.random.normal(0, 0.1, (self.embedding_dim,))
                
                def _preprocess_text(self, text):
                    # Basic text preprocessing
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
                            # Average word vectors
                            word_vectors = [self.word_vectors[word] for word in words]
                            vec = np.mean(word_vectors, axis=0)
                        # Normalize the vector
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        embeddings.append(vec)
                    
                    return np.array(embeddings)
            
            embedding_model = BasicWordEmbedding()
            logger.info("Successfully initialized basic word embedding model")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
    return embedding_model

def _load_llm_model():
    """Internal function to load the LLM model."""
    global model_loading_status, tokenizer, model
    
    try:
        logger.info("Loading basic LLM model")
        
        # Create a simple LLM model that doesn't require external downloads
        class BasicLLM:
            def __init__(self):
                self.device = 'cpu'
                self.responses = {
                    "default": "I don't have enough information to answer that question.",
                    "greeting": "Hello! How can I help you with your studies today?",
                    "thanks": "You're welcome! Let me know if you need anything else.",
                    "schedule": "Based on your study progress, I recommend focusing on the chapters you find most challenging first.",
                    "explain": "This topic involves several key concepts that are important to understand."
                }
            
            def generate(self, prompt, max_length=100, temperature=0.7):
                # Very basic response generation based on keywords
                prompt = prompt.lower()
                
                if any(word in prompt for word in ["hello", "hi", "hey"]):
                    return self.responses["greeting"]
                elif any(word in prompt for word in ["thank", "thanks"]):
                    return self.responses["thanks"]
                elif any(word in prompt for word in ["schedule", "plan", "time"]):
                    return self.responses["schedule"]
                elif any(word in prompt for word in ["explain", "what is", "how does"]):
                    return self.responses["explain"]
                else:
                    return self.responses["default"]
        
        # Create a simple tokenizer
        class BasicTokenizer:
            def __init__(self):
                pass
                
            def encode(self, text, return_tensors=None):
                # Just split by spaces as a very basic tokenization
                return text.split()
                
            def decode(self, tokens):
                if isinstance(tokens, list):
                    return " ".join(tokens)
                return str(tokens)
        
        # Set the global variables
        tokenizer = BasicTokenizer()
        model = BasicLLM()
        model_loading_status = "ready"
        logger.info("Basic LLM model loaded successfully")
            
    except Exception as e:
        model_loading_status = "failed"
        logger.error(f"Failed to load LLM model: {str(e)}")
    
    return model

def load_llm_model_in_background(background_tasks: BackgroundTasks):
    """Load LLM model in background to avoid blocking the API."""
    global model_loading_status
    
    if model_loading_status == "not_started":
        model_loading_status = "loading"
        background_tasks.add_task(_load_llm_model)
        return {"status": "loading", "message": f"Started loading {LLM_MODEL_NAME} in the background"}
    elif model_loading_status == "loading":
        return {"status": "loading", "message": f"Still loading {LLM_MODEL_NAME}"}
    elif model_loading_status == "ready":
        return {"status": "ready", "message": f"{LLM_MODEL_NAME} is loaded and ready"}
    else:  # failed
        return {"status": "failed", "message": f"Failed to load {LLM_MODEL_NAME}"}

def create_text_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    
    return chunks

def initialize_vector_store():
    """Initialize FAISS vector store."""
    global vector_index, document_embeddings
    
    if len(document_chunks) == 0:
        return
    
    embedding_model = get_embedding_model()
    document_embeddings = embedding_model.encode(document_chunks)
    
    # Initialize FAISS index - using L2 distance
    vector_dimension = document_embeddings.shape[1]
    vector_index = faiss.IndexFlatL2(vector_dimension)
    
    # Add embeddings to the index
    vector_index.add(np.array(document_embeddings).astype('float32'))
    logger.info(f"Vector store initialized with {len(document_chunks)} chunks")

def search_similar_chunks(query, top_k=5):
    """Search for chunks similar to the query."""
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
        
        # Search for similar chunks
        distances, indices = vector_index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Return the similar chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(document_chunks):
                results.append({
                    "chunk": document_chunks[idx],
                    "distance": float(distances[0][i])
                })
        
        return results
    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        return []

def generate_llm_response(query, context_chunks, max_tokens=512, temperature=0.7):
    """Generate a response using the LLM model with RAG context."""
    global model_loading_status, tokenizer, model
    
    if model_loading_status != "ready":
        # If the model isn't loaded, provide a placeholder response
        return f"I'm still preparing my knowledge base. The model '{LLM_MODEL_NAME}' is currently {model_loading_status}. Please try again shortly."
    
    try:
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["chunk"] for chunk in context_chunks])
        
        # Create a prompt for the LLM that includes the context and query
        prompt = f"""<|system|>
You are a helpful AI assistant that answers questions based on the provided context.
Use the following context to answer the user's question. If you don't know the answer based on the context, say so.

Context:
{context}
</s>

Question: {query}

Answer: """

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt)

        # Generate a response
        response = model.generate(prompt, max_length=len(inputs) + max_tokens, temperature=temperature)

        # Convert the response to text
        response = tokenizer.decode(response)

        return response
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        return "Failed to generate response. Please try again later."

def generate_llm_response_with_vector_search(query, max_tokens=512, temperature=0.7, top_k=5):
    """Generate a response using the LLM model with vector search."""
    global model_loading_status, tokenizer, model
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(query, top_k=top_k)
    
    # If no similar chunks found, return a message
    if not similar_chunks:
        return "I couldn't find any relevant information in the uploaded documents. Please try a different query or upload more documents."
    
    # If the model isn't loaded or failed to load, provide a response based on the retrieved chunks
    if model_loading_status != "ready":
        # Create a simple summary from the retrieved chunks
        response = f"Here's what I found about '{query}':\n\n"
        
        for i, chunk in enumerate(similar_chunks):
            # Add a snippet from each chunk
            response += f"Excerpt {i+1}:\n{chunk['chunk'][:300]}...\n\n"
        
        response += f"\nNote: The LLM model '{LLM_MODEL_NAME}' is currently {model_loading_status}, so I'm providing raw excerpts instead of a generated response."
        return response
    
    try:
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["chunk"] for chunk in similar_chunks])
        
        # Create a prompt for the LLM that includes the context and query
        # Format for GPT-2 is simpler than for Llama models
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt)

        # Generate a response
        response = model.generate(prompt, max_length=len(inputs) + max_tokens, temperature=temperature)

        # Convert the response to text and extract only the answer part
        full_response = tokenizer.decode(response)
        
        # Try to extract just the answer part (after "Answer:")
        try:
            answer_part = full_response.split("Answer:")[-1].strip()
            return answer_part
        except:
            # If extraction fails, return the full response
            return full_response
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        
        # Fallback to providing the retrieved chunks directly
        response = f"I found some relevant information about '{query}', but couldn't generate a proper response. Here are the excerpts:\n\n"
        
        for i, chunk in enumerate(similar_chunks):
            response += f"Excerpt {i+1}:\n{chunk['chunk'][:300]}...\n\n"
            
        return response

@app.post("/ingest", tags=["Document Management"])
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        Path("uploads").mkdir(exist_ok=True)
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        # Split into chunks and store
        global document_chunks
        document_chunks = create_text_chunks(text)
        
        # Initialize vector store
        initialize_vector_store()
        
        return {"message": "PDF processed successfully", "chunks": len(document_chunks)}
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        return {"error": str(e)}

@app.post("/schedule", tags=["Schedule"])
async def generate_schedule(request: ScheduleRequest):
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
        chapters_per_day = max(1, len(request.remaining_chapters) / remaining_days)
        
        current_date = datetime.datetime.now()
        chapter_index = 0
        
        while chapter_index < len(request.remaining_chapters):
            day_schedule = {
                "date": current_date.strftime("%Y-%m-%d"),
                "chapters": [],
                "estimated_hours": 0
            }
            
            # Assign chapters for this day
            daily_chapters = min(chapters_per_day, len(request.remaining_chapters) - chapter_index)
            for _ in range(int(daily_chapters)):
                if chapter_index < len(request.remaining_chapters):
                    day_schedule["chapters"].append(request.remaining_chapters[chapter_index])
                    day_schedule["estimated_hours"] += avg_time
                    chapter_index += 1
            
            schedule.append(day_schedule)
            current_date = current_date.replace(day=current_date.day + 1)
        
        return {"schedule": schedule}
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        return {"error": str(e)}

@app.get("/test-cors")
async def test_cors():
    """Simple endpoint to test CORS configuration"""
    return {"message": "CORS test successful", "status": "ok"}

@app.get("/api/query", tags=["Query"])
async def query_simple(query: str = "default query"):
    """
    A simplified version of the query endpoint that returns a direct response.
    This is for testing purposes to troubleshoot CORS issues.
    """
    return {
        "answer": f"You asked: {query}",
        "status": "ok"
    }

@app.get("/query", response_model=QueryResponse, tags=["Query"], summary="Query documents using GET method", operation_id="query_documents_get")
async def query_documents_get(
    query: str = Query(
        ..., 
        title="Query String",
        description="The question to ask about your documents",
        example="What is the main topic?"
    )
):
    """
    Process a query using keyword matching and vector similarity search.
    
    Parameters:
        query (str): The question to ask about your documents
        
    Returns:
        QueryResponse: Contains both keyword-based and vector-based search results
    """
    try:
        # Simple keyword-based search for backward compatibility
        query_terms = re.findall(r'\w+', query.lower())
        
        # Check if documents have been ingested
        if not document_chunks:
            return {
                "answer": "No documents have been ingested yet. Please upload a PDF file using the /ingest endpoint first.",
                "relevant_chunks": [],
                "vector_answer": "No documents have been ingested yet.",
                "vector_chunks": []
            }
        
        # Score chunks based on term frequency
        scored_chunks = []
        for chunk in document_chunks:
            chunk_text = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_text)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Return top 3 most relevant chunks
        relevant_chunks = [chunk for _, chunk in sorted(scored_chunks, reverse=True)[:3]]
        
        # Also try the vector-based approach
        try:
            similar_chunks = search_similar_chunks(query, top_k=3)
            vector_chunks = [chunk["chunk"] for chunk in similar_chunks] if similar_chunks else []
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            similar_chunks = []
            vector_chunks = []
        
        # Generate response using LLM with vector search
        try:
            vector_response = generate_llm_response_with_vector_search(query, max_tokens=512, temperature=0.7, top_k=3)
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            vector_response = "Error generating response. Please check the logs."
        
        # Provide both responses for comparison
        return {
            "answer": "Based on keyword matching:" if relevant_chunks else "No relevant chunks found using keyword matching.",
            "relevant_chunks": relevant_chunks,
            "vector_answer": vector_response,
            "vector_chunks": vector_chunks
        }
    except Exception as e:
        logger.error(f"Error in GET query: {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "relevant_chunks": [],
            "vector_answer": "Error occurred during processing.",
            "vector_chunks": []
        }

@app.post("/query", response_model=QueryResponse, tags=["Query"], operation_id="query_documents_post")
async def query_documents_post(request: QueryRequest):
    """
    Process a query using vector similarity search and LLM generation.
    
    - **query**: The question to ask
    - **max_tokens**: Maximum tokens in the response
    - **temperature**: Temperature for response generation
    - **top_k**: Number of similar chunks to retrieve
    """
    try:
        # Generate response using LLM with vector search
        response = generate_llm_response_with_vector_search(
            request.query, 
            max_tokens=request.max_tokens, 
            temperature=request.temperature, 
            top_k=request.top_k
        )
        
        # Get the similar chunks for reference
        similar_chunks = search_similar_chunks(request.query, top_k=request.top_k)
        
        return {
            "answer": response,
            "relevant_chunks": [],
            "vector_answer": response,
            "vector_chunks": [chunk["chunk"] for chunk in similar_chunks]
        }
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        
@app.post("/feedback", tags=["System"])
async def provide_feedback(request: FeedbackRequest):
    try:
        # Store feedback in a database or file
        # For now, just log it
        logger.info(f"Feedback received: {request.query}, {request.answer}, {request.relevance}, {request.helpfulness}")
        
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return {"error": str(e)}

@app.get("/model-status", tags=["System"])
async def get_model_status(background_tasks: BackgroundTasks):
    """Get the status of the models and initiate loading if not started."""
    global model_loading_status, document_chunks
    
    # Start loading the model if not already loading
    status_info = load_llm_model_in_background(background_tasks)
    
    return ModelStatus(
        embedding_model=EMBEDDING_MODEL_NAME,
        llm_model=LLM_MODEL_NAME,
        status=status_info["status"],
        document_chunks=len(document_chunks),
        message=status_info["message"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
