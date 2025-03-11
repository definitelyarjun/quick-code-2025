from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import json
import pdfplumber
from pathlib import Path
import os
import re

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for MVP
document_store = []

class StudyProgress(BaseModel):
    chapter: str
    time_taken: float  # in hours
    completion_percentage: float
    difficulty_rating: Optional[int] = None

class ScheduleRequest(BaseModel):
    test_date: str
    completed_chapters: List[StudyProgress]
    remaining_chapters: List[str]

@app.post("/ingest")
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
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        document_store.extend(chunks)
        
        return {"message": "PDF processed successfully", "chunks": len(chunks)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/schedule")
async def generate_schedule(request: ScheduleRequest):
    try:
        # Parse test date
        test_date = datetime.strptime(request.test_date, "%Y-%m-%d")
        days_until_test = (test_date - datetime.now()).days
        
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
        
        current_date = datetime.now()
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
        return {"error": str(e)}

@app.post("/query")
async def query_documents(query: str):
    try:
        # Simple keyword-based search for MVP
        query_terms = re.findall(r'\w+', query.lower())
        
        # Score chunks based on term frequency
        scored_chunks = []
        for chunk in document_store:
            chunk_text = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_text)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Return top 3 most relevant chunks
        relevant_chunks = [chunk for _, chunk in sorted(scored_chunks, reverse=True)[:3]]
        
        return {
            "answer": "Based on keyword matching:",
            "relevant_chunks": relevant_chunks
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
