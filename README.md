# Study Planner and Learning Assistant

A personalized, college-specific study planner that uses RAG (Retrieval-Augmented Generation) to help students organize their study schedule and get answers from their study materials.

## Features

- Upload and process study materials (PDFs)
- Track study progress by chapter
- Generate adaptive study schedules based on:
  - Learning pace
  - Content complexity
  - Test deadline
- Query study materials for relevant information
- Visual progress tracking

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install required packages:
```bash
pip install fastapi uvicorn pdfplumber python-multipart
```

4. Start the backend server:
```bash
uvicorn app:app --reload
```

The backend will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will run on http://localhost:3000

## Using the Application

1. Open http://localhost:3000 in your browser
2. Upload study materials (PDFs) using the file upload button
3. Set your test date using the calendar
4. Add or import chapters you need to study
5. Track your progress by marking chapters as complete and logging study time
6. Use the query system to ask questions about your study materials

## Technology Stack

- Backend:
  - FastAPI - Web framework for building APIs
  - pdfplumber - PDF text extraction
  - Simple keyword-based search for MVP
  - (Note: Full RAG functionality with sentence-transformers and FAISS will be added in future releases)

- Frontend:
  - React - UI framework
  - TailwindCSS - Styling
  - Custom UI components for calendar and popover

## Future Enhancements

1. Implement full RAG system with:
   - Hugging Face's sentence-transformers for text embeddings
   - FAISS for efficient vector similarity search
   - Integration with LLMs for better answer generation

2. User authentication and persistent storage
3. Advanced scheduling algorithms
4. PDF annotation and note-taking features
