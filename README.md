# Study Planner and Learning Partner.

A personalized, college-specific study planner that uses RAG (Retrieval-Augmented Generation) to help students organize their study schedule and get answers from their study materials, ALso helps teachers plan what to teach based on content material sourced from KTU Notes.

## Commit history timeline

https://repo.surf/definitelyarjun/quick-code-2025

## Project Architecture

![Architecture](https://github.com/user-attachments/assets/5ec952e5-c44b-4029-9d39-c16fe22fcf4b)
```
+------+     +---------+             +---------+
| User |     | Frontend|             | Backend |
+------+     +---------+             +---------+
   |              |                       |
   | Upload Study |     Process PDFs      |
   | Materials -->|---------------------->|
   |              |                       |
   | Set Test Date|                       |
   | ----------->|                        |
   |              |                       |
   | Input Study  |   Generate Schedule   |
   | Progress --> |---------------------->|
   |              |                       |
   |              |  Return Adaptive Plan |
   |              |<----------------------| 
   |              |                       |
   | Ask Questions|       RAG Query       |
   | ----------->|----------------------->|
   |              |                       |
   |              | Context-Aware Response|
   |              |<----------------------| 
   |              |                       |
+------+     +---------+             +---------+
| User |     | Frontend|             | Backend |
+------+     +---------+             +---------+
```

## Team Members

Arjun Jayakumar (Team Lead- Backend and LLM's)
Aditya S Nair (Frontend)
Abhiram Krishna (UI and Presentation)
Rajkamal SP (UI,UX Design)

## Features

- A vector database consisting of prexisting notes fetched from KTU Notes.
- Track study progress by chapter
- Generate adaptive study schedules based on:
  - Learning pace
  - Content complexity
  - Test deadline
- Query study materials for relevant information, Also test your skills by solving previous year Question papers.
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
pip install -r requirements.txt

4. Start the backend server:
```bash
python start.py
```

The backend will run on http://localhost:8000

### Frontend Setup (in a separate terminal)

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
2. Upload study materials (PDFs) using the file upload button (In the future this wont be neccessary as we will store the vecotirized data in the PDF to a DB, Also this is only required on the first run after which the model uses cached data.)
4. Ask queries regarding the subject material and also ask questions regarding study planning
5. Track your progress by marking chapters as complete and logging study time

## Interface Demo Screenshot

![Interface](https://github.com/user-attachments/assets/f6606461-fc24-49a6-9130-2b08977b4c88)

## Technology Stack

- Backend Technologies:
   Python 3.8+ as the core programming language
   FastAPI for the web framework
   Pydantic for data validation
   Uvicorn as the ASGI server
   PyMuPDF (fitz) for PDF processing
   Sentence-Transformers for text embeddings
   FAISS for vector similarity search
   Transformers from Hugging Face for NLP models
   PyTorch as the deep learning framework
   Accelerate & BitsAndBytes for model optimization

- Frontend Technologies:
   React 18 for the UI framework
   TailwindCSS for styling
   Lucide React for icons
   React Router for navigation
   date-fns for date handling
   clsx & tailwind-merge for class name utilities

```
quick-code-2025/
├── .gitignore
├── README.md
├── Sample-os.pdf
│
├── backend/
│   ├── .gitignore
│   ├── app.py
│   ├── create_test_pdf.py
│   ├── README.md
│   ├── requirements.txt
│   ├── start.py
│   ├── test_document.txt
│   ├── test_rag.py
│   ├── feedback/
│   ├── model_cache/
│   │   └── embeddings/
│   │       ├── document_chunks.json
│   │       └── document_embeddings.npy
│   ├── uploaded_pdfs/
│   │   ├── Module_1.pdf
│   │   └── Sample-os.pdf
│   └── uploads/
│
├── embeddings/
│   ├── architecture.jpg
│   └── UI - With Functionalities .png
│
└── frontend/
    ├── .gitignore
    ├── package-lock.json
    ├── package.json
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── public/
    │   ├── index.html
    │   └── manifest.json
    └── src/
        ├── App.js
        ├── index.css
        ├── index.js
        ├── reportWebVitals.js
        ├── components/
        │   ├── StudyPlanner.jsx
        │   ├── StudyQuerySystem.jsx
        │   └── ui/
        │       ├── button.jsx
        │       ├── calendar.jsx
        │       ├── card.jsx
        │       ├── chat-bubble.jsx
        │       ├── chat-message-list.jsx
        │       ├── file-upload.jsx
        │       ├── input.jsx
        │       ├── label.jsx
        │       ├── popover.jsx
        │       ├── progress.jsx
        │       ├── study-planner.jsx
        │       ├── study-query-system.jsx
        │       └── tabs.jsx
        ├── lib/
        │   └── utils.js
        └── services/
            └── api.js
```

## Future Enhancements

1. Implement full RAG system with:
   - FAISS for efficient vector similarity search (Current implementation doesnt store the chunked PDF to a Database, In the future we will chunk and vectorize all the study materials sourced from KTU Notes and store them in the DB)
   - LLM's with higher paramters for better results.

2. User authentication and persistent storage
3. Advanced scheduling algorithms
4. PDF annotation and note-taking features
