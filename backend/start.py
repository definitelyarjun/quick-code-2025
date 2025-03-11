#!/usr/bin/env python3
"""
Startup script for the RAG Study Assistant API.
This script verifies the environment and starts the FastAPI application.
"""

import os
import sys
import subprocess
import importlib.util
import shutil

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "pdfplumber", 
        "sentence_transformers", "faiss", "numpy", 
        "transformers", "torch", "tqdm", "fitz"
    ]
    
    missing = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            # Special case for PyMuPDF
            if package == "fitz" and importlib.util.find_spec("PyMuPDF") is not None:
                continue
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_default_pdf():
    """Check if the default PDF exists."""
    module_pdf = "Module_1.pdf"
    
    # Check in current directory
    if os.path.exists(module_pdf):
        print(f"Found default PDF: {module_pdf}")
        return True
    
    # Check in uploaded_pdfs directory
    pdf_dir = "uploaded_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, module_pdf)
    
    if os.path.exists(pdf_path):
        print(f"Found default PDF in {pdf_dir}: {module_pdf}")
        return True
    
    # PDF not found
    print(f"Warning: Default PDF '{module_pdf}' not found.")
    print(f"Place '{module_pdf}' in the current directory or in the '{pdf_dir}' directory for automatic processing.")
    return False

def start_server():
    """Start the FastAPI server."""
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("model_cache", exist_ok=True)
    os.makedirs("model_cache/embeddings", exist_ok=True)
    os.makedirs("uploaded_pdfs", exist_ok=True)
    os.makedirs("feedback", exist_ok=True)
    
    # Determine uvicorn path
    uvicorn_path = shutil.which("uvicorn")
    if not uvicorn_path:
        print("Error: uvicorn not found in PATH. Please install with: pip install uvicorn")
        return False
    
    print("\n" + "=" * 50)
    print("Starting RAG Study Assistant API...")
    print("=" * 50)
    print(f"Server will be available at: http://localhost:8000")
    print(f"API documentation will be available at: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    # Start the server
    try:
        subprocess.run([uvicorn_path, "app:app", "--reload"])
        return True
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return True
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def main():
    """Main function."""
    print("RAG Study Assistant API - Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check default PDF
    check_default_pdf()
    
    # Start server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main() 