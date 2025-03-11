#!/usr/bin/env python3
"""
Test script for the RAG system.
This demonstrates how to use the RAG API to query documents.
"""

import requests
import json
import sys
import os

# Configuration
API_URL = "http://localhost:8000"

def test_model_status():
    """Test the model status endpoint."""
    print("Testing model status...")
    response = requests.get(f"{API_URL}/model-status")
    print(json.dumps(response.json(), indent=2))
    print("=" * 50)

def upload_document(pdf_path):
    """Upload a document to the RAG system."""
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist.")
        return False
    
    print(f"Uploading document: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/ingest", files=files)
    
    if response.status_code == 200:
        print("Document uploaded successfully.")
        print(json.dumps(response.json(), indent=2))
        print("=" * 50)
        return True
    else:
        print(f"Error uploading document: {response.status_code}")
        print(response.text)
        print("=" * 50)
        return False

def query_document(query_text, max_tokens=512, temperature=0.7, top_k=5):
    """Query the document using RAG."""
    print(f"Querying: '{query_text}'")
    
    # Use the POST endpoint for more control
    payload = {
        "query": query_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k
    }
    
    response = requests.post(f"{API_URL}/query", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n🤖 ANSWER:")
        print(result["answer"])
        
        print("\n📚 SOURCES:")
        for i, chunk in enumerate(result["relevant_chunks"]):
            print(f"\nSource {i+1}:")
            # Print a preview of the chunk (first 150 chars)
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            print(preview)
        
        print("=" * 50)
        return result
    else:
        print(f"Error querying document: {response.status_code}")
        print(response.text)
        print("=" * 50)
        return None

def main():
    """Main function."""
    # Check if the server is running
    try:
        requests.get(f"{API_URL}/test-cors")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {API_URL}")
        print("Make sure the server is running with 'uvicorn app:app --reload'")
        sys.exit(1)
    
    # Test model status
    test_model_status()
    
    # Upload document if specified
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        upload_document(pdf_path)
    
    # Interactive query loop
    print("Enter queries to test the RAG system (or 'exit' to quit):")
    while True:
        query = input("\nYour query: ")
        if query.lower() in ("exit", "quit", "q"):
            break
        
        query_document(query)

if __name__ == "__main__":
    main() 