# RAG-Based Document Question Answering System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that answers questions using information retrieved from custom documents.

## Problem Statement

General-purpose LLMs may not have access to private or domain-specific information. This project uses document retrieval to provide relevant context to an LLM and generate more context-aware answers.

## Features

- Document loading and text extraction
- Text chunking for efficient retrieval
- Semantic embeddings using `all-MiniLM-L6-v2`
- Vector storage and similarity search using PostgreSQL and pgvector
- Context-aware answer generation using Google Gemini
- Web search module for extended information retrieval

## Tech Stack

- Python
- LangChain
- Sentence Transformers
- PostgreSQL
- pgvector
- Google Gemini

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/parikansal20/RAG-Project.git
   cd RAG-Project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure PostgreSQL, pgvector, and your Gemini API key.

4. Run the application:

   ```bash
   python main.py
   ```

## Future Improvements

- Retrieval and response quality evaluation
- Hybrid search and reranking
- Improved handling of irrelevant queries
- Performance optimization

## Author

**Pari Kansal**

[GitHub Profile](https://github.com/parikansal20)
