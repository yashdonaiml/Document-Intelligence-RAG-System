# 🧠 Enhanced Document Q&A System (Local)

This project provides a powerful, locally-run solution for performing Question & Answering on your PDF documents. It features an intelligent system that can understand the structure of your files, automatically classify and separate different documents within a single PDF, and use a local Large Language Model (LLM) to answer your questions based on their content.

The entire system is designed to run on your local machine, ensuring your data remains private. It's built with a Retrieval-Augmented Generation (RAG) pipeline that is robust even without a dedicated GPU, with fallback mechanisms for key features.

![UI Screenshot](Screenshot%202025-12-31%20030835.png)

## Features

- **Local First & Private**: All processing, from document analysis to answer generation, happens on your machine. No data is sent to external services.
- **Interactive UI**: A clean and user-friendly web interface built with Gradio for uploading files and chatting with your documents.
- **Intelligent Document Analysis**:
    - **Automatic Splitting**: If you upload a single PDF containing multiple logical documents (e.g., a loan package with an application, pay stubs, and a tax return), the system automatically detects the boundaries and treats them as separate items.
    - **Document Classification**: Uses a combination of regex and LLM-based analysis to automatically classify documents into types like 'Bank Statement', 'Tax Document', 'Resume', etc.
- **Advanced RAG Pipeline**:
    - **Vector Search**: Uses `sentence-transformers` and `faiss` to create a semantic index of your documents for fast and accurate context retrieval.
    - **Local LLM Integration**: Powered by `llama-cpp-python` to run efficient GGUF models (like Mistral 7B) locally on your CPU.
    - **Source-Grounded Answers**: The LLM is instructed to answer questions based *only* on the information present in your documents, with citations provided for verification.
- **Graceful Fallback**: If the LLM fails to load, the application still functions. The Q&A feature will return the most relevant raw text chunk instead of a generated answer.

## How It Works

The system follows a sophisticated Retrieval-Augmented Generation (RAG) pipeline:

1.  **Upload & Pre-processing**: You upload one or more PDF files. The system uses PyMuPDF (`fitz`) to extract text from each page. A Tesseract OCR fallback is included for scanned pages.
2.  **Document Structuring**: The script iterates through all pages and uses heuristics (like page number resets) and optional LLM checks to determine the boundaries between logical documents. Each detected document is then classified by type.
3.  **Chunking & Indexing**: Each logical document is broken down into smaller, overlapping text chunks. The `all-MiniLM-L6-v2` sentence-transformer model converts these chunks into vector embeddings. These embeddings are stored in a `faiss` index for efficient semantic search.
4.  **Retrieval**: When you ask a question, the query is converted into a vector. The `faiss` index is searched to find the most relevant text chunks from your documents that are semantically similar to your question.
5.  **Generation**: The retrieved chunks (the "context") and your original question are sent to the locally-running Mistral LLM. The model generates a coherent, natural-language answer based on the provided context.
6.  **Response**: The final answer, along with the source documents and page numbers it was derived from, is displayed in the chat interface.

## Setup and Installation

### 1. Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tessdoc): This is required for the OCR fallback on scanned documents. Make sure to add it to your system's PATH during installation.

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-url>
```

### 3. Install Dependencies
It's recommended to use a virtual environment.
```bash
# Create a virtual environment
python -m venv .venv
# Activate it (Windows)
.\.venv\Scripts\activate
# (macOS/Linux)
# source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 4. Download the LLM
The system is configured to use the **Mistral 7B Instruct v0.2 GGUF** model.

1.  Create a `models` directory in the root of the project.
2.  Download the model file (`mistral-7b-instruct-v0.2.Q4_K_M.gguf`) from a trusted source like Hugging Face.
    - [Link to model on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
3.  Place the downloaded `.gguf` file inside the `models` directory. The final path should be `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

## Usage

1.  Ensure all dependencies are installed and the LLM is in the correct directory.
2.  Run the main script from the project's root directory:
    ```bash
    python final_document_q&a_ai.py
    ```
3.  Wait for the models to load. You will see progress messages in the console.
4.  Once the Gradio server starts, open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).
5.  **Upload** one or more PDF files using the file input box.
6.  Click **"Process Files"** and wait for the analysis to complete. The status and detected document structure will be displayed.
7.  **Ask questions** about your documents in the chatbox and get answers!

## Configuration

You can modify the `final_document_q&a_ai.py` script to change the default configuration:

- **LLM Model**: To use a different GGUF model, change the `MODEL_PATH` variable.
- **Embedding Model**: To use a different sentence-transformer, change the `EMBEDDING_MODEL_NAME` variable.
- **LLM Parameters**: You can adjust parameters like context size (`n_ctx`) or thread count (`n_threads`) within the `Llama` initialization block.
