# -*- coding: utf-8 -*-
"""final - Document Q&A AI

Automated for Local Windows Execution
"""

import os
import sys
import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

# Third-party imports
import gradio as gr
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure model directory exists
if not os.path.exists("models"):
    try:
        os.makedirs("models")
    except OSError:
        pass

# Initialize LLM
mistral_llm = None
try:
    from llama_cpp import Llama
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading LLM from {MODEL_PATH}...")
            # Attempt to avoid GPU offload if that was causing issues, or rely on auto
            mistral_llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=0, # Force CPU to avoid CUDA mismatches
                chat_format="mistral-instruct",
                verbose=False,
            )
            print("✅ LLM Loaded Successfully")
        except Exception as e:
            print(f"⚠️ Failed to load LLM: {e}")
    else:
        print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}")
except ImportError:
    print("Error: llama-cpp-python not installed. Features dependent on LLM will use heuristics.")

def llm_generate(prompt: str, max_tokens=256, temperature=0.2) -> str:
    if not mistral_llm:
        # Graceful failure - silent return so fallback logic takes over
        return "" 
    
    try:
        resp = mistral_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You must follow instructions exactly. Output must be strictly formatted."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["</s>", "\n\n\n"],
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return ""

# Initialize embedding models
print("Loading embedding models...")
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    embed_model = None

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class PageInfo:
    """Stores information about a single page"""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    filename: Optional[str] = None

@dataclass
class LogicalDocument:
    """Represents a logical document within a PDF"""
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    filename: str
    chunks: List[Dict] = None

@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk"""
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    filename: str
    embedding: Optional[np.ndarray] = None

# -----------------------------------------------------------------------------
# Document Intelligence Functions (Heuristic & Regex Enhanced)
# -----------------------------------------------------------------------------

def classify_document_type(text: str, max_length: int = 2000) -> str:
    """
    Classify the document type based on its content using Regex patterns first, then LLM.
    """
    text_sample = text[:max_length].lower()

    # Robust regex patterns for classification
    patterns = {
        'Income/PaySlip': [r'pay\s?stub', r'pay\s?slip', r'earnings\s statement', r'wage\s statement'],
        'Tax Document': [r'w-2', r'1040', r'tax\s return', r'internal revenue service'],
        'Bank Statement': [r'bank\s statement', r'account\s summary', r'checking\s account', r'savings\s account'],
        'Credit Report': [r'credit\s report', r'credit\s score', r'equifax', r'transunion', r'experian'],
        'Appraisal': [r'appraisal\s report', r'residential\s appraisal', r'market\s value'],
        'Purchase Contract': [r'purchase\s and\s sale', r'sales\s contract', r'real\s estate\s purchase'],
        'Deed': [r'deed\s of\s trust', r'warranty\s deed', r'quitclaim\s deed'],
        'Note': [r'promissory\s note', r'fixed\s rate\s note', r'multistate\s note'],
        'Closing Disclosure': [r'closing\s disclosure', r'settlement\s statement', r'hud-1'],
        'Loan Application': [r'uniform\s residential\s loan\s application', r'form\s 1003', r'borrower\s information'],
        # Improved Resume Pattern: Only matches if typical sections appear, not just the word "resume"
        'Resume': [r'\bresume\b.*experience', r'curriculum\s vitae', r'education.*skills'],
        'Invoice': [r'invoice\s #', r'bill\s to:', r'amount\s due'],
    }

    # 1. Regex Match
    for doc_type, regex_list in patterns.items():
        for pattern in regex_list:
            if re.search(pattern, text_sample):
                print(f"🔍 Classified as {doc_type} via regex: '{pattern}'")
                return doc_type

    # 2. LLM Fallback (Only if regex fails and LLM is loaded)
    if mistral_llm:
        valid_types = list(patterns.keys()) + ['Other']
        prompt = f"""
Classify this document sample into exactly ONE of these types:
{", ".join(valid_types)}

Sample:
{text_sample[:1000]}

Category:
""".strip()
        try:
            raw = llm_generate(prompt, max_tokens=20, temperature=0.0)
            cleaned = re.sub(r"[^A-Za-z /]", "", raw).strip()
            # Fuzzy match result to valid types
            for t in valid_types:
                if t.lower() in cleaned.lower():
                    return t
        except Exception:
            pass
            
    return "Other"

def detect_document_boundary(prev_text: str, curr_text: str, current_doc_type: str = None) -> bool:
    """
    Detect if two consecutive pages belong to the same document.
    """
    if not prev_text or not curr_text:
        return False
    
    # 1. Page Number Heuristics (Strongest)
    # Check for "Page 1 of X" or just "Page 1" at the start of current page
    page_reset_pattern = r'page\s+1(?!\d)'
    curr_header = curr_text[:500].lower()
    
    if re.search(page_reset_pattern, curr_header):
        print("✂️ Boundary detected: Page number reset")
        return False # New document

    # 2. Similarity Check (Jaccard on sets of words)
    # If the vocabulary changes drastically, it's likely a new document
    def get_words(t): return set(re.findall(r'\w{4,}', t.lower()))
    prev_words = get_words(prev_text)
    curr_words = get_words(curr_text)
    
    if not prev_words or not curr_words:
        return True
        
    intersection = len(prev_words.intersection(curr_words))
    union = len(prev_words.union(curr_words))
    jaccard = intersection / union if union > 0 else 0
    
    # 3. Type Consistency Check via Regex
    # If current page explicitly looks like a START of a new specific doc type
    start_patterns = [
        r'uniform\s residential\s loan\s application',
        r'appraisal\s report',
        r'closing\s disclosure'
    ]
    for p in start_patterns:
        if re.search(p, curr_header):
            print(f"✂️ Boundary detected: New document header '{p}'")
            return False

    # 4. LLM Verification (Only if ambiguous and LLM available)
    if mistral_llm and jaccard < 0.1: # Low similarity, ask LLM
        prompt = f"""
Do these pages belong to the SAME document? Answer JSON {{ "same": true/false }}.

End of Page A:
{prev_text[-400:]}

Start of Page B:
{curr_text[:400]}
"""
        try:
            raw = llm_generate(prompt, max_tokens=50)
            if "false" in raw.lower():
                return False
        except:
            pass

    # Default: Assume same document if no strong signal to split
    return True

# -----------------------------------------------------------------------------
# PDF Processing (Multi-File Support)
# -----------------------------------------------------------------------------

def extract_and_analyze_pdf(pdf_path: str) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Extract text from PDF and perform intelligent document analysis.
    """
    print(f"📖 Processing: {pdf_path}")
    
    filename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_path}: {e}")
        raise

    pages_info = []
    for i, page in enumerate(doc):
        text = page.get_text()
        
        # Simple generic OCR fallback
        if not text.strip():
            try:
                pix = page.get_pixmap(dpi=300)
                from PIL import Image
                import pytesseract
                import io
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            except Exception:
                pass
        
        pages_info.append(PageInfo(page_num=i, text=text, filename=filename))

    doc.close()
    
    if not pages_info:
        print(f"Warning: No text found in {filename}")
        return [], []

    print(f"✅ Extracted {len(pages_info)} pages from {filename}")
    
    # Analyze Structure (Heuristic Loop)
    logical_docs = []
    current_doc_type = None
    current_doc_pages = []
    doc_counter = 0

    for i, page_info in enumerate(pages_info):
        if i == 0:
            current_doc_type = classify_document_type(page_info.text)
            page_info.doc_type = current_doc_type
            page_info.page_in_doc = 0
            current_doc_pages = [page_info]
        else:
            prev_text = pages_info[i-1].text
            # Use stronger heuristic boundary detector
            is_same = detect_document_boundary(prev_text, page_info.text, current_doc_type)

            if is_same:
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = len(current_doc_pages)
                current_doc_pages.append(page_info)
            else:
                # Finalize previous
                logical_doc = LogicalDocument(
                    doc_id=f"{filename}_doc_{doc_counter}",
                    doc_type=current_doc_type,
                    page_start=current_doc_pages[0].page_num,
                    page_end=current_doc_pages[-1].page_num,
                    text="\n\n".join([p.text for p in current_doc_pages]),
                    filename=filename
                )
                logical_docs.append(logical_doc)
                doc_counter += 1

                # Start new
                current_doc_type = classify_document_type(page_info.text)
                print(f"  ➜ Page {i}: Start of new document ({current_doc_type})")
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = 0
                current_doc_pages = [page_info]

    # Add last document
    if current_doc_pages:
        logical_doc = LogicalDocument(
            doc_id=f"{filename}_doc_{doc_counter}",
            doc_type=current_doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text="\n\n".join([p.text for p in current_doc_pages]),
            filename=filename
        )
        logical_docs.append(logical_doc)

    return pages_info, logical_docs

# -----------------------------------------------------------------------------
# Chunking (Improved Overlap)
# -----------------------------------------------------------------------------

def chunk_document_with_metadata(logical_doc: LogicalDocument,
                                chunk_size: int = 500,
                                overlap: int = 150) -> List[ChunkMetadata]: 
    """
    Chunk a logical document while preserving rich metadata.
    """
    chunks_metadata = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_0",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=0,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=logical_doc.text,
            filename=logical_doc.filename,
        )
        chunks_metadata.append(chunk_meta)
    else:
        stride = chunk_size - overlap
        for i, start_idx in enumerate(range(0, len(words), stride)):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            try:
                chunk_position = start_idx / len(words)
                page_range = logical_doc.page_end - logical_doc.page_start
                relative_page = int(chunk_position * page_range)
                chunk_page_start = logical_doc.page_start + relative_page
                chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)
            except ZeroDivisionError:
                chunk_page_start = logical_doc.page_start
                chunk_page_end = logical_doc.page_end

            chunk_meta = ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=chunk_page_start,
                page_end=chunk_page_end,
                text=chunk_text,
                filename=logical_doc.filename,
            )
            chunks_metadata.append(chunk_meta)
            
            if end_idx >= len(words):
                break

    return chunks_metadata

# -----------------------------------------------------------------------------
# Intelligent Retrieval with Query Signals & Boost (Fallback Enhanced)
# -----------------------------------------------------------------------------

def extract_query_signals(query: str) -> List[str]:
    """
    Extract semantic signals - HEURISTIC FALLBACK if LLM fails.
    """
    if not mistral_llm:
        # Simple extraction of nouns/keywords
        ignore_words = {'the', 'a', 'an', 'what', 'is', 'are', 'in', 'of', 'for', 'to', 'document', 'file', 'this', 'that', 'with', 'and', 'or'}
        words = re.findall(r'\b\w{4,}\b', query.lower())
        return [w for w in words if w not in ignore_words]

    prompt = f"""
Extract 3–5 key semantic signals (nouns) from this question. JSON list ONLY.
Question: {query}
"""
    try:
        raw = llm_generate(prompt, max_tokens=80, temperature=0.0)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except:
        return []

def semantic_boost(retrieved_chunks: List[Tuple[ChunkMetadata, float]], 
                  query_signals: List[str]) -> List[Tuple[ChunkMetadata, float]]:
    """
    Boost score if signals are found in chunk text.
    """
    if not query_signals:
        return retrieved_chunks
        
    boosted = []
    for chunk, score in retrieved_chunks:
        text = chunk.text.lower()
        overlap = sum(1 for s in query_signals if s.lower() in text)
        if overlap > 0:
            score = min(score + 0.05 * overlap, 1.0)
        boosted.append((chunk, score))
    
    return boosted

class IntelligentRetriever:
    def __init__(self):
        self.index = None
        self.chunks_metadata = []
        self.doc_type_indices = {}
        self.total_queries = 0
        self.cache_hits = 0

    def build_indices(self, chunks_metadata: List[ChunkMetadata]):
        print("🔨 Building vector indices...")
        self.chunks_metadata = chunks_metadata
        
        if not chunks_metadata:
            return

        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = embed_model.encode(texts, show_progress_bar=True)

        for i, chunk in enumerate(chunks_metadata):
            chunk.embedding = embeddings[i]

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        doc_types = set(chunk.doc_type for chunk in chunks_metadata)
        self.doc_type_indices = {}
        
        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata) if chunk.doc_type == doc_type]
            if type_indices:
                type_embeddings = embeddings[type_indices]
                type_index = faiss.IndexFlatL2(dim)
                type_index.add(type_embeddings)
                self.doc_type_indices[doc_type] = {
                    'index': type_index,
                    'mapping': type_indices
                }

    def retrieve(self, query: str, k: int = 4,
                filter_doc_type: Optional[str] = None) -> List[Tuple[ChunkMetadata, float]]:
        
        self.total_queries += 1
        query_embedding = embed_model.encode([query])

        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            type_data = self.doc_type_indices[filter_doc_type]
            D, I = type_data['index'].search(query_embedding, k)
            chunk_indices = [type_data['mapping'][i] for i in I[0]]
            distances = D[0]
        else:
            D, I = self.index.search(query_embedding, k)
            chunk_indices = I[0]
            distances = D[0]

        max_dist = max(distances) if len(distances) > 0 else 1.0
        if max_dist == 0: max_dist = 1.0
        
        scores = [(max_dist - d) / max_dist for d in distances]
        
        results = []
        for idx, i in enumerate(chunk_indices):
            if i < len(self.chunks_metadata):
                results.append((self.chunks_metadata[i], scores[idx]))

        return results

# -----------------------------------------------------------------------------
# Answer Generation (Fallback Aware)
# -----------------------------------------------------------------------------

def generate_answer_with_sources(query: str, 
                               retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    
    query_signals = extract_query_signals(query)
    retrieved_chunks = semantic_boost(retrieved_chunks, query_signals)
    retrieved_chunks = [(c, s) for c, s in retrieved_chunks if s > 0.2]
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)
    
    # Build Sources List First
    sources = []
    for chunk, score in retrieved_chunks:
        sources.append({
            'doc_type': chunk.doc_type,
            'filename': chunk.filename,
            'pages': f"{chunk.page_start}-{chunk.page_end}",
            'relevance': f"{score:.2%}",
            'preview': chunk.text[:100].replace("\n", " ") + "..."
        })

    if not retrieved_chunks:
        return {'answer': "No relevant information found.", 'sources': [], 'confidence': 0.0}

    # If LLM is missing, return a "Best Match" extraction instead of trying to generate
    if not mistral_llm:
        best_chunk = retrieved_chunks[0][0]
        return {
            'answer': f"**Note: LLM not loaded. Showing most relevant text from document:**\n\n{best_chunk.text}",
            'sources': sources,
            'confidence': retrieved_chunks[0][1]
        }

    # Standard LLM Generation
    context_parts = []
    for chunk, _ in retrieved_chunks:
        context_parts.append(f"[SOURCE: {chunk.doc_type} p{chunk.page_start}-{chunk.page_end}]\n{chunk.text}")
    context = "\n\n".join(context_parts)

    prompt = f"""
Answer based ONLY on these sources.
QUESTION: {query}

SOURCES:
{context}

ANSWER:
""".strip()

    try:
        answer = llm_generate(prompt, max_tokens=350)
        return {'answer': answer, 'sources': sources, 'confidence': retrieved_chunks[0][1]}
    except:
        return {'answer': "Error generating answer.", 'sources': sources, 'confidence': 0.0}

# -----------------------------------------------------------------------------
# Enhanced Document Store (Multi-File)
# -----------------------------------------------------------------------------

class EnhancedDocumentStore:
    def __init__(self):
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.retriever = IntelligentRetriever()
        self.is_ready = False
        self.processing_stats = {}

    def process_files(self, file_paths: List[str]):
        """
        Process multiple PDF files and build/update index.
        """
        start_time = datetime.now()
        
        new_pages = 0
        new_docs = 0
        new_chunks = 0
        
        for path in file_paths:
            # Extract
            p_info, l_docs = extract_and_analyze_pdf(path)
            self.pages_info.extend(p_info)
            self.logical_docs.extend(l_docs)
            
            # Chunk
            for doc in l_docs:
                chunks = chunk_document_with_metadata(doc)
                doc.chunks = chunks
                self.chunks_metadata.extend(chunks)
                new_chunks += len(chunks)
            
            new_pages += len(p_info)
            new_docs += len(l_docs)

        # Re-build index (Full rebuild for simplicity in this script)
        self.retriever.build_indices(self.chunks_metadata)
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        self.processing_stats = {
            'total_files': len(set(p.filename for p in self.pages_info)),
            'total_pages': len(self.pages_info),
            'documents_found': len(self.logical_docs),
            'total_chunks': len(self.chunks_metadata),
            'processing_time': f"{process_time:.1f}s"
        }
        
        self.is_ready = True
        return True, self.processing_stats

    def query(self, question: str, filter_type: Optional[str] = None, 
             auto_route: bool = True, k: int = 4) -> Dict:
        if not self.is_ready:
            return {'answer': "Please upload documents first.", 'sources': [], 'confidence': 0.0}
            
        retrieved = self.retriever.retrieve(question, k=k, filter_doc_type=filter_type)
        return generate_answer_with_sources(question, retrieved)

    def get_structure(self):
        return [
            {
                'type': d.doc_type,
                'filename': d.filename,
                'pages': f"{d.page_start}-{d.page_end}",
                'chunks': len(d.chunks or [])
            }
            for d in self.logical_docs
        ]

# -----------------------------------------------------------------------------
# Gradio Interface (Fixed Chat History & Inputs)
# -----------------------------------------------------------------------------

doc_store = EnhancedDocumentStore()

def process_upload(files):
    if not files:
        return "⚠️ No files uploaded.", "", gr.update(choices=["All"])
    
    # gr.Files returns list of paths in recent versions
    file_paths = [f.name for f in files] if hasattr(files[0], 'name') else files
    
    success, stats = doc_store.process_files(file_paths)
    
    if success:
        msg = f"""
        ✅ **Processed {stats['total_files']} Files**
        - Pages: {stats['total_pages']}
        - Logical Docs: {stats['documents_found']}
        - Chunks: {stats['total_chunks']}
        - Time: {stats['processing_time']}
        """
        
        structure = doc_store.get_structure()
        display = "\n".join([f"• [{d['filename']}] **{d['type']}** (p{d['pages']})" for d in structure])
        
        # Update filter options based on doc types found
        doc_types = ["All"] + list(set(d['type'] for d in structure))
        
        return msg, display, gr.update(choices=doc_types)
    else:
        return "❌ Error processing files.", "", gr.update()

def chat_handler(message, history, doc_filter, auto_route, num_chunks):
    history = history or []
    
    if not doc_store.is_ready:
        response = "Please upload documents first."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history

    # Query
    filter_val = None if doc_filter == "All" else doc_filter
    result = doc_store.query(message, filter_type=filter_val, auto_route=auto_route, k=num_chunks)
    
    # Format Response
    ans = result['answer']
    if result['sources']:
        ans += "\n\n📍 **Sources:**\n"
        for src in result['sources']:
            ans += f"• [{src['filename']}] {src['doc_type']} (p{src['pages']}) - {src['relevance']}\n"
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ans})
    
    return history

def ask_summary(history, doc_filter, auto_route, num_chunks):
    return chat_handler(
        "Can you provide a summary of the main points in this document?",
        history, doc_filter, auto_route, num_chunks
    )

def ask_amounts(history, doc_filter, auto_route, num_chunks):
    return chat_handler(
        "What are all the monetary amounts or financial figures mentioned?",
        history, doc_filter, auto_route, num_chunks
    )

def clear_all():
    global doc_store
    doc_store = EnhancedDocumentStore()
    return None, "Wait for upload...", "", gr.update(choices=["All"]), [], ""

def create_interface():
    with gr.Blocks(title="Enhanced Document Q&A", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Enhanced Document Q&A System (Local)")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="📄 Upload PDFs", 
                    file_count="multiple",
                    file_types=[".pdf"]
                )
                process_btn = gr.Button("🔄 Process Files", variant="primary")
                clear_btn = gr.Button("🗑️ Clear All")
                
                status_out = gr.Markdown("⏳ Waiting for upload...")
                structure_out = gr.Markdown(label="Document Structure")
                
                gr.Markdown("### Settings")
                doc_filter = gr.Dropdown(choices=["All"], value="All", label="Filter by Type")
                auto_route = gr.Checkbox(value=True, label="Auto-Route")
                num_chunks = gr.Slider(1, 10, value=4, label="Chunks")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                msg_input = gr.Textbox(placeholder="Ask a question...", label="Question")
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_chat_btn = gr.Button("Clear Chat")
                
                with gr.Row():
                    summ_btn = gr.Button("📝 Summary")
                    amt_btn = gr.Button("💰 Amounts")

        # Events
        process_btn.click(
            process_upload,
            inputs=[pdf_input],
            outputs=[status_out, structure_out, doc_filter]
        )
        
        clear_btn.click(
            clear_all,
            outputs=[pdf_input, status_out, structure_out, doc_filter, chatbot, msg_input]
        )
        
        # Chat Events - ensure all inputs are passed
        msg_input.submit(
            chat_handler,
            inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])
        
        send_btn.click(
            chat_handler,
            inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])
        
        # Example Buttons - Passing inputs explicitly to avoid closure issues
        summ_btn.click(
            ask_summary,
            inputs=[chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        )
        
        amt_btn.click(
            ask_amounts,
            inputs=[chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        )
        
        clear_chat_btn.click(lambda: [], outputs=[chatbot])

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
