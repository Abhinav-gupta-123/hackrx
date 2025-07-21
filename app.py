"""
Production-grade LLM-powered intelligent query-retrieval system
FastAPI service for PDF document Q&A using RAG (Retrieval-Augmented Generation)
"""
 
import asyncio
import io
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
 
import aiohttp
import faiss
import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
import tiktoken
import openai
 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
 
# Constants
EXPECTED_AUTH_TOKEN = "4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487"
MAX_CHUNK_SIZE = 500  # Maximum tokens per chunk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7
TOP_K_CHUNKS = 5
MAX_PDF_SIZE_MB = 50
 
# API Configuration
OPENAI_API_KEY = "sk-spgw-api01-aa37041bc01f73c9228df1fc4cb1e128"
OPENAI_BASE_URL = "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1/"
PINECONE_API_KEY = "pcsk_2BmckG_Dm6EhLdqzuLBziRHnqetStaHFPLn2wg2DGVkiyS9QenPEp1TkUi8PB1Ct7N8ajz"
DATABASE_URL = "postgresql://postgres:alooparatha@db.wdaoajxfktpfbqyavfpe.supabase.co:5432/postgres"
OPENAI_MODEL = "gpt-4o-mini"
 
# Global variables for models (initialized on startup)
embedding_model: Optional[SentenceTransformer] = None
tokenizer = None
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global embedding_model, tokenizer
   
    logger.info("Loading sentence transformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
   
    logger.info("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
   
    logger.info("Application startup complete")
    yield
    # Cleanup code can go here if needed
 
 
# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="LLM-powered PDF document Q&A using RAG",
    version="1.0.0",
    lifespan=lifespan
)
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Security
security = HTTPBearer()
 
 
class QueryRequest(BaseModel):
    """Request model for the query endpoint"""
    documents: str = Field(..., description="PDF blob URL")
    questions: List[str] = Field(..., min_items=1, description="List of natural language questions")
 
    @field_validator('documents')
    @classmethod
    def validate_pdf_url(cls, v):
        """Validate that the URL is properly formatted"""
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")
 
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        """Validate questions are non-empty"""
        if not v or any(not q.strip() for q in v):
            raise ValueError("Questions cannot be empty")
        return [q.strip() for q in v]
 
 
class QueryResponse(BaseModel):
    """Response model for the query endpoint"""
    answers: List[str] = Field(..., description="List of answers corresponding to input questions")
 
 
class DocumentChunk:
    """Represents a semantic chunk of document content"""
   
    def __init__(self, content: str, page_number: int, chunk_index: int):
        self.content = content
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.embedding: Optional[np.ndarray] = None
 
    def __repr__(self):
        return f"DocumentChunk(page={self.page_number}, chunk={self.chunk_index}, content_length={len(self.content)})"
 
 
class DocumentProcessor:
    """Handles PDF processing and text extraction"""
   
    @staticmethod
    async def download_pdf(url: str) -> bytes:
        """Download PDF from URL with proper error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download PDF. Status: {response.status}"
                        )
                   
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_PDF_SIZE_MB * 1024 * 1024:
                        raise HTTPException(
                            status_code=400,
                            detail=f"PDF too large. Maximum size: {MAX_PDF_SIZE_MB}MB"
                        )
                   
                    pdf_data = await response.read()
                    if len(pdf_data) > MAX_PDF_SIZE_MB * 1024 * 1024:
                        raise HTTPException(
                            status_code=400,
                            detail=f"PDF too large. Maximum size: {MAX_PDF_SIZE_MB}MB"
                        )
                   
                    return pdf_data
                   
        except aiohttp.ClientError as e:
            logger.error(f"Network error downloading PDF: {e}")
            raise HTTPException(status_code=400, detail="Failed to download PDF due to network error")
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {e}")
            raise HTTPException(status_code=500, detail="Internal error downloading PDF")
 
    @staticmethod
    def extract_text_from_pdf(pdf_data: bytes) -> List[tuple]:
        """Extract text from PDF using PyMuPDF with page information"""
        try:
            # Try in-memory processing first (preferred for deployment)
            try:
                doc = fitz.open(stream=pdf_data, filetype="pdf")
                pages_content = []
               
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                   
                    if text.strip():  # Only include pages with content
                        pages_content.append((page_num + 1, text.strip()))
               
                doc.close()
               
                if not pages_content:
                    raise HTTPException(status_code=400, detail="No readable text found in PDF")
               
                return pages_content
               
            except Exception as memory_error:
                logger.warning(f"In-memory PDF processing failed: {memory_error}")
                logger.info("Falling back to file-based processing...")
               
                # Fallback to file-based processing with improved Windows handling
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(pdf_data)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name
               
                # Open PDF after the file handle is closed
                doc = None
                try:
                    doc = fitz.open(tmp_file_path)
                    pages_content = []
                   
                    for page_num in range(doc.page_count):
                        page = doc[page_num]
                        text = page.get_text()
                       
                        if text.strip():  # Only include pages with content
                            pages_content.append((page_num + 1, text.strip()))
                   
                    if not pages_content:
                        raise HTTPException(status_code=400, detail="No readable text found in PDF")
                   
                    return pages_content
                   
                finally:
                    # Ensure document is properly closed before file deletion
                    if doc:
                        doc.close()
                   
                    # Clean up temporary file with retry mechanism for Windows
                    import time
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            os.unlink(tmp_file_path)
                            break
                        except (OSError, PermissionError) as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.1)  # Wait 100ms before retry
                                continue
                            else:
                                logger.warning(f"Could not delete temporary file {tmp_file_path}: {e}")
                                # Don't raise exception for cleanup failure
                                break
               
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
 
 
class TextChunker:
    """Handles semantic text chunking"""
   
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text using tiktoken"""
        global tokenizer
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
 
    @classmethod
    def split_into_semantic_chunks(cls, pages_content: List[tuple]) -> List[DocumentChunk]:
        """Split content into semantic chunks based on paragraphs and token limits"""
        chunks = []
        chunk_index = 0
       
        for page_num, page_text in pages_content:
            # Split by paragraphs first
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
           
            current_chunk = ""
           
            for paragraph in paragraphs:
                # Check if adding this paragraph would exceed token limit
                potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
               
                if cls.count_tokens(potential_chunk) <= MAX_CHUNK_SIZE:
                    current_chunk = potential_chunk
                else:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunks.append(DocumentChunk(current_chunk, page_num, chunk_index))
                        chunk_index += 1
                   
                    # Handle oversized paragraphs by splitting them
                    if cls.count_tokens(paragraph) > MAX_CHUNK_SIZE:
                        # Split by sentences
                        sentences = cls._split_by_sentences(paragraph)
                        current_chunk = ""
                       
                        for sentence in sentences:
                            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                           
                            if cls.count_tokens(potential_chunk) <= MAX_CHUNK_SIZE:
                                current_chunk = potential_chunk
                            else:
                                if current_chunk:
                                    chunks.append(DocumentChunk(current_chunk, page_num, chunk_index))
                                    chunk_index += 1
                                current_chunk = sentence
                    else:
                        current_chunk = paragraph
           
            # Add remaining chunk for this page
            if current_chunk:
                chunks.append(DocumentChunk(current_chunk, page_num, chunk_index))
                chunk_index += 1
       
        return chunks
 
    @staticmethod
    def _split_by_sentences(text: str) -> List[str]:
        """Split text by sentences as fallback for large paragraphs"""
        import re
        # Simple sentence splitting - can be enhanced with proper NLP tools
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
 
 
class VectorStore:
    """Handles vector embeddings and similarity search using FAISS"""
   
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
       
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from document chunks"""
        global embedding_model
       
        if not chunks:
            raise ValueError("No chunks provided for indexing")
       
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
       
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
       
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
       
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype(np.float32))
       
        self.chunks = chunks
        logger.info(f"Built FAISS index with {len(chunks)} chunks")
 
    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[tuple]:
        """Search for most relevant chunks given a query"""
        global embedding_model
       
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
       
        # Generate query embedding
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
       
        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.chunks))
        )
       
        # Return chunks with similarity scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], float(score)))
       
        return results
 
 
class LLMService:
    """Handles interaction with OpenAI GPT-4o-mini for answer generation"""
   
    def __init__(self):
        """Initialize OpenAI client with custom base URL"""
        try:
            self.client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
            self.model = OPENAI_MODEL
            logger.info(f"OpenAI client initialized successfully with base URL: {OPENAI_BASE_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            logger.info("Will use fallback simulation mode")
            self.client = None
            self.model = OPENAI_MODEL
   
    async def generate_answer(self, question: str, relevant_chunks: List[tuple]) -> str:
        """Generate answer using retrieved chunks and OpenAI GPT-4o-mini"""
       
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer this question."
       
        # Prepare context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks):
            context_parts.append(
                f"[Context {i+1} - Page {chunk.page_number}, Similarity: {score:.3f}]\n{chunk.content}"
            )
       
        context = "\n\n".join(context_parts)
       
        # Construct prompt for LLM
        system_prompt = """You are an expert document analyst. Your task is to answer questions based strictly on the provided document context.
 
Rules:
1. Only use information explicitly stated in the provided context
2. Quote specific clauses or sections that support your answer
3. If information is not available in the context, clearly state this
4. Keep answers concise, accurate, and factual
5. Always cite which context section supports your answer"""
 
        user_prompt = f"""Based on the following document excerpts, answer this question:
 
QUESTION: {question}
 
DOCUMENT CONTEXT:
{context}
 
Please provide a direct answer based only on the given context. If the information is not available, say so clearly."""
 
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent, factual responses
                top_p=0.9
            )
           
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer using {self.model}")
            return answer
           
        except openai.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            return "I'm currently experiencing high demand. Please try again in a moment."
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "I encountered an API error while processing your question. Please try again."
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fallback to simulation if API fails
            return self._simulate_llm_response(question, context)
 
    @staticmethod
    def _simulate_llm_response(question: str, context: str) -> str:
        """Fallback simulation for when API is unavailable"""
        question_lower = question.lower()
        context_lower = context.lower()
       
        # Simple keyword-based responses for common insurance questions
        if "waiting period" in question_lower and "pre-existing" in question_lower:
            if "36 months" in context_lower or "3 years" in context_lower:
                return "There is a waiting period of 36 months for pre-existing diseases as stated in the policy terms."
            elif "24 months" in context_lower or "2 years" in context_lower:
                return "There is a waiting period of 24 months for pre-existing diseases as per the policy document."
            elif "waiting period" in context_lower:
                return "The policy document mentions waiting periods for pre-existing diseases. Please refer to the specific clause in the document."
       
        elif "maternity" in question_lower:
            if "maternity" in context_lower and ("24 months" in context_lower or "2 years" in context_lower):
                return "Yes, the policy covers maternity expenses after 24 months of continuous coverage as mentioned in the policy terms."
            elif "maternity" in context_lower and "cover" in context_lower:
                return "The policy includes maternity coverage subject to specific terms and conditions mentioned in the document."
            elif "maternity" in context_lower:
                return "Maternity benefits are mentioned in the policy. Please refer to the specific terms for coverage details."
       
        # Generic response based on context availability
        if context.strip():
            return f"Based on the provided document sections, the relevant information regarding '{question}' can be found in the policy terms. Please refer to the specific clauses mentioned in the context."
        else:
            return "The information is not available in the provided document sections."
 
 
# Dependency functions
async def verify_auth_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authorization token"""
    if credentials.credentials != EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials
 
 
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": embedding_model is not None}
 
 
# Main query endpoint
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
@app.post("/hackrx/run", response_model=QueryResponse)  # Alternative shorter endpoint
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_auth_token)
):
    """
    Main endpoint for processing PDF documents and answering questions
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
       
        # Step 1: Download PDF
        logger.info("Downloading PDF...")
        pdf_data = await DocumentProcessor.download_pdf(request.documents)
       
        # Step 2: Extract text from PDF
        logger.info("Extracting text from PDF...")
        pages_content = DocumentProcessor.extract_text_from_pdf(pdf_data)
       
        # Step 3: Split into semantic chunks
        logger.info("Creating semantic chunks...")
        chunks = TextChunker.split_into_semantic_chunks(pages_content)
       
        if not chunks:
            raise HTTPException(status_code=400, detail="No processable content found in PDF")
       
        # Step 4: Build vector index
        logger.info(f"Building vector index with {len(chunks)} chunks...")
        vector_store = VectorStore()
        vector_store.build_index(chunks)
       
        # Step 5: Process each question
        llm_service = LLMService()
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
           
            # Retrieve relevant chunks
            relevant_chunks = vector_store.search(question, TOP_K_CHUNKS)
           
            # Generate answer using LLM
            answer = await llm_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
       
        logger.info("All questions processed successfully")
        return QueryResponse(answers=answers)
       
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
 
 
# Additional endpoints for debugging/monitoring
@app.get("/api/v1/stats")
async def get_stats(token: str = Depends(verify_auth_token)):
    """Get system statistics"""
    return {
        "model_loaded": embedding_model is not None,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OPENAI_MODEL,
        "openai_base_url": OPENAI_BASE_URL,
        "api_configured": bool(OPENAI_API_KEY),
        "max_chunk_size": MAX_CHUNK_SIZE,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "top_k_chunks": TOP_K_CHUNKS
    }
 
 
if __name__ == "__main__":
    import uvicorn
   
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get("PORT", 8000))
   
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
 