import os
import time
import logging
from openai import OpenAI
 
# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
 
# Set your custom endpoint config
api_key_4o_mini = "sk-spgw-api01-aa37041bc01f73c9228df1fc4cb1e128"
api_base_4o_mini = "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1/"
deployment_name_4o_mini = "gpt-4o-mini"  

# Authentication
EXPECTED_AUTH_TOKEN = "4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487"

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_SIZE = 500
SIMILARITY_THRESHOLD = 0.7
TOP_K_CHUNKS = 5
MAX_PDF_SIZE_MB = 50

# Prompts for Document Q&A
system_prompt = """You are an expert document analyst. Your task is to answer questions based strictly on the provided document context. 

Rules:
1. Only use information explicitly stated in the provided context
2. Quote specific clauses or sections that support your answer
3. If information is not available in the context, clearly state this
4. Keep answers concise, accurate, and factual

user_prompt = """Based on the following document excerpts, answer this question:

QUESTION: {question}

DOCUMENT CONTEXT:
{context}

Please provide a direct answer based only on the given context. If the information is not available, say so clearly."""