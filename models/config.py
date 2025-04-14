# config.py
import os

# --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Choose the collection containing the research documents
DEFAULT_COLLECTION_NAME = "blog_collection" # Or "labour_collection", "san_francisco_budget_2023"

# --- Ollama Configuration ---
# OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_BASE_URL = "http://10.200.2.34:32411" # For remote access
# LLM_MODEL = "llama3.2:latest" # Model for generation/reasoning steps
LLM_MODEL = "llama3.2:3b-instruct-fp16" # Model for generation/reasoning steps
EMBEDDING_MODEL = "mxbai-embed-large:335m" # Model for vector store embeddings

# --- Embedding Model Configuration (Alternative: HuggingFace) ---
USE_HUGGINGFACE_EMBEDDING = False
HUGGINGFACE_EMBEDDING_MODEL = "avsolatorio/GIST-small-Embedding-v0"

# --- LlamaIndex Settings ---
REQUEST_TIMEOUT = 600.0 # Increased timeout for workflow steps
SIMILARITY_TOP_K = 3 # For the RAG tool

# --- Workflow Configuration ---
MAX_REVIEWS = 2 # Max refinement loops (original was 3, including initial write)
MAX_QUESTIONS_PER_BATCH = 8 # Limit questions generated initially
MAX_QUESTIONS_PER_REVIEW = 4 # Limit questions generated during review

# --- File Paths ---
PROMPT_FILE_PATH = "prompts.json"
# Example Data Paths (Uncomment if indexing is part of the app)
# DATA_PATH_BLOG = "./data"
# DATA_PATH_LABOUR = "/home/tuandatebayo/WorkSpace/law/labour"
# DATA_PATH_BUDGET = "./" # Assuming san_francisco_budget_2023.pdf is here