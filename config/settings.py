import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_MODEL_DIR = BASE_DIR / 'models'
# LLM_MODEL_NAME = 'llama-2-7b-chat.Q4_K_M.gguf' # Example
LLM_MODEL_PATH = LLM_MODEL_DIR / 'llama-2-7b-chat.Q4_K_M.gguf' # To be replaced
N_CTX = 4096
MAX_TOKENS_LLM = 512

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Vector DB
CHROMA_PERSIST_DIR = BASE_DIR / 'data' / 'chroma_db_store'
CHROMA_COLLECTION_NAME_PREFIX = 'research_doc'

# SQLite DB
SQLITE_DB_PATH = BASE_DIR / 'data' / 'research_bot.db'

# Chunking
CHUNK_SIZE = 768
CHUNK_OVERLAP = 100

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = LOGS_DIR / 'app.log'
