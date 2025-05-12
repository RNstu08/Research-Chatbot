# template.py
import os
from pathlib import Path

PROJECT_NAME = ""

# Define the directory structure as a list of paths relative to the project root
directories = [
    # Main application structure
    "app",
    "app/data_processing",
    "app/semantic_indexing",
    "app/retrieval_augmentation",
    "app/services",
    "app/ui",
    "app/utils",
    # Configuration
    "config",
    # Data storage (Gitignored by default for dynamic data)
    "data",
    "data/chroma_db_store",  # For ChromaDB persistence
    "data/uploaded_pdfs",   # Optional: for storing uploaded files temporarily
    # LLM models (Gitignored if large)
    "models",
    # Jupyter notebooks for experimentation
    "notebooks",
    # Tests
    "tests",
    "tests/data_processing",
    "tests/semantic_indexing",
    "tests/retrieval_augmentation",
    "tests/services",
    # Logs
    "logs"
]

# Define files to create (mostly __init__.py to make Python packages)
# Path objects are used for easier file creation.
# Content can be added for some files if needed.
files_to_create = {
    # Main project files
    Path(PROJECT_NAME) / "run.py": "",
    Path(PROJECT_NAME) / "README.md": f"# {PROJECT_NAME}\n\nProject description.",
    Path(PROJECT_NAME) / "requirements.txt": "# Python dependencies\n",
    Path(PROJECT_NAME) / ".gitignore": (
        "# Byte-compiled / optimized / DLL files\n__pycache__/\n*.py[cod]\n*$py.class\n\n"
        "# C extensions\n*.so\n\n# Distribution / packaging\n.Python\nbuild/\ndevelop-eggs/\ndist/\n"
        "downloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\npip-wheel-metadata/\n"
        "share/python-wheels/\n*.egg-info/\n.installed.cfg\n*.egg\nMANIFEST\n\n# PyInstaller\n*.manifest\n"
        "*.spec\n\n# Installer logs\npip-log.txt\npip-delete-this-directory.txt\n\n# Unit test / coverage reports\n"
        "htmlcov/\n.tox/\n.nox/\n.coverage\n.coverage.*\n.cache\nnosetests.xml\ncoverage.xml\n*.cover\n"
        "*.py,cover\n.hypothesis/\n.pytest_cache/\n\n# Environments\n.env\n.venv\nenv/\nvenv/\nENV/\n"
        "env.bak/\nvenv.bak/\n\n# VS Code\n.vscode/*\n!.vscode/settings.json\n!.vscode/tasks.json\n"
        "!.vscode/launch.json\n!.vscode/extensions.json\n*.code-workspace\n\n# Local secrets\nsecrets.yml\n"
        "*.local\n*.key\n*.pem\n\n# Models (if large and not versioned)\nmodels/*.gguf\nmodels/*.bin\n\n"
        "# Databases & Data\ndata/\nlogs/\n\n# OS generated files\n.DS_Store\n.DS_Store?\n._*\n.Spotlight-V100\n"
        ".Trashes\nehthumbs.db\nThumbs.db\n"
    ),
    # App modules
    Path(PROJECT_NAME) / "app" / "__init__.py": "",
    Path(PROJECT_NAME) / "app/data_processing/__init__.py": "",
    Path(PROJECT_NAME) / "app/data_processing/pdf_parser.py": "# PDF text extraction logic here\n",
    Path(PROJECT_NAME) / "app/data_processing/text_chunker.py": "# Text chunking strategies here\n",
    Path(PROJECT_NAME) / "app/semantic_indexing/__init__.py": "",
    Path(PROJECT_NAME) / "app/semantic_indexing/embedding_generator.py": "# Embedding model utilities\n",
    Path(PROJECT_NAME) / "app/semantic_indexing/vector_store.py": "# VectorDB (ChromaDB) interaction\n",
    Path(PROJECT_NAME) / "app/retrieval_augmentation/__init__.py": "",
    Path(PROJECT_NAME) / "app/retrieval_augmentation/llm_interface.py": "# LLM (LLaMA 2 GGUF) interaction\n",
    Path(PROJECT_NAME) / "app/retrieval_augmentation/rag_pipeline.py": "# RAG orchestration logic\n",
    Path(PROJECT_NAME) / "app/services/__init__.py": "",
    Path(PROJECT_NAME) / "app/services/history_manager.py": "# SQLite for file/chat history\n",
    Path(PROJECT_NAME) / "app/ui/__init__.py": "",
    Path(PROJECT_NAME) / "app/ui/streamlit_app.py": "# Streamlit UI code\n",
    Path(PROJECT_NAME) / "app/utils/__init__.py": "",
    Path(PROJECT_NAME) / "app/utils/logging_config.py": "# Logging setup\n",
    # Config
    Path(PROJECT_NAME) / "config/__init__.py": "",
    Path(PROJECT_NAME) / "config/settings.py": (
        "import os\nfrom pathlib import Path\n\n"
        "BASE_DIR = Path(__file__).resolve().parent.parent\n"
        "LOGS_DIR = BASE_DIR / 'logs'\nLOGS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "# LLM Configuration\nLLM_MODEL_DIR = BASE_DIR / 'models'\n"
        "# LLM_MODEL_NAME = 'llama-2-7b-chat.Q4_K_M.gguf' # Example\n"
        "LLM_MODEL_PATH = LLM_MODEL_DIR / 'your_model_name.gguf' # To be replaced\n"
        "N_CTX = 4096\nMAX_TOKENS_LLM = 512\n\n"
        "# Embedding Model\nEMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'\n\n"
        "# Vector DB\nCHROMA_PERSIST_DIR = BASE_DIR / 'data' / 'chroma_db_store'\n"
        "CHROMA_COLLECTION_NAME_PREFIX = 'research_doc'\n\n"
        "# SQLite DB\nSQLITE_DB_PATH = BASE_DIR / 'data' / 'research_bot.db'\n\n"
        "# Chunking\nCHUNK_SIZE = 768\nCHUNK_OVERLAP = 100\n\n"
        "# Logging\nLOG_LEVEL = 'INFO'\nLOG_FILE = LOGS_DIR / 'app.log'\n"
    ),
    # Data placeholders
    Path(PROJECT_NAME) / "data/chroma_db_store/.gitkeep": "", # To keep the dir in git if empty
    Path(PROJECT_NAME) / "data/uploaded_pdfs/.gitkeep": "",
    # Models placeholder
    Path(PROJECT_NAME) / "models/.gitkeep": "",
    # Notebooks
    Path(PROJECT_NAME) / "notebooks/01_data_exploration.ipynb": "", # Placeholder for a notebook
    # Tests
    Path(PROJECT_NAME) / "tests/__init__.py": "",
    Path(PROJECT_NAME) / "tests/data_processing/__init__.py": "",
    Path(PROJECT_NAME) / "tests/data_processing/test_text_chunker.py": "# Tests for text chunker\n",
    Path(PROJECT_NAME) / "tests/semantic_indexing/__init__.py": "",
    Path(PROJECT_NAME) / "tests/retrieval_augmentation/__init__.py": "",
    Path(PROJECT_NAME) / "tests/services/__init__.py": "",
    # Logs placeholder
    Path(PROJECT_NAME) / "logs/.gitkeep": "",
}

def create_project_structure(base_path):
    """Creates the project directory structure and initial files."""
    project_root = Path(base_path) / PROJECT_NAME
    project_root.mkdir(exist_ok=True)
    print(f"Created project root: {project_root}")

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    for file_path_relative, content in files_to_create.items():
        # The file_path_relative already includes PROJECT_NAME, so we join with base_path
        file_path_absolute = Path(base_path) / file_path_relative
        # Ensure parent directory exists, though previous loop should handle most
        file_path_absolute.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path_absolute, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path_absolute}")

if __name__ == "__main__":
    # Run this script from the directory where you want 'research_chatbot' folder to be created
    current_working_directory = Path.cwd()
    create_project_structure(current_working_directory)
    print(f"\nProject '{PROJECT_NAME}' structure created successfully in '{current_working_directory}'.")
    print("Next steps:")
    print(f"1. cd {PROJECT_NAME}")
    print("2. Initialize Git: git init")
    print("3. Create and activate a virtual environment (e.g., python -m venv .venv && .\\.venv\\Scripts\\activate)")
    print("4. Install dependencies from requirements.txt (once populated): pip install -r requirements.txt")