
---

## Phase 0: Foundations, Advanced Concepts & Environment Setup

### 1. Revisiting Core Concepts (Essential for an Expert Understanding)

* **LLMs (Large Language Models - e.g., LLaMA 2):**
    * **What:** Neural networks trained on massive text datasets to understand, generate, and manipulate human language. They work by predicting the next token (word or sub-word) in a sequence.
    * **Why for our project:** LLaMA 2 is open-source, powerful, and can run locally. This gives us data privacy, no API costs, and control over the model. It's the "brain" that will synthesize answers.
    * **Key consideration:** LLMs have a "context window" – a limit on the amount of text (tokens) they can consider at once. This is a primary driver for needing chunking.

* **Transformers (The Architecture Behind LLMs):**
    * **What:** A specific neural network architecture that revolutionized NLP. Its key mechanism is "self-attention," which allows the model to weigh the importance of different words in the input text relative to each other, capturing long-range dependencies and context.
    * **Why:** Understanding this helps appreciate *how* LLMs can grasp nuances in language. Most modern LLMs and embedding models are transformer-based.

* **Tokens:**
    * **What:** LLMs don't see words directly; they see "tokens." Tokens can be whole words, parts of words (sub-words), or even individual characters, depending on the tokenizer used. The tokenizer converts raw text into a sequence of token IDs.
    * **Why:** Token limits define context windows and influence processing cost. The choice of tokenizer is tied to the LLM.

* **Vector Databases:** Specialized databases designed to store, manage, and efficiently search through high-dimensional vectors (like our text embeddings). Examples: ChromaDB, FAISS, Pinecone, Weaviate.

* **RAG (Retrieval Augmented Generation):** The core pattern of our application.
    * **Retriever:** This component is responsible for finding the most relevant information from your knowledge base (the research papers). It takes the user's query, turns it into an embedding, and uses a vector database to find the document chunks with the most similar embeddings.
        * *Key concept:* Similarity search (often cosine similarity) in vector space.
    * **Generator:** This is the LLM. It receives the user's query *and* the context retrieved by the Retriever.
    * **Orchestrator:** The logic that coordinates the Retriever and Generator. It ensures the query is processed, context is fetched, the prompt is correctly formatted for the LLM, and the answer is returned. **LangChain is a popular orchestrator**.
    * **Why RAG?** It grounds the LLM's answers in specific documents, significantly reducing "hallucinations" (made-up answers) and making responses factual and verifiable based on the provided source material. It's more efficient and targeted than fine-tuning an LLM for domain-specific knowledge.

### 2. Deep Dive: Sophisticated Text Chunking Strategies for Research Papers

Chunking is arguably one of the most critical steps for effective RAG. Poor chunking leads to poor retrieval, which leads to poor answers. Research papers present unique challenges: dense information, structured sections (abstract, intro, methods, results, conclusion, references), figures, tables, and citations.

* **a. Fixed-Size Chunking:**
    * **What:** Splitting text into chunks of exactly `N` characters or tokens, with an optional `M` characters/tokens of overlap.
    * **How (LangChain):** `CharacterTextSplitter` (can be configured to split by character count).
    * **Pros:** Simple to implement.
    * **Cons for Research Papers:** Very likely to cut sentences, paragraphs, or even crucial data points (like a result in a table description) mid-way. Semantic meaning is often lost. Overlap helps but doesn't fully solve it.
    * **Verdict:** Generally not recommended as a primary strategy for research papers due to their complex structure.

* **b. Recursive Character Splitting:**
    * **What:** Attempts to split text based on a hierarchy of separators. It tries the first separator (e.g., `\n\n` for paragraphs). If resulting chunks are still too large, it applies the next separator (e.g., `\n` for lines), then spaces, then characters, until chunks are within the desired size.
    * **How (LangChain):** `RecursiveCharacterTextSplitter`. This is often the default starting point.
    * **Pros:** Better than fixed-size as it tries to respect natural boundaries (paragraphs, lines). Good balance of simplicity and effectiveness.
    * **Cons for Research Papers:** While better, it might still not fully understand the logical structure of a paper (e.g., differentiating a heading from a paragraph, or a list item). It's structure-agnostic beyond the characters provided.
    * **Verdict:** A good baseline, but we can do better for research papers.

* **c. Sentence Splitting (NLP-based):**
    * **What:** Uses NLP libraries (`nltk`, `spaCy`) to accurately detect sentence boundaries. Chunks can then be formed by grouping one or more sentences.
    * **How (LangChain):** `NLTKTextSplitter`, `SpacyTextSplitter`. Or custom logic using these libraries.
    * **Pros:** Ensures full sentences are kept together, preserving immediate semantic units.
    * **Cons for Research Papers:**
        * Individual sentences might be too short for effective embedding (lacking broader context).
        * Grouping sentences requires a strategy (e.g., group up to `X` tokens).
        * Research paper sentences can be very long and complex.
    * **Verdict:** Useful if sentence integrity is paramount, but often needs a secondary grouping strategy.

* **d. NLP-Enhanced / Document-Structure-Aware Chunking (More Advanced):**
    * **What:** Attempts to parse the document based on its logical structure.
        * **Markdown/HTML:** If the PDF can be reliably converted to Markdown or HTML (some tools attempt this), you can split based on headers (`#`, `##`, `<h1>`, `<h2>`), lists, tables, etc. LangChain has `MarkdownHeaderTextSplitter`.
        * **LaTeX-Aware:** For papers sourced as LaTeX, parsing the LaTeX structure (sections, subsections, itemize, tables) can yield highly coherent chunks. This is specialized and requires LaTeX parsing libraries (e.g., `pylatexenc`).
        * **Layout-Aware (PDF-specific):** Tools like `Nougat` (Meta AI), `Marker`, or commercial solutions (e.g., some features in Adobe's APIs, Amazon Textract's layout analysis) try to understand the visual layout (columns, headers, paragraphs, tables, figures) directly from the PDF. These can be very powerful but are more complex to integrate. `unstructured.io` library also offers layout-aware parsing.
    * **Pros:** Potentially the most meaningful chunks as they align with the author's intended structure. Can separate out tables/figures for specific processing.
    * **Cons:** Conversion to structured formats (MD, HTML) from PDF can be imperfect. True layout-aware models can be computationally heavier or require specific setups. LaTeX source is not always available.
    * **Verdict:** Ideal if feasible. For our project, we might start with recursive and then explore if a reliable PDF-to-Markdown conversion step can be added to leverage `MarkdownHeaderTextSplitter`.

* **e. Semantic Chunking (Embedding-based):**
    * **What:** This approach splits text based on semantic similarity rather than fixed sizes or syntactic rules.
        1.  Break text into small initial units (e.g., sentences).
        2.  Generate embeddings for each unit.
        3.  Iterate through the units: if the semantic similarity (cosine similarity between embeddings) of adjacent units drops significantly below a threshold, it signifies a topic change, and a new chunk is started. Alternatively, group sentences into chunks as long as their combined embedding stays coherent or until a certain size is reached.
    * **How:** Requires custom logic or specialized libraries (e.g., `semantic-text-splitter` in Python, or LlamaIndex's `SemanticSplitterNodeParser`).
        ```python
        # Conceptual example for semantic chunking logic
        # from sentence_transformers import SentenceTransformer
        # from sklearn.metrics.pairwise import cosine_similarity
        # sentences = [...] # list of sentences from the document
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embeddings = model.encode(sentences)
        #
        # chunks = []
        # current_chunk_sentences = [sentences[0]]
        # for i in range(len(sentences) - 1):
        #     similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0]
        #     if similarity < threshold: # e.g., threshold = 0.7 or 0.8
        #         chunks.append(" ".join(current_chunk_sentences))
        #         current_chunk_sentences = [sentences[i+1]]
        #     else:
        #         if len(" ".join(current_chunk_sentences) + " " + sentences[i+1]) < MAX_CHUNK_SIZE:
        #             current_chunk_sentences.append(sentences[i+1])
        #         else: # Chunk is getting too big
        #             chunks.append(" ".join(current_chunk_sentences))
        #             current_chunk_sentences = [sentences[i+1]]
        # if current_chunk_sentences:
        #     chunks.append(" ".join(current_chunk_sentences))
        ```
    * **Pros:** Creates chunks that are inherently semantically coherent. Can adapt to varying content density.
    * **Cons:** More computationally intensive upfront due to embedding generation for all small units. Threshold tuning for similarity can be an iterative process.
    * **Verdict:** High-quality results, especially for nuanced text. A good advanced option to consider after establishing a baseline. The `semchunk` library mentioned in the search results also aims for this.

**Considerations Specific to Research Papers:**

* **Abstract & Conclusion:** These are often dense summaries. They might benefit from being treated as distinct chunks or given special weight.
* **Introduction & Methods:** These sections often contain critical context.
* **Results & Discussion:** Core findings are here. Tables and figures within these sections are vital.
* **Tables & Figures:** Often, text extractors struggle with tables. Figures are usually images.
    * **Strategy:** Extract text as best as possible. For tables, specialized table extraction tools (e.g., `camelot-py`, or PDF parsers that identify table structures) might be needed. For figures, image-to-text (OCR, or caption-based understanding if available) or multimodal models (like GPT-4V, LLaVA - more advanced) would be needed. For our initial scope, we'll focus on text, but it's crucial to acknowledge this limitation. Consider chunking captions separately and linking them to figure/table descriptions.
* **Citations:** Usually, you want to preserve them with the text they refer to, but they can also be noisy. Some strategies might try to normalize or remove them during a cleaning step if they aren't central to the Q&A.
* **References/Bibliography:** Usually less critical for Q&A about the *content* of the paper itself, unless the question is *about* the references. Could be a separate, lower-priority set of chunks or excluded.

**Choosing the "Best" Strategy for Research Papers (Iterative Approach):**

1.  **Start with `RecursiveCharacterTextSplitter` (from LangChain):** It's a good, robust baseline. Configure it with separators like `\n\n` (paragraph), `\n` (line), and then sentence-ending punctuation if possible, or default to spaces.
    * **Chunk Size:** Aim for a size that balances context and specificity. For research papers, 512-1024 tokens is a common range. This depends on your embedding model's ideal input and the LLM's context window.
    * **Overlap:** Essential. 10-20% of the chunk size (e.g., 100-200 tokens for a 1000-token chunk) helps maintain context across chunk boundaries.

2.  **Experiment with Document Structure Awareness:**
    * If you can reliably convert PDFs to Markdown, try `MarkdownHeaderTextSplitter`. This can give much better results by chunking based on sections and subsections. This would be my recommended "next step" after recursive splitting. Many modern PDF parsers like `unstructured` attempt to provide this structure.

3.  **Evaluate and Iterate:** The "best" strategy is empirical.
    * Ask representative questions about your papers.
    * Examine the retrieved chunks: Are they relevant? Do they contain the full answer? Are they too broad or too narrow?
    * Adjust chunk size, overlap, and strategy based on these observations.

4.  **Consider Semantic Chunking as an Advanced Optimization:** If the above methods still struggle with specific types of queries or document sections, investing in semantic chunking can provide a significant quality boost.

For this project, we will implement `RecursiveCharacterTextSplitter` first, making it configurable. We will structure the code in `app/data_processing/text_chunker.py` such that you could later add and switch to other strategies like a Markdown-based one or a semantic one.

### 3. Architectural Blueprint Review

**Brief Architectural Overview**

Our GenAI Research Chatbot will operate as follows:

1.  **User Interface (Streamlit):** The user uploads a research paper (PDF) and types questions.
2.  **Data Processing Pipeline:**
    * The uploaded PDF is parsed to extract raw text.
    * This text is cleaned and then intelligently divided into smaller, semantically meaningful **chunks**.
3.  **Semantic Indexing Pipeline:**
    * Each text chunk is converted into a numerical representation (an **embedding**) using a specialized model (e.g., SentenceTransformer).
    * These embeddings, along with their corresponding text chunks and metadata, are stored in a **Vector Database** (ChromaDB). This database allows for efficient similarity searches.
4.  **Retrieval Augmented Generation (RAG) Pipeline:**
    * When the user asks a question, the question is also converted into an embedding.
    * The Vector Database is queried to find the text chunks whose embeddings are most similar to the question's embedding. These are the most relevant pieces of information from the paper.
    * A **prompt** is constructed, combining the user's original question with the retrieved relevant text chunks.
    * This augmented prompt is fed to a **Large Language Model (LLM)** (LLaMA 2, running locally).
    * The LLM generates an answer based *specifically* on the provided context from the paper, rather than just its general knowledge.
5.  **History Management:** Details of processed papers and chat interactions are stored persistently (e.g., in an SQLite database) for future reference.
6.  **Configuration & Logging:** All operations are guided by a central configuration, and important events are logged for monitoring and debugging.

This modular design ensures that each part of the system can be developed, tested, and improved independently.

## `template.py` for Project Scaffolding (Windows)

Create a file named `template.py` in the directory where you want your main project folder (`research_chatbot`) to be created.

```python
# template.py
import os
from pathlib import Path

PROJECT_NAME = "research_chatbot"

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
```

**How to use `template.py` on Windows:**

1.  Save the code above as `template.py` in a directory of your choice (e.g., `C:\Users\YourUser\Projects`).
2.  Open Command Prompt (CMD) or PowerShell.
3.  Navigate to that directory: `cd C:\Users\YourUser\Projects`
4.  Run the script: `python template.py`

This will create the `research_chatbot` folder and all its subdirectories and initial files within `C:\Users\YourUser\Projects`. You can then `cd research_chatbot` to start working inside your newly structured project.


Let's visualize the flow (textually):

```
USER ----> [STREAMLIT UI] ----> [APPLICATION_SERVICES]
             ^      |                     |
             |      | (Query)             | (Orchestrates)
             |      |                     V
             |      +-------------- [RETRIEVAL_AUGMENTATION_PIPELINE]
             |                            |      ^
(Answer/Sources)                          |      | (Context)
             |                            | (LLM Query + Context)
             |                            V      |
             |                      [LLM (Local LLaMA 2)]
             |                            |
             +-------------- [SEMANTIC_INDEXING_PIPELINE (Vector DB)]
                                          ^
                                          | (Chunks & Embeddings)
                                          |
                                 [DATA_PROCESSING_PIPELINE (PDF -> Chunks)]
                                          ^
                                          | (PDF File)
                                 [FILE UPLOAD/SELECTION]
```
**Component Responsibilities:**

* **`data_processing_pipeline`:** Takes raw PDFs, extracts text, cleans it, and chunks it intelligently.
* **`semantic_indexing_pipeline`:** Takes text chunks, generates embeddings, and stores them in ChromaDB along with metadata.
* **`retrieval_augmentation_pipeline`:** Takes a user query, finds relevant chunks from ChromaDB, prepares a prompt with context, and gets an answer from the LLM.
* **`application_services`:** Manages the overall user interaction flow, state, history, and coordinates the other pipelines. Contains the Streamlit UI.
* **LLM (Local LLaMA 2):** The core generation engine.
* **Configuration (`config.py`):** Stores paths, model names, parameters.
* **Logging:** Records important events, errors, and information for debugging and monitoring.

### 4. Professional Development Environment (Windows Focus)

* **Python & Virtual Environments:**
    * **Why `venv`?** Isolates project dependencies, preventing conflicts between projects. Standard Python practice.
    * **Commands (in your project root `research_chatbot/` after `template.py` has run):**
        1.  `cd research_chatbot`
        2.  `python -m venv .venv` (Creates the virtual environment folder)
        3.  `.\.venv\Scripts\activate` (Activates it in CMD/PowerShell. Your prompt should change.)

* **Version Control: Git**
    * **Why?** Tracks code changes, allows rollbacks, collaboration. Indispensable.
    * **Commands (in `research_chatbot/` root, after activating `venv`):**
        1.  `git init` (Initializes a new Git repository)
        2.  The `.gitignore` file was created by `template.py`. Review it.
        3.  `git add .` (Stages all files for the first commit)
        4.  `git commit -m "Initial project structure from template"` (Commits the initial structure)

* **IDE: VS Code**
    * Open the `research_chatbot` folder in VS Code (`File > Open Folder...`).
    * **Extensions:**
        * "Python" (Microsoft) - Essential.
        * "Pylance" (Microsoft) - For rich language support (autocompletion, type checking).
        * "GitLens" (GitKraken) - Supercharges Git integration.
        * Optional: "Rainbow CSV", "Todo Tree".
    * **Select Python Interpreter:**
        * VS Code should auto-detect your `.venv`. If not, press `Ctrl+Shift+P`, type "Python: Select Interpreter", and choose the one from `.venv\Scripts\python.exe`.
        * Open a new terminal in VS Code (`Terminal > New Terminal`). It should automatically activate the `.venv`.

* **Dependency Management (`requirements.txt`):**
    * The `template.py` created an empty `requirements.txt`.
    * As we install packages (e.g., `pip install streamlit langchain`), we'll update this file.
    * **Command to update (after installing packages):** `pip freeze > requirements.txt`
    * **Command to install from file (e.g., on a new setup):** `pip install -r requirements.txt`

### 5. Project Directory Structure Review

The `template.py` script has already created the structure. Familiarize yourself with the layout:
* `app/`: Core application logic.
    * `data_processing/`: PDF parsing, text chunking.
    * `semantic_indexing/`: Embedding generation, vector store interaction.
    * `retrieval_augmentation/`: LLM interface, RAG pipeline.
    * `services/`: Higher-level logic, history management.
    * `ui/`: Streamlit UI.
    * `utils/`: Common utilities like logging.
* `config/`: Application settings (`settings.py`).
* `data/`: Persistent storage (ChromaDB, SQLite DB, uploaded PDFs).
* `models/`: For downloaded LLM files.
* `notebooks/`: For experiments.
* `tests/`: For automated tests.
* `logs/`: For log files.
* Root files: `run.py` (entry point), `requirements.txt`, `README.md`, `.gitignore`.

This modular structure promotes:
* **Separation of Concerns:** Each module has a clear responsibility.
* **Testability:** Easier to write unit tests for individual modules.
* **Maintainability:** Changes in one module are less likely to break others.
* **Scalability:** Easier to add new features or modify existing ones.

This structure promotes modularity and separation of concerns.

```
research_chatbot/
├── .venv/                         # Python virtual environment
├── .git/                          # Git repository data
├── .vscode/                       # VS Code workspace settings (optional, e.g., launch.json)
|
├── app/                           # Core application logic and modules
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py          # PDF text extraction
│   │   └── text_chunker.py        # Text chunking strategies
│   ├── semantic_indexing/
│   │   ├── __init__.py
│   │   ├── embedding_generator.py # Embedding model utilities
│   │   └── vector_store.py        # ChromaDB interaction
│   ├── retrieval_augmentation/
│   │   ├── __init__.py
│   │   ├── llm_interface.py       # LLaMA 2 GGUF interaction
│   │   └── rag_pipeline.py        # RAG orchestration
│   ├── services/                  # Higher-level services, history management
│   │   ├── __init__.py
│   │   └── history_manager.py     # SQLite for file/chat history
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py       # Streamlit UI code
│   └── utils/                     # Common utilities (e.g., logging setup)
│       ├── __init__.py
│       └── logging_config.py
|
├── config/
│   ├── __init__.py
│   └── settings.py                # Application configuration (paths, model names)
|
├── data/                          # For persistent data like SQLite DB, ChromaDB persistence
│   └── chroma_db_store/           # ChromaDB persistent storage path
│   └── research_bot.db            # SQLite database file
|
├── models/                        # For downloaded GGUF model files (add to .gitignore if large)
│   └── placeholder.txt            # Create this dir, add actual models later
|
├── notebooks/                     # Jupyter notebooks for experimentation (optional)
│   └── exploration.ipynb
|
├── tests/                         # For unit and integration tests
│   ├── __init__.py
│   ├── data_processing/
│   │   └── test_text_chunker.py
│   └── ... (other test modules mirroring app structure)
|
├── .gitignore                     # Specifies intentionally untracked files that Git should ignore
├── Dockerfile                     # For containerizing the application (Phase 5)
├── LICENSE                        # Your project's license (e.g., MIT, Apache 2.0)
├── README.md                      # Project description, setup, and usage instructions
├── requirements.txt               # Python package dependencies
└── run.py                         # Main entry point to launch the Streamlit application
```

```

**`.gitignore` content (create this file in `research_chatbot/` root):**
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/version info into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# VS Code
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
*.code-workspace

# Local secrets and sensitive data
secrets.yml
*.local
*.key
*.pem

# Models (if large and not versioned)
models/*.gguf
models/*.bin

# Databases
data/*.db
data/*.sqlite
data/*.sqlite3
data/chroma_db_store/ # If Chroma is writing a lot here

# Logs
logs/
*.log

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
```
After creating `.gitignore`, add and commit it:
```bash
git add .gitignore
git commit -m "feat: Setup comprehensive .gitignore"
git add . # Add all newly created files and directories
git commit -m "feat: Initialize project structure and core module files"
```
### 6. Configuration Management (`config/settings.py`)

* The `template.py` created `config/settings.py` with placeholders.
* **Why?** Centralizes all configurable parameters (file paths, model names, API keys if any, thresholds). Avoids hardcoding values in your application logic. Makes it easy to switch models or adjust parameters without code changes.
* We will import from `config.settings` in our modules.
    * Example: `from config.settings import LLM_MODEL_PATH`
* **Action:** You will need to download a LLaMA 2 GGUF model (e.g., from Hugging Face, TheBloke's repositories are popular for GGUF versions) and update `LLM_MODEL_PATH` in `config/settings.py` with the actual model filename and place the model in the `models/` directory. For example, if you download `llama-2-7b-chat.Q4_K_M.gguf`, you'd set `LLM_MODEL_PATH = LLM_MODEL_DIR / "llama-2-7b-chat.Q4_K_M.gguf"`.

Hardcoding paths, model names, or parameters directly in your application logic is bad practice. A config file makes it easy to change these without touching the core code.

**`config/settings.py`:**
```python
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent # This will be research_chatbot/

# LLM Configuration
LLM_MODEL_DIR = BASE_DIR / "models"
# Example: LLM_MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf" # You'll need to download this
LLM_MODEL_PATH = LLM_MODEL_DIR / "your_llm_model_name.gguf" # Placeholder, replace with actual name
N_CTX = 4096  # Context window size for LLaMA
MAX_TOKENS_LLM = 512 # Max tokens LLaMA should generate for an answer

# Embedding Model Configuration
# Using a SentenceTransformer model. See https://www.sbert.net/docs/pretrained_models.html
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Alternatives: "BAAI/bge-small-en-v1.5", "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Vector Database Configuration
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db_store"
CHROMA_COLLECTION_NAME_PREFIX = "research_doc" # We might create a collection per document

# SQLite Database for History
SQLITE_DB_PATH = BASE_DIR / "data" / "research_bot.db"

# Text Chunking Configuration
CHUNK_SIZE = 768  # Target size of text chunks in tokens (approx)
CHUNK_OVERLAP = 100 # Number of tokens to overlap between chunks

# Logging Configuration
LOG_LEVEL = "INFO" # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE = BASE_DIR / "logs" / "app.log" # Make sure 'logs' directory exists or create it

# Ensure logs directory exists
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Add other configurations as needed:
# API_HOST = "0.0.0.0"
# API_PORT = 8000
# STREAMLIT_SERVER_PORT = 8501
```
*Action: Create the `logs` directory: `mkdir logs` in the project root.*
*Then `git add logs` (if you want to track an empty dir, or add a `.gitkeep` file inside it then add that).*
*Or better, let the logger setup create it if it doesn't exist (which `settings.py` now does for `LOGS_DIR`).*

### 7. Logging Best Practices (`app/utils/logging_config.py`)

* `template.py` created a placeholder for `app/utils/logging_config.py`. Let's fill it.
* **Why?** Logging provides insights into your application's behavior, helps diagnose errors, and tracks important events. It's crucial for debugging and monitoring.

**`app/utils/logging_config.py`:**
```python
import logging
import sys
from logging.handlers import RotatingFileHandler
from config import settings # Assuming your project root is in PYTHONPATH or you run from root

def setup_logging():
    """Sets up logging configuration for the application."""
    
    # Ensure the logs directory exists (settings.py should also do this, but good to double-check)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Define log format
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Root logger configuration (basic setup)
    # logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a root logger instance
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(settings.LOG_LEVEL.upper()) # Set its level

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler (RotatinngFileHandler for better log management)
    file_handler = RotatingFileHandler(
        settings.LOG_FILE, 
        maxBytes=10*1024*1024,  # 10 MB per file
        backupCount=5  # Keep 5 backup files
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    # Optionally silence overly verbose loggers from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("another_library").setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully.")

def get_logger(name: str) -> logging.Logger:
    """
    Utility function to get a logger instance for a specific module.
    This helps in identifying the source of log messages.
    """
    return logging.getLogger(name)

# Example of how to use in other modules:
# from app.utils.logging_config import get_logger
# logger = get_logger(__name__) # __name__ gives the current module's name
#
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")
```

**To use this:** In `run.py` (our main entry point), we'll call `setup_logging()` once at the beginning.

This completes a very thorough Phase 0. We have:
* A project structure generated by `template.py`.
* A Git repository initialized.
* A Python virtual environment activated and VS Code configured.
* A deep understanding of core concepts, especially various chunking strategies and their relevance to research papers.
* Foundations for configuration and logging.


