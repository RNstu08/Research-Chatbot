# 📚 Research Paper AI Chatbot

This project implements a local Retrieval Augmented Generation (RAG) chatbot designed to answer questions about research papers (PDFs) that you upload. It leverages local language models (LLaMA 2 GGUF) and vector databases (ChromaDB) to provide contextually relevant answers based *only* on the content of the uploaded document, ensuring data privacy and offline capability.

## ✨ Features

* **PDF Upload:** Upload research papers in PDF format.
* **Local Processing:** Text extraction, chunking, embedding generation, and LLM inference all happen locally.
* **Vector Storage:** Uses ChromaDB to store document chunks and embeddings persistently.
* **RAG Pipeline:** Employs a Retrieval Augmented Generation pipeline using LangChain to ground LLM answers in the document's content.
* **Chat Interface:** Provides a user-friendly chat interface built with Streamlit.
* **History:** Remembers processed files and chat history per document using an SQLite database.
* **Configurable:** Key parameters (model paths, chunking settings) are managed via a configuration file.

## 📂 Project Structure

```
research_chatbot/
├── .venv/                         # Python virtual environment (created by user)
├── .git/                          # Git repository data (created by user)
├── app/                           # Core application logic and modules
│   ├── __init__.py
│   ├── data_processing/           # PDF parsing and text chunking
│   │   ├── __init__.py
│   │   ├── pdf_parser.py
│   │   └── text_chunker.py
│   ├── semantic_indexing/         # Embedding generation and vector store interaction
│   │   ├── __init__.py
│   │   ├── embedding_generator.py
│   │   └── vector_store.py
│   ├── retrieval_augmentation/    # LLM loading and RAG pipeline logic
│   │   ├── __init__.py
│   │   ├── llm_interface.py
│   │   └── rag_pipeline.py
│   ├── services/                  # History management
│   │   ├── __init__.py
│   │   └── history_manager.py
│   ├── ui/                        # Streamlit user interface
│   │   ├── __init__.py
│   │   └── streamlit_app.py
│   └── utils/                     # Common utilities (logging)
│       ├── __init__.py
│       └── logging_config.py
├── config/                        # Configuration files
│   ├── __init__.py
│   └── settings.py
├── data/                          # Persistent data (DBs, potentially uploads)
│   ├── chroma_db_store/           # ChromaDB persistent storage
│   └── research_bot.db            # SQLite database file (created on first run)
├── logs/                          # Log files
│   └── app.log                    # Main application log file (created on first run)
├── models/                        # Downloaded LLM model files (e.g., GGUF)
│   └── .gitkeep                   # Placeholder (user adds models here)
├── notebooks/                     # Jupyter notebooks for experimentation (optional)
│   └── 01_data_exploration.ipynb
├── tests/                         # Unit and integration tests (to be added)
│   ├── __init__.py
│   └── ...
├── .gitignore                     # Files ignored by Git
├── README.md                      # This file
├── requirements.txt               # Python package dependencies
└── run.py                         # Script for initial setup (DB, Logging)
```

## 🚀 Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository (If applicable):**
If you have this project in a Git repository, clone it:
```bash
git clone <your-repo-url>
cd research_chatbot
```
If you built it locally following the guide, just navigate to the `research_chatbot` directory.

**2. Prerequisites (Windows):**
* **Python:** Ensure Python 3.9+ is installed. Add it to your system PATH.
* **Git:** Ensure Git is installed ([git-scm.com](https://git-scm.com/)).
* **C++ Build Tools:** Required for `llama-cpp-python`. Install "Build Tools for Visual Studio" from the [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) page (select "Desktop development with C++" workload).
* **CMake:** Required for `llama-cpp-python`. Download from [cmake.org](https://cmake.org/download/) and ensure it's added to your system PATH during installation.

**3. Create and Activate Virtual Environment:**
Open PowerShell or Command Prompt in the `research_chatbot` directory.
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate
```
*(Your prompt should now start with `(.venv)`)*

**4. Install Dependencies:**
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt

# Ensure llama-cpp-python builds correctly (might need re-run after build tools install)
# Consider --force-reinstall --no-cache-dir if facing issues after installing build tools
pip install llama-cpp-python --upgrade
```
*Troubleshooting:* If `llama-cpp-python` fails, double-check C++ Build Tools and CMake installation and PATH configuration. You might need to restart your terminal.

**5. Download LLM Model:**
* Download a LLaMA 2 compatible model in **GGUF format**. Models from "TheBloke" on Hugging Face are recommended (e.g., `llama-2-7b-chat.Q4_K_M.gguf`).
* Place the downloaded `.gguf` file inside the `research_chatbot/models/` directory.

**6. Configure Model Path:**
* Open the `research_chatbot/config/settings.py` file.
* Update the `LLM_MODEL_PATH` variable to point to the exact filename of the model you downloaded. Example:
    ```python
    LLM_MODEL_PATH = LLM_MODEL_DIR / "llama-2-7b-chat.Q4_K_M.gguf"
    ```

## ▶️ Running the Application

1.  **Perform Initial Setup:**
    Run the `run.py` script once to initialize the database and logging. Make sure your virtual environment is active.
    ```powershell
    python run.py
    ```
    *(You should see setup messages in the console)*

2.  **Launch Streamlit App:**
    Start the Streamlit application using the following command:
    ```powershell
    streamlit run app/ui/streamlit_app.py
    ```

3.  **Access the App:**
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🛠️ Usage

1.  **Upload PDF:** Use the sidebar to upload a research paper PDF.
2.  **Process:** Click the "Process Uploaded PDF" button. Wait for the processing to complete (parsing, chunking, embedding, storing).
3.  **Select:** Choose the processed paper from the "Select Processed Paper" dropdown in the sidebar.
4.  **Chat:** Use the chat interface in the main area to ask questions about the selected paper. View sources used for answers in the expander below assistant messages.

## ⚙️ Configuration

Key parameters can be adjusted in `config/settings.py`:

* `LLM_MODEL_PATH`: Path to the downloaded GGUF model file.
* `N_CTX`: Context window size for the LLM.
* `MAX_TOKENS_LLM`: Max tokens the LLM should generate.
* `EMBEDDING_MODEL_NAME`: Name of the SentenceTransformer model from Hugging Face.
* `CHROMA_PERSIST_DIR`: Directory to store the ChromaDB vector database.
* `SQLITE_DB_PATH`: Path to the SQLite database for history.
* `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for text chunking.
* `LOG_LEVEL`, `LOG_FILE`: Logging configuration.

## 💻 Key Technologies

* **Python:** Core programming language.
* **LangChain:** Framework for building LLM applications (RAG pipeline, wrappers).
* **LLaMA 2 (via `llama-cpp-python`):** Local Large Language Model for generation.
* **SentenceTransformers (via `langchain_community`):** Embedding generation.
* **ChromaDB:** Local vector database for storing and retrieving embeddings.
* **PyMuPDF (`fitz`):** PDF text extraction.
* **Streamlit:** Web application framework for the UI.
* **SQLite:** Database for storing file processing and chat history.
```