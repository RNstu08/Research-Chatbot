**Phase 4: Application Layer**.

This phase is where we build the user-facing part of our application using Streamlit and set up a system to remember processed files and chat conversations using SQLite.

We'll break this into three parts:
* **Part A:** History Management (`app/services/history_manager.py`) - The database interaction.
* **Part B:** Streamlit User Interface (`app/ui/streamlit_app.py`) - The web app itself.
* **Part C:** Main Application Runner (`run.py`) - The script to start everything.

### Step 1: Install Streamlit Library

If `streamlit` wasn't pulled in as a dependency by other `langchain` components, let's install it explicitly.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run:
    ```powershell
    pip install streamlit
    ```
2.  **Observe:** Pip will install Streamlit and its dependencies.

---

### Part A: History Management (`app/services/history_manager.py`)

This module will handle all interactions with an SQLite database to store and retrieve information about processed PDF files and their associated chat messages.

1.  **Action:** Open the file `app/services/history_manager.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/services/history_manager.py
    import sqlite3
    from typing import List, Tuple, Optional, Dict, Any
    import json
    from datetime import datetime

    from config import settings
    from app.utils.logging_config import get_logger

    logger = get_logger(__name__)

    DB_PATH = settings.SQLITE_DB_PATH

    def _get_db_connection() -> sqlite3.Connection:
        """Establishes and returns a connection to the SQLite database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row # Access columns by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database at {DB_PATH}: {e}", exc_info=True)
            raise

    def init_db():
        """Initializes the database schema if it doesn't already exist."""
        logger.info(f"Initializing database at {DB_PATH} if not already set up.")
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Table for processed files
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL UNIQUE,
                        file_hash TEXT NOT NULL UNIQUE, /* SHA256 hash of file content */
                        collection_name TEXT NOT NULL UNIQUE, /* ChromaDB collection name */
                        total_chunks INTEGER,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.debug("Table 'processed_files' checked/created.")

                # Table for chat messages
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id INTEGER NOT NULL, /* Foreign key to processed_files */
                        role TEXT NOT NULL, /* 'user' or 'assistant' */
                        content TEXT NOT NULL,
                        sources_json TEXT, /* JSON string of source document metadata */
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (file_id) REFERENCES processed_files (id)
                    )
                """)
                logger.debug("Table 'chat_messages' checked/created.")
                
                conn.commit()
                logger.info("Database initialization complete.")
        except sqlite3.Error as e:
            logger.error(f"Error during database initialization: {e}", exc_info=True)
            # It's critical if DB can't be initialized, so re-raise
            raise RuntimeError("Failed to initialize the application database.") from e

    def add_processed_file(file_name: str, file_hash: str, collection_name: str, total_chunks: int) -> Optional[int]:
        """Adds a record of a processed file to the database."""
        logger.info(f"Adding processed file to DB: {file_name}, hash: {file_hash[:8]}..., collection: {collection_name}")
        sql = """
            INSERT INTO processed_files (file_name, file_hash, collection_name, total_chunks)
            VALUES (?, ?, ?, ?)
        """
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_name, file_hash, collection_name, total_chunks))
                conn.commit()
                file_id = cursor.lastrowid
                logger.info(f"File '{file_name}' added to DB with ID: {file_id}")
                return file_id
        except sqlite3.IntegrityError: # Handles UNIQUE constraint violations
            logger.warning(f"File '{file_name}' or hash '{file_hash[:8]}' or collection '{collection_name}' likely already exists in DB.")
            # Retrieve the existing file_id if it already exists based on hash or name
            return get_processed_file_id_by_hash(file_hash) or get_processed_file_id_by_name(file_name)
        except sqlite3.Error as e:
            logger.error(f"Error adding processed file '{file_name}' to DB: {e}", exc_info=True)
            return None
            
    def get_processed_file_id_by_hash(file_hash: str) -> Optional[int]:
        """Retrieves the ID of a processed file by its hash."""
        sql = "SELECT id FROM processed_files WHERE file_hash = ?"
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_hash,))
                row = cursor.fetchone()
                return row['id'] if row else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching file by hash '{file_hash[:8]}': {e}", exc_info=True)
            return None

    def get_processed_file_id_by_name(file_name: str) -> Optional[int]:
        """Retrieves the ID of a processed file by its name."""
        sql = "SELECT id FROM processed_files WHERE file_name = ?"
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_name,))
                row = cursor.fetchone()
                return row['id'] if row else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching file by name '{file_name}': {e}", exc_info=True)
            return None


    def get_all_processed_files() -> List[Dict[str, Any]]:
        """Retrieves all processed files from the database."""
        logger.debug("Fetching all processed files from DB.")
        sql = "SELECT id, file_name, collection_name, processed_at, total_chunks FROM processed_files ORDER BY processed_at DESC"
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                # Convert rows to dictionaries for easier use
                files = [dict(row) for row in cursor.fetchall()]
                logger.debug(f"Retrieved {len(files)} processed files.")
                return files
        except sqlite3.Error as e:
            logger.error(f"Error fetching all processed files: {e}", exc_info=True)
            return []

    def get_processed_file_details(file_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves details for a specific processed file by its ID."""
        logger.debug(f"Fetching details for processed file ID: {file_id}")
        sql = "SELECT id, file_name, file_hash, collection_name, total_chunks, processed_at FROM processed_files WHERE id = ?"
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                logger.warning(f"No processed file found with ID: {file_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching processed file ID {file_id}: {e}", exc_info=True)
            return None

    def add_chat_message(file_id: int, role: str, content: str, sources: Optional[List[Dict]] = None):
        """Adds a chat message to the database for a given file_id."""
        logger.debug(f"Adding chat message to DB for file_id {file_id}, role: {role}")
        sources_json = json.dumps(sources) if sources else None
        sql = """
            INSERT INTO chat_messages (file_id, role, content, sources_json)
            VALUES (?, ?, ?, ?)
        """
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_id, role, content, sources_json))
                conn.commit()
                logger.debug(f"Chat message added for file_id {file_id}, role {role}.")
        except sqlite3.Error as e:
            logger.error(f"Error adding chat message for file_id {file_id}: {e}", exc_info=True)

    def get_chat_history(file_id: int) -> List[Dict[str, Any]]:
        """Retrieves chat history for a given file_id."""
        logger.debug(f"Fetching chat history for file_id {file_id}.")
        sql = "SELECT role, content, sources_json, timestamp FROM chat_messages WHERE file_id = ? ORDER BY timestamp ASC"
        messages: List[Dict[str, Any]] = []
        try:
            with _get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (file_id,))
                for row in cursor.fetchall():
                    message = dict(row)
                    if message['sources_json']:
                        try:
                            message['sources'] = json.loads(message['sources_json'])
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode sources_json for a message in file_id {file_id}")
                            message['sources'] = None # Or some error indicator
                    else:
                        message['sources'] = None
                    del message['sources_json'] # Remove raw JSON string
                    messages.append(message)
                logger.debug(f"Retrieved {len(messages)} chat messages for file_id {file_id}.")
                return messages
        except sqlite3.Error as e:
            logger.error(f"Error fetching chat history for file_id {file_id}: {e}", exc_info=True)
            return []

    # Call init_db() when the module is first imported to ensure tables exist.
    # This is a simple way to handle initialization. For more complex apps,
    # you might have an explicit setup step.
    # init_db() # Be cautious with calling this at import time in some frameworks, 
                # for Streamlit this is generally okay for simple cases.
                # We'll call it explicitly from the app setup.

    if __name__ == '__main__':
        print("Running History Manager Tests...")
        init_db() # Ensure DB is set up for testing

        # Test adding a file
        print("\nTesting add_processed_file...")
        test_file_id = add_processed_file("test_paper.pdf", "testhash123", "collection_test_paper", 120)
        if test_file_id:
            print(f"Added/found test_paper.pdf with ID: {test_file_id}")
            
            # Test retrieving the file
            print("\nTesting get_all_processed_files...")
            all_files = get_all_processed_files()
            print(f"All processed files: {all_files}")
            assert any(f['file_name'] == "test_paper.pdf" for f in all_files)

            print("\nTesting get_processed_file_details...")
            details = get_processed_file_details(test_file_id)
            print(f"Details for ID {test_file_id}: {details}")
            assert details is not None and details['file_name'] == "test_paper.pdf"

            # Test adding chat messages
            print("\nTesting add_chat_message...")
            add_chat_message(test_file_id, "user", "What is this paper about?")
            source_example = [{"page_content": "This is context from page 1.", "metadata": {"source": "test_paper.pdf", "page": 1}}]
            add_chat_message(test_file_id, "assistant", "It's about AI.", sources=source_example)
            
            # Test retrieving chat history
            print("\nTesting get_chat_history...")
            history = get_chat_history(test_file_id)
            print(f"Chat history for file ID {test_file_id}:")
            for msg in history:
                print(msg)
            assert len(history) >= 2
            assert history[0]['role'] == 'user'
            assert history[1]['role'] == 'assistant'
            assert history[1]['sources'] is not None

            print("\nHistory Manager tests completed.")
        else:
            print("Failed to add test_paper.pdf for testing.")
    ```
3.  **Action:** Save the file (`app/services/history_manager.py`).
4.  **Explanation:**
    * **`DB_PATH`**: Gets the database file path from `config/settings.py`.
    * **`_get_db_connection()`**: A helper to connect to SQLite. `conn.row_factory = sqlite3.Row` allows accessing columns by name (like `row['file_name']`).
    * **`init_db()`**: Creates two tables:
        * `processed_files`: Stores metadata about each PDF that has been processed (name, hash for potential de-duplication, the ChromaDB collection name used for its chunks, number of chunks, and timestamp). `UNIQUE` constraints help prevent duplicate entries.
        * `chat_messages`: Stores individual chat messages, linked to a `file_id`. Includes `role` ('user' or 'assistant'), `content`, `sources_json` (for assistant messages, storing the source document chunks used), and a `timestamp`.
    * **`add_processed_file()`**: Inserts a new record. Handles `sqlite3.IntegrityError` if a file with the same unique fields (name, hash, or collection) is added, and tries to return the existing ID.
    * **`get_all_processed_files()`**: Fetches all processed files, ordered by processing time (newest first).
    * **`get_processed_file_details()`**: Fetches details for one specific file ID.
    * **`add_chat_message()`**: Stores a chat message. `sources` are serialized to a JSON string for storage.
    * **`get_chat_history()`**: Retrieves messages for a specific `file_id`, deserializing `sources_json` back into a Python list of dictionaries.
    * **`if __name__ == '__main__':`**: Contains test functions to verify DB operations.
    * **`init_db()` call:** We will call `init_db()` once when the Streamlit app starts to ensure the database schema is ready.

---

### Part B: Streamlit User Interface (`app/ui/streamlit_app.py`)

This is the main application file that users will interact with. It will orchestrate the backend modules we've built.

1.  **Action:** Open the file `app/ui/streamlit_app.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/ui/streamlit_app.py
    import streamlit as st
    from pathlib import Path
    import hashlib
    import time # For simulating processing delays

    # Import backend modules
    from app.data_processing.pdf_parser import parse_pdf
    from app.data_processing.text_chunker import chunk_text
    from app.semantic_indexing.vector_store import add_documents_to_vector_store, get_vector_store_retriever, _generate_collection_name
    from app.retrieval_augmentation.rag_pipeline import create_or_get_rag_chain, get_rag_response
    from app.services import history_manager # Will import functions from history_manager
    from app.utils.logging_config import get_logger, setup_logging

    # --- Page Configuration and Initial Setup ---
    st.set_page_config(page_title="Research Paper Chatbot", layout="wide")
    logger = get_logger(__name__) # Get logger for this UI module
    # setup_logging() # Setup logging configuration (will be called from run.py)


    def calculate_file_hash(file_bytes: bytes) -> str:
        """Calculates SHA256 hash of file content."""
        return hashlib.sha256(file_bytes).hexdigest()

    # --- Initialize Session State ---
    if "messages" not in st.session_state: # Chat messages for the current session
        st.session_state.messages = []
    if "current_doc_id" not in st.session_state: # DB ID of the currently selected processed file
        st.session_state.current_doc_id = None
    if "current_doc_name" not in st.session_state: # Filename of current doc
        st.session_state.current_doc_name = None
    if "rag_chain" not in st.session_state: # Cached RAG chain for the current doc
        st.session_state.rag_chain = None
    if "processed_files_list" not in st.session_state: # Cache for processed files dropdown
        st.session_state.processed_files_list = []


    def load_processed_files():
        """Loads processed files from the database into session state."""
        st.session_state.processed_files_list = history_manager.get_all_processed_files()

    def display_chat_history():
        """Displays chat messages from session state."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("View Sources"):
                        for i, source_doc in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:** (Page: {source_doc.metadata.get('page', 'N/A')}, Chunk start index: {source_doc.metadata.get('start_index', 'N/A')})")
                            st.caption(source_doc.page_content)


    # --- Main Application Logic ---
    st.title("📚 Research Paper Chatbot")
    st.markdown("Upload a research paper (PDF), select it, and ask questions!")

    # Initialize/Load processed files list on first load or refresh
    if not st.session_state.processed_files_list:
        load_processed_files()

    # --- Sidebar for File Management ---
    with st.sidebar:
        st.header("📄 Document Management")

        uploaded_file = st.file_uploader("Upload a new research paper (PDF)", type="pdf")

        if uploaded_file is not None:
            if st.button(f"Process '{uploaded_file.name}'"):
                with st.spinner(f"Processing {uploaded_file.name}... This may take a moment."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = calculate_file_hash(file_bytes)
                        file_name = uploaded_file.name
                        
                        # Check if file hash already exists in DB
                        existing_file_id_by_hash = history_manager.get_processed_file_id_by_hash(file_hash)
                        if existing_file_id_by_hash:
                            st.info(f"This file ('{file_name}') has already been processed (ID: {existing_file_id_by_hash}). Please select it from the list below.")
                            st.session_state.current_doc_id = existing_file_id_by_hash
                            doc_details = history_manager.get_processed_file_details(existing_file_id_by_hash)
                            st.session_state.current_doc_name = doc_details['file_name'] if doc_details else file_name
                        else:
                            logger.info(f"Starting processing for new file: {file_name}")
                            page_texts = parse_pdf(file_bytes) # Pass bytes directly
                            if not page_texts:
                                st.error("Could not extract text from the PDF.")
                                raise ValueError("PDF parsing failed to extract text.")
                            
                            full_text = "\n\n".join(page_texts)
                            chunks = chunk_text(full_text, file_name=file_name)
                            if not chunks:
                                st.error("Could not chunk the extracted text.")
                                raise ValueError("Text chunking failed.")

                            # Generate collection name (same logic as in vector_store.py but we need it here for history)
                            collection_name = _generate_collection_name(file_hash) # Use hash for collection ID
                                                        
                            add_documents_to_vector_store(chunks, file_hash) # Use hash for vector store ID
                            logger.info(f"Documents added to vector store for {file_name} using hash {file_hash[:8]} as ID.")
                            
                            # Add to history DB
                            db_file_id = history_manager.add_processed_file(
                                file_name=file_name, 
                                file_hash=file_hash, 
                                collection_name=collection_name,
                                total_chunks=len(chunks)
                            )
                            if db_file_id:
                                st.success(f"'{file_name}' processed and added to vector store and history (ID: {db_file_id}).")
                                st.session_state.current_doc_id = db_file_id
                                st.session_state.current_doc_name = file_name
                                load_processed_files() # Refresh list
                            else:
                                st.error(f"Failed to add '{file_name}' to processing history database.")
                                raise ValueError("Failed to save file processing history.")
                        
                        # Clear previous chat and load new RAG chain
                        st.session_state.messages = []
                        if st.session_state.current_doc_id:
                            doc_details = history_manager.get_processed_file_details(st.session_state.current_doc_id)
                            if doc_details:
                                st.session_state.rag_chain = create_or_get_rag_chain(doc_details['file_hash']) # RAG chain uses file_hash as ID
                                st.session_state.messages = history_manager.get_chat_history(st.session_state.current_doc_id)
                            else:
                                st.session_state.rag_chain = None

                    except Exception as e:
                        logger.error(f"Error processing uploaded file {uploaded_file.name}: {e}", exc_info=True)
                        st.error(f"An error occurred during processing: {e}")
                        st.session_state.current_doc_id = None # Reset on error
                        st.session_state.current_doc_name = None
                        st.session_state.rag_chain = None
                st.rerun() # Rerun to reflect selection change and load chat


        st.markdown("---")
        st.subheader("Select Processed Paper")
        
        # Create a list of display names for the selectbox
        # Format: "filename.pdf (ID: X)"
        # Store a mapping of display name to file_id for easy retrieval
        if st.session_state.processed_files_list:
            options_map = {
                f"{f['file_name']} (Chunks: {f['total_chunks']}, ID: {f['id']})": f['id'] 
                for f in st.session_state.processed_files_list
            }
            display_options = list(options_map.keys())
            
            # Find current selection index if any
            current_selection_display = None
            if st.session_state.current_doc_id:
                for disp_name, f_id in options_map.items():
                    if f_id == st.session_state.current_doc_id:
                        current_selection_display = disp_name
                        break
            
            selected_display_name = st.selectbox(
                "Choose a paper to chat with:",
                options=display_options,
                index=display_options.index(current_selection_display) if current_selection_display in display_options else 0,
                key="selected_processed_file_display"
            )

            # Handle selection change
            if selected_display_name and options_map[selected_display_name] != st.session_state.current_doc_id:
                selected_file_id = options_map[selected_display_name]
                logger.info(f"User selected processed file ID: {selected_file_id} from dropdown.")
                st.session_state.current_doc_id = selected_file_id
                doc_details = history_manager.get_processed_file_details(selected_file_id)
                if doc_details:
                    st.session_state.current_doc_name = doc_details['file_name']
                    # RAG chain and vector store use file_hash as the document_id
                    st.session_state.rag_chain = create_or_get_rag_chain(doc_details['file_hash'])
                    st.session_state.messages = history_manager.get_chat_history(selected_file_id)
                else:
                    st.error(f"Could not load details for selected file ID {selected_file_id}.")
                    st.session_state.current_doc_name = None
                    st.session_state.rag_chain = None
                    st.session_state.messages = []
                st.rerun() # Rerun to reflect changes
        else:
            st.info("No papers processed yet. Upload one to get started!")


    # --- Main Chat Area ---
    if st.session_state.current_doc_id and st.session_state.current_doc_name:
        st.header(f"Chat with: {st.session_state.current_doc_name}")

        if st.session_state.rag_chain:
            display_chat_history() # Display loaded or current chat messages

            if prompt := st.chat_input(f"Ask a question about '{st.session_state.current_doc_name}'..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                history_manager.add_chat_message(st.session_state.current_doc_id, "user", prompt)
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Thinking..."):
                    try:
                        answer, sources = get_rag_response(st.session_state.rag_chain, prompt)
                        if answer:
                            assistant_message_content = answer
                            # Prepare sources metadata for storage and display
                            formatted_sources = None
                            if sources:
                                formatted_sources = [
                                    {
                                        "page_content": doc.page_content,
                                        "metadata": doc.metadata
                                    } for doc in sources
                                ]
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": assistant_message_content,
                                "sources": formatted_sources
                            })
                            history_manager.add_chat_message(
                                st.session_state.current_doc_id, 
                                "assistant", 
                                assistant_message_content,
                                sources=formatted_sources
                            )
                            # Display immediately (will also be caught by display_chat_history on next interaction)
                            with st.chat_message("assistant"):
                                st.markdown(assistant_message_content)
                                if formatted_sources:
                                    with st.expander("View Sources"):
                                        for i, source_doc in enumerate(formatted_sources):
                                            st.write(f"**Source {i+1}:** (Page: {source_doc['metadata'].get('page', 'N/A')}, Chunk start index: {source_doc['metadata'].get('start_index', 'N/A')})") # Access metadata correctly
                                            st.caption(source_doc['page_content'])
                        else:
                            st.warning("The assistant could not generate an answer based on the provided document.")
                            st.session_state.messages.append({"role": "assistant", "content": "I couldn't find an answer to that in the document."})
                            history_manager.add_chat_message(st.session_state.current_doc_id, "assistant", "I couldn't find an answer to that in the document.")
                    except Exception as e:
                        logger.error(f"Error getting RAG response: {e}", exc_info=True)
                        st.error(f"An error occurred while generating the response: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
                        # Do not save error messages to history typically, or save a generic error message
                # No st.rerun() here, chat_input handles updates automatically
        else:
            st.warning(f"Could not load the RAG model for '{st.session_state.current_doc_name}'. Processing might have failed or is incomplete.")
    else:
        st.info("Upload a PDF or select a previously processed one from the sidebar to start chatting.")

    ```
3.  **Action:** Save the file (`app/ui/streamlit_app.py`).
4.  **Explanation (Key Parts):**
    * **`st.set_page_config`**: Sets the browser tab title and layout.
    * **`st.session_state`**: Streamlit's way to store variables that persist across user interactions (reruns). We use it to keep track of chat messages, the currently selected document, its RAG chain, and the list of processed files.
    * **`load_processed_files()`**: Fetches the list of files from our SQLite DB.
    * **Sidebar (`st.sidebar`)**:
        * `st.file_uploader`: Allows users to upload PDF files.
        * "Process PDF" button: Triggers the backend pipeline:
            * Calculates a `file_hash` to uniquely identify the file content. This hash is used as the `document_id` for the vector store's collection name to ensure consistency if the same file content is processed again (even with a different filename).
            * Calls `parse_pdf`, `chunk_text`, `add_documents_to_vector_store`.
            * Saves metadata to the SQLite DB using `history_manager.add_processed_file`.
            * Loads the RAG chain for the newly processed document.
            * Uses `st.spinner` to show a loading indicator during processing.
            * Handles cases where a file with the same hash has already been processed.
        * `st.selectbox`: Lists previously processed files. Selecting one:
            * Updates `st.session_state.current_doc_id` and `current_doc_name`.
            * Loads the corresponding RAG chain using `create_or_get_rag_chain` (using the `file_hash` from the DB as the `document_id` for the chain).
            * Loads the chat history for this document.
            * `st.rerun()`: Forces Streamlit to re-execute the script from top to bottom to reflect state changes.
    * **Main Chat Area:**
        * Only active if a document is selected (`current_doc_id` is set) and its RAG chain is loaded.
        * `display_chat_history()`: Renders messages from `st.session_state.messages`. For assistant messages with sources, it uses `st.expander` to show them.
        * `st.chat_input`: Provides the text input field for user questions.
        * When a query is entered:
            * It's added to `st.session_state.messages` and the DB.
            * `get_rag_response` is called.
            * The assistant's answer and sources are displayed and saved.

---

### Part C: Main Application Runner (`run.py`)

This script will be the entry point to start your Streamlit application.

1.  **Action:** Open the file `run.py` (in the project root `research_chatbot/`) in VS Code.
2.  **Action:** Replace its content with the following Python code:

    ```python
    # run.py
    import sys
    import os
    from pathlib import Path

    # Ensure the 'app' directory is in the Python path
    # This allows imports like 'from app.utils...'
    # This is more robust for different ways of running the script.
    current_dir = Path(__file__).parent.resolve()
    sys.path.append(str(current_dir))

    # Import setup_logging and init_db after path modification
    from app.utils.logging_config import setup_logging
    from app.services.history_manager import init_db

    def main():
        """Main function to set up and run the Streamlit application."""
        # 1. Setup Logging (Call this first)
        try:
            setup_logging()
            logger = setup_logging() # To get a logger instance if needed here
            # logger.info("Application starting up...") # This might log twice if setup_logging also logs
        except Exception as e:
            print(f"Critical error during logging setup: {e}")
            # Fallback to basic print logging if logging setup fails
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Logging setup failed: {e}", exc_info=True)
            # Decide if you want to exit or try to continue
            # sys.exit("Failed to initialize logging. Application cannot start.")

        # 2. Initialize Database (Call this before Streamlit tries to access it)
        try:
            init_db()
        except Exception as e:
            # Use the logger if available, otherwise print
            log_func = logging.error if 'logger' in locals() and logger else print
            log_func(f"Critical error during database initialization: {e}")
            sys.exit("Failed to initialize database. Application cannot start.")


        # 3. Construct the path to the Streamlit app script
        # Assuming run.py is in project_chatbot/ and streamlit_app.py is in project_chatbot/app/ui/
        streamlit_app_path = Path(__file__).parent / "app" / "ui" / "streamlit_app.py"
        
        if not streamlit_app_path.exists():
            log_func = logging.error if 'logger' in locals() and logger else print
            log_func(f"Streamlit app script not found at: {streamlit_app_path}")
            sys.exit("Streamlit application script not found.")

        # 4. Prepare and run the Streamlit command
        # This approach is more robust than os.system for running Streamlit programmatically
        try:
            from streamlit.web import cli as stcli
            # Prepare arguments for Streamlit CLI
            # Note: For programmatic running, some Streamlit CLI args might behave differently
            # or might be better set via Streamlit's config.toml or server options.
            args = [
                "run",
                str(streamlit_app_path),
                "--server.port", "8501",
                "--server.address", "0.0.0.0" # Listen on all available network interfaces
            ]
            sys.exit(stcli.main(args)) # Execute Streamlit
        except ImportError:
            log_func = logging.error if 'logger' in locals() and logger else print
            log_func("Streamlit is not installed. Please install it: pip install streamlit")
            sys.exit("Streamlit not found.")
        except Exception as e:
            log_func = logging.error if 'logger' in locals() and logger else print
            log_func(f"Failed to run Streamlit application: {e}")
            sys.exit("Streamlit application failed to start.")


    if __name__ == "__main__":
        main()
    ```
3.  **Action:** Save the file (`run.py`).
4.  **Explanation:**
    * **Path Setup:** It adds the project's root directory (where `run.py` is) to `sys.path`. This helps ensure that Python can find your `app` package and its modules (like `app.utils.logging_config` and `app.services.history_manager`) correctly, regardless of how or from where `run.py` is executed.
    * **`setup_logging()` and `init_db()`**: These are called right at the start to ensure logging and the database are ready before the Streamlit app logic begins. This is crucial.
    * **Streamlit CLI Execution:**
        * It imports `streamlit.web.cli as stcli`.
        * Constructs the command arguments similar to how you'd run Streamlit from the command line: `streamlit run app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`.
        * `sys.exit(stcli.main(args))`: This is the recommended programmatic way to launch a Streamlit app. It gives Streamlit control of the process.
        * `--server.address 0.0.0.0` makes the app accessible from other devices on your network (using your machine's IP address and port 8501). If you only want local access, you can use `127.0.0.1` or omit it for Streamlit's default.

### Step 4: Update `requirements.txt`

Ensure Streamlit is recorded.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run:
    ```powershell
    pip freeze > requirements.txt
    ```

### Step 5: Commit Changes to Git

1.  **Action:** In your PowerShell terminal, run:
    ```powershell
    git add app/services/history_manager.py app/ui/streamlit_app.py run.py requirements.txt
    git commit -m "feat: Implement Streamlit UI, history management, and app runner"
    ```

---

Phase 4 is complete! This was a major step, providing the interactive frontend and data persistence for our application.

**Next Steps Before Running:**
1.  **Download LLM Model:** Ensure you have downloaded a LLaMA 2 GGUF model (e.g., `llama-2-7b-chat.Q4_K_M.gguf`) and placed it in the `research_chatbot/models/` directory.
2.  **Update `config/settings.py`:** Make sure `LLM_MODEL_PATH` in `config/settings.py` points to the correct filename of your downloaded model.

**To Run Your Application:**
1.  Open your PowerShell terminal.
2.  Navigate to your project directory: `cd D:\Machine_learning\Projects\genAI\research_chatbot`
3.  Activate the virtual environment: `.\.venv\Scripts\activate`
4.  Run the application: `python run.py`
5.  Open your web browser and go to `http://localhost:8501` (or the URL shown in the terminal).