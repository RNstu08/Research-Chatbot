**Phase 2: Semantic Indexing Module**.

The goal here is to convert the text chunks we created in Phase 1 into numerical representations called embeddings and store them efficiently so we can later search for chunks relevant to a user's query. We'll use a SentenceTransformer model for embedding and ChromaDB as our local vector database.

We will primarily work within the `app/semantic_indexing/` directory.

### Step 1: Install Necessary Libraries

We need libraries for generating embeddings and interacting with the vector database.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run the following `pip` commands:
    ```powershell
    # Installs sentence-transformers (HuggingFace embeddings) & dependencies
    pip install sentence-transformers

    # Installs ChromaDB vector database
    pip install chromadb

    # LangChain integration modules for ChromaDB and HuggingFace Embeddings
    # (might partially overlap with previous langchain installs, but ensures these components are present)
    pip install langchain-community langchain-chroma
    ```
2.  **Observe:** Pip will install these packages and their dependencies into your `.venv`. `langchain-community` provides the wrappers we'll use.

### Step 2: Implement Embedding Model Loading (`app/semantic_indexing/embedding_generator.py`)

This module will be responsible for loading the chosen SentenceTransformer model using a LangChain wrapper, making it easy to integrate into the rest of the system.

1.  **Action:** Open the file `app/semantic_indexing/embedding_generator.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/semantic_indexing/embedding_generator.py
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from config import settings
    from app.utils.logging_config import get_logger

    logger = get_logger(__name__)

    # Cache the embedding model instance for efficiency
    _embedding_model = None

    def get_embedding_model():
        """
        Loads and returns the HuggingFace embedding model specified in settings.
        Caches the model instance for subsequent calls.

        Returns:
            An instance of HuggingFaceEmbeddings.
            
        Raises:
            ImportError: If sentence-transformers is not installed.
            Exception: For issues during model loading.
        """
        global _embedding_model
        
        if _embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
                # Initialize the HuggingFaceEmbeddings class from langchain_community.
                # It uses sentence-transformers under the hood.
                # model_kwargs can be used to specify device (e.g., {'device': 'cuda'} or {'device': 'cpu'})
                # encode_kwargs can be used to specify normalization preferences (e.g., {'normalize_embeddings': False})
                _embedding_model = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'}, # Explicitly use CPU, change to 'cuda' if GPU is available/configured
                    encode_kwargs={'normalize_embeddings': True} # Normalize embeddings to unit length
                )
                logger.info("Embedding model loaded successfully.")
            except ImportError as ie:
                 logger.error("Sentence Transformers library not found. Please install it: pip install sentence-transformers")
                 raise ie
            except Exception as e:
                logger.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
                raise e # Re-raise the exception after logging

        return _embedding_model

    # Example Usage (can be run directly for testing)
    if __name__ == '__main__':
        print("Attempting to load embedding model...")
        try:
            embedding_function = get_embedding_model()
            print(f"Successfully loaded embedding model: {embedding_function.model_name}")
            
            # Test embedding a simple text
            test_text = "This is a test sentence."
            print(f"\nEmbedding test sentence: '{test_text}'")
            embedding_vector = embedding_function.embed_query(test_text) # Use embed_query for single texts
            
            print(f"Embedding successful!")
            print(f"Vector dimension: {len(embedding_vector)}")
            print(f"First 5 vector values: {embedding_vector[:5]}")
            
            # Test loading again (should use cache)
            print("\nAttempting to load embedding model again (should be cached)...")
            embedding_function_2 = get_embedding_model()
            if embedding_function is embedding_function_2: # Check if it's the same object instance
                 print("Successfully retrieved cached embedding model instance.")
            else:
                 print("Warning: Model was loaded again, caching might not be working as expected.")

        except Exception as e:
            print(f"An error occurred during embedding model test: {e}")

    ```
3.  **Action:** Save the file (`app/semantic_indexing/embedding_generator.py`).
4.  **Explanation:**
    * **`HuggingFaceEmbeddings`**: This LangChain class provides a convenient wrapper around SentenceTransformer models hosted on Hugging Face. It standardizes the interface for embedding both single queries (`embed_query`) and lists of documents (`embed_documents`).
    * **`get_embedding_model`**: This function retrieves the model name from `config/settings.py`.
    * **Caching (`_embedding_model`):** We use a simple global variable (`_embedding_model`) to cache the loaded model. Loading embedding models can take time and memory, so loading it only once is crucial for performance. More sophisticated caching or singleton patterns could be used in larger applications.
    * **`model_kwargs={'device': 'cpu'}`**: We explicitly tell the model to run on the CPU. If you have a compatible GPU and the necessary drivers (like CUDA for NVIDIA), you could change this to `'cuda'`.
    * **`encode_kwargs={'normalize_embeddings': True}`**: Normalizing embeddings to unit length is generally recommended for cosine similarity calculations, which is the default for many vector stores.
    * **Error Handling:** Includes checks for `ImportError` and general exceptions during model loading.
    * **`if __name__ == '__main__':` block**: Allows testing the model loading and a simple embedding task directly.

### Step 3: Implement Vector Store Interaction (`app/semantic_indexing/vector_store.py`)

This module manages the connection to ChromaDB and handles adding documents (chunks + embeddings) and retrieving the store for querying.

1.  **Action:** Open the file `app/semantic_indexing/vector_store.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/semantic_indexing/vector_store.py
    import chromadb
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from typing import List, Optional
    import hashlib

    # Import our configuration, logger, and embedding function
    from config import settings
    from app.utils.logging_config import get_logger
    from app.semantic_indexing.embedding_generator import get_embedding_model

    logger = get_logger(__name__)

    # Store the Chroma client instance globally within the module
    _chroma_client = None

    def _get_chroma_client():
        """Initializes and returns a persistent ChromaDB client."""
        global _chroma_client
        if _chroma_client is None:
            try:
                logger.info(f"Initializing ChromaDB client with persistence directory: {settings.CHROMA_PERSIST_DIR}")
                # Ensure the directory exists (ChromaDB might not create it automatically in all versions/setups)
                settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                
                _chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
                logger.info("ChromaDB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
                raise RuntimeError(f"Could not initialize ChromaDB client at {settings.CHROMA_PERSIST_DIR}") from e
        return _chroma_client

    def _generate_collection_name(document_id: str) -> str:
        """Generates a sanitized collection name for ChromaDB."""
        # Use a prefix and hash the document_id for a unique, valid name
        # Chroma collection names have restrictions (e.g., length, characters)
        hashed_id = hashlib.sha256(document_id.encode()).hexdigest()[:16] # Short hash
        safe_prefix = settings.CHROMA_COLLECTION_NAME_PREFIX.replace(" ", "_").lower()
        collection_name = f"{safe_prefix}_{hashed_id}"
        
        # Basic validation/sanitization (Chroma has specific rules, this is a simplified check)
        if not (3 <= len(collection_name) <= 63):
             raise ValueError(f"Generated collection name '{collection_name}' length is outside ChromaDB limits (3-63).")
        if not collection_name[0].isalnum() or not collection_name[-1].isalnum():
             raise ValueError(f"Generated collection name '{collection_name}' must start and end with alphanumeric characters.")
        # Replace invalid consecutive dots or invalid characters if needed (more robust sanitization could be added)
        collection_name = collection_name.replace("..", "_dotdot_") 

        logger.debug(f"Generated collection name: {collection_name} for document ID: {document_id}")
        return collection_name


    def add_documents_to_vector_store(documents: List[Document], document_id: str):
        """
        Embeds documents and adds them to a specific ChromaDB collection.
        Creates the collection if it doesn't exist.

        Args:
            documents: A list of LangChain Document objects (chunks).
            document_id: A unique identifier for the source document (e.g., filename).
        """
        if not documents:
            logger.warning(f"No documents provided for document ID '{document_id}'. Skipping vector store addition.")
            return

        collection_name = _generate_collection_name(document_id)
        logger.info(f"Adding {len(documents)} chunks to ChromaDB collection: '{collection_name}' for document ID: '{document_id}'")

        try:
            embedding_function = get_embedding_model()
            client = _get_chroma_client() # Ensures client is initialized

            # Use LangChain's Chroma integration to handle embedding and storage
            # This will create the collection if it doesn't exist.
            # NOTE: If the collection already exists, from_documents might behave differently 
            # depending on Chroma/LangChain versions (e.g., add, overwrite, error). 
            # For simplicity, we assume we create fresh or add non-duplicates here.
            # More robust logic might involve checking existence and deciding update strategy.
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=str(settings.CHROMA_PERSIST_DIR),
                client=client # Pass the initialized client
            )
            
            # Explicitly persist changes immediately after adding documents
            # vector_store.persist() # Often redundant if PersistentClient is used, but good practice to ensure writes
            
            logger.info(f"Successfully added {len(documents)} chunks to collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB collection '{collection_name}': {e}", exc_info=True)
            # Decide how to handle errors: raise, log and continue, etc.
            # For now, we log and don't raise to potentially allow partial processing if needed elsewhere.

    def get_vector_store_retriever(document_id: str, search_k: int = 4) -> Optional[Chroma]:
        """
        Gets an existing Chroma vector store for a given document ID, ready for retrieval.

        Args:
            document_id: The unique identifier for the source document.
            search_k: The default number of relevant chunks to retrieve.

        Returns:
            A LangChain Chroma vector store instance configured as a retriever, 
            or None if the collection cannot be loaded or doesn't exist.
        """
        collection_name = _generate_collection_name(document_id)
        logger.info(f"Attempting to load vector store for collection: '{collection_name}'")

        try:
            embedding_function = get_embedding_model()
            client = _get_chroma_client()

            # Check if collection exists before trying to load (more robust)
            try:
                 client.get_collection(name=collection_name)
                 logger.debug(f"Collection '{collection_name}' exists.")
            except Exception as get_coll_error:
                # This exception handling might vary slightly across chromadb versions
                logger.warning(f"Collection '{collection_name}' not found or error checking existence: {get_coll_error}. Cannot create retriever.")
                return None


            # Initialize the Chroma vector store object pointing to the existing collection
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=str(settings.CHROMA_PERSIST_DIR) 
            )
            
            logger.info(f"Successfully loaded vector store for collection '{collection_name}'.")
            # Configure the store as a retriever
            # search_type="similarity" is default, search_kwargs={'k': search_k} sets how many results to fetch
            return vector_store.as_retriever(search_kwargs={'k': search_k})

        except Exception as e:
            logger.error(f"Failed to get vector store retriever for collection '{collection_name}': {e}", exc_info=True)
            return None

    # Example Usage (can be run directly for testing)
    if __name__ == '__main__':
        # This test assumes pdf_parser and text_chunker work and are in the path
        # It also assumes you have set a valid test_pdf_path below
        print("\n--- Running Vector Store Test ---")
        try:
            from app.data_processing.pdf_parser import parse_pdf
            from app.data_processing.text_chunker import chunk_text
            
            test_pdf_path_vs = Path("path/to/your/test.pdf") # <--- IMPORTANT: CHANGE THIS PATH
            test_doc_id = test_pdf_path_vs.name # Use filename as document ID

            if not test_pdf_path_vs.exists():
                 print(f"Test PDF file not found at: {test_pdf_path_vs}")
            else:
                print(f"Processing document: {test_doc_id}")
                page_texts_vs = parse_pdf(test_pdf_path_vs)
                if page_texts_vs:
                    full_text_vs = "\n\n".join(page_texts_vs)
                    chunks_vs = chunk_text(full_text_vs, file_name=test_doc_id)
                    
                    if chunks_vs:
                        print(f"Generated {len(chunks_vs)} chunks. Attempting to add to vector store...")
                        # Add documents to store
                        add_documents_to_vector_store(chunks_vs, test_doc_id)
                        print("Addition process finished (check logs for success/errors).")

                        # Attempt to retrieve the store
                        print("\nAttempting to retrieve the vector store...")
                        retriever = get_vector_store_retriever(test_doc_id, search_k=3)

                        if retriever:
                            print("Successfully obtained retriever.")
                            # Test retrieval (requires a query)
                            test_query = "What is the main conclusion?" # Example query
                            print(f"\nTesting retrieval with query: '{test_query}'")
                            try:
                                relevant_docs = retriever.invoke(test_query) # Use invoke for LCEL interface
                                print(f"Retrieved {len(relevant_docs)} relevant chunks:")
                                for i, doc in enumerate(relevant_docs):
                                    print(f"  Chunk {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Start Index: {doc.metadata.get('start_index', 'N/A')}):")
                                    # print(f"    Content: {doc.page_content[:150].replace(chr(10), ' ')}...")
                            except Exception as retrieve_err:
                                print(f"Error during retrieval test: {retrieve_err}")
                        else:
                            print("Failed to obtain retriever (collection might not exist or error occurred).")
                    else:
                        print("Chunking failed, cannot test vector store.")
                else:
                    print("PDF parsing failed, cannot test vector store.")

        except ImportError:
            print("Could not import pdf_parser or text_chunker. Ensure they are accessible.")
        except Exception as e:
            print(f"An error occurred during the vector store test: {e}")
            logger.error("Vector store test failed", exc_info=True)

    ```
3.  **Action:** Save the file (`app/semantic_indexing/vector_store.py`).
4.  **Explanation:**
    * **ChromaDB Client (`_get_chroma_client`)**: Initializes a `PersistentClient`. This client connects to or creates database files in the directory specified by `settings.CHROMA_PERSIST_DIR`. This ensures our indexed data survives application restarts. We cache the client instance.
    * **Collection Name (`_generate_collection_name`)**: ChromaDB requires collection names with specific constraints (length, characters). We generate a unique and reasonably safe name using a prefix from settings and a hash of the `document_id` (like the filename).
    * **`add_documents_to_vector_store`**:
        * Takes the list of `Document` objects (chunks) from `text_chunker`.
        * Gets the embedding model instance.
        * Uses `Chroma.from_documents()`: This is a convenient LangChain function that:
            1.  Takes your list of `Document` objects.
            2.  Uses the provided `embedding_function` to generate embeddings for the `page_content` of each document.
            3.  Connects to the ChromaDB client and `persist_directory`.
            4.  Creates the specified `collection_name` if it doesn't exist.
            5.  Adds the documents (text + metadata) and their corresponding embeddings to the collection.
    * **`get_vector_store_retriever`**:
        * Takes a `document_id` to identify which document's collection we want.
        * Initializes a `Chroma` object pointing to the *existing* persisted collection. It's crucial that this step *doesn't* try to re-add documents, but just loads the existing store.
        * Includes a check (`client.get_collection`) to see if the collection actually exists before trying to load it.
        * `vector_store.as_retriever()`: This converts the vector store object into a LangChain `Retriever`. A retriever is an object with a standard interface (`invoke` or `get_relevant_documents`) for fetching relevant documents based on a query string. `search_kwargs={'k': search_k}` configures it to return the top `k` most similar chunks.
    * **`if __name__ == '__main__':` block**: This test orchestrates the whole process for a sample PDF: parse -> chunk -> add to store -> get retriever -> perform a test query. **Remember to update `test_pdf_path_vs`**.

### Step 4: Update `requirements.txt`

Record the newly installed packages.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run:
    ```powershell
    pip freeze > requirements.txt
    ```

### Step 5: Commit Changes to Git

Save your progress.

1.  **Action:** In your PowerShell terminal, run the following commands:
    ```powershell
    git add app/semantic_indexing/embedding_generator.py app/semantic_indexing/vector_store.py requirements.txt
    git commit -m "feat: Implement embedding generation and ChromaDB vector store modules"
    ```

---

Phase 2 is complete! We now have the capability to:
* Load a powerful SentenceTransformer embedding model.
* Take text chunks (as `Document` objects) and embed them.
* Store these chunks and their embeddings persistently in a local ChromaDB vector store, organized by document.
* Retrieve a specific document's vector store as a LangChain `Retriever`, ready for similarity searches.

We have built the core components of our knowledge base indexing pipeline.

