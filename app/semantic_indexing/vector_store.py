# VectorDB (ChromaDB) interaction

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