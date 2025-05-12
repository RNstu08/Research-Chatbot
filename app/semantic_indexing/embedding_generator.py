# Embedding model utilities

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