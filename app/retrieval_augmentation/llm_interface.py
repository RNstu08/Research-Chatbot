# LLM (LLaMA 2 GGUF) interaction

from langchain_community.llms import LlamaCpp
from config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Cache the LLM instance for efficiency
_llm_instance = None

def load_llm():
    """
    Loads and returns the LlamaCpp language model specified in settings.
    Caches the model instance for subsequent calls.

    Returns:
        An instance of LlamaCpp language model or None if loading fails.
    """
    global _llm_instance

    if _llm_instance is None:
        # Validate that the model path exists
        if not settings.LLM_MODEL_PATH.exists():
            logger.error(f"LLM model file not found at: {settings.LLM_MODEL_PATH}")
            logger.error("Please download the model and place it in the 'models' directory, then update 'config/settings.py'.")
            return None # Or raise an error

        logger.info(f"Loading LLM from path: {settings.LLM_MODEL_PATH}")
        logger.info(f"LLM settings - n_ctx: {settings.N_CTX}, max_tokens: {settings.MAX_TOKENS_LLM}")

        try:
            # Initialize LlamaCpp
            # Key parameters:
            # - model_path: Path to the GGUF model file.
            # - n_ctx: The context window size the model should use. Match model's capability.
            # - n_gpu_layers: Number of layers to offload to GPU. 0 for CPU-only. 
            #                 Requires llama-cpp-python compiled with GPU support (e.g., cuBLAS).
            # - n_batch: Number of tokens to process in parallel. Often related to n_ctx.
            # - max_tokens: Maximum number of tokens to generate in the response.
            # - temperature: Controls randomness (0=deterministic, >0=more random).
            # - top_p: Nucleus sampling parameter.
            # - verbose: If True, prints llama.cpp internal logs (can be noisy).
            _llm_instance = LlamaCpp(
                model_path=str(settings.LLM_MODEL_PATH),
                n_ctx=settings.N_CTX,
                n_gpu_layers=0,  # Set to 0 for CPU only inference
                n_batch=512,      # Should be appropriate for context size
                max_tokens=settings.MAX_TOKENS_LLM,
                temperature=0.7, # Adjust for desired creativity/factuality balance
                top_p=0.9,
                verbose=False,    # Set to True for detailed llama.cpp logs
                # stop=["\nUSER:", "\nASSISTANT:"] # Optional sequences to stop generation
            )
            logger.info("LLM instance loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}", exc_info=True)
            # Depending on desired behavior, could raise the error
            return None 

    return _llm_instance

# Example Usage (can be run directly for testing, requires model download)
if __name__ == '__main__':
    print("Attempting to load LLM...")
    llm = load_llm()

    if llm:
        print("LLM loaded successfully.")
        test_prompt = "Explain the concept of Retrieval Augmented Generation in one sentence."
        print(f"\nSending test prompt: '{test_prompt}'")

        try:
            response = llm.invoke(test_prompt) # Use invoke method
            print("\nLLM Response:")
            print(response)
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
    else:
        print("LLM loading failed. Check model path in config/settings.py and logs.")