**Phase 3: Retrieval & Generation Module**.

This is where the core "intelligence" comes together. We will load our local Large Language Model (LLaMA 2 GGUF) and create the Retrieval Augmented Generation (RAG) pipeline. This pipeline will use the vector store retriever (from Phase 2) to find relevant text chunks based on a user's query and then feed those chunks, along with the query, to the LLM to generate a grounded answer.

We will work primarily within the `app/retrieval_augmentation/` directory.

### Step 1: Install `llama-cpp-python` Library

This library allows us to run the LLaMA GGUF model efficiently on your CPU (or GPU if compiled with support). **Installation can sometimes be tricky** as it often involves C++ compilation.

1.  **Prerequisites (Important for Windows):**
    * **C++ Build Tools:** `llama-cpp-python` needs to compile C++ code. The easiest way to get the necessary tools on Windows is by installing **Microsoft C++ Build Tools**.
        * Go to the Visual Studio Downloads page: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
        * Scroll down to "Tools for Visual Studio".
        * Download the "Build Tools for Visual Studio".
        * Run the installer. Select the "Desktop development with C++" workload. You don't need the full Visual Studio IDE, just the build tools.
    * **CMake:** Ensure CMake is installed and added to your system's PATH. You can download it from [cmake.org/download/](https://cmake.org/download/). During installation, choose the option to "Add CMake to the system PATH for all users" or "current user". You can verify by opening a *new* PowerShell/CMD window and typing `cmake --version`.

2.  **Installation Action:** Once the prerequisites are installed, open your PowerShell terminal (with `.venv` active) inside the `research_chatbot` directory and run:

    ```powershell
    # This command attempts to build llama-cpp-python from source using your C++ tools.
    # It enables basic CPU acceleration features (like AVX2 if your CPU supports it).
    # For GPU (CUDA via cuBLAS), you'd need different flags and CUDA toolkit installed, 
    # but let's stick to CPU for simplicity first.
    pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```
    * **`--force-reinstall --upgrade --no-cache-dir`**: These flags help ensure you're getting the latest version and avoiding potential issues with cached builds.

3.  **Troubleshooting (If Installation Fails):**
    * **Check Build Tools:** Double-check that the C++ Build Tools (specifically the "Desktop development with C++" workload) and CMake are correctly installed and accessible in your PATH (you might need to restart PowerShell/CMD after installing them).
    * **Pre-built Wheels:** Sometimes, pre-compiled versions ("wheels") are available for specific Python versions and operating systems, bypassing the need for local compilation. Check the `llama-cpp-python` releases page on GitHub or PyPI for available wheels (`.whl` files). If you find one matching your Python version (e.g., `cp310` for Python 3.10, `win_amd64` for Windows 64-bit), you can download it and install using `pip install path/to/downloaded_wheel.whl`. This often avoids compilation issues but might not have the latest features or optimizations.
    * **Check Logs:** Carefully read any error messages during installation. They often point to missing tools or configuration problems.

4.  **Verification (Optional):** After installation, you can try importing it in a Python interpreter:
    * Run `python` in your terminal.
    * Type `from llama_cpp import Llama`
    * If it imports without error, the installation was likely successful. Type `exit()` to leave the Python interpreter.

### Step 2: Implement LLM Loading (`app/retrieval_augmentation/llm_interface.py`)

This module handles loading the LLaMA GGUF model using the LangChain wrapper around `llama-cpp-python`.

1.  **Action:** Open the file `app/retrieval_augmentation/llm_interface.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/retrieval_augmentation/llm_interface.py
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

    ```
3.  **Action:** Save the file (`app/retrieval_augmentation/llm_interface.py`).
4.  **Explanation:**
    * **`LlamaCpp`**: LangChain's wrapper for `llama-cpp-python`. It provides a standard LLM interface (`invoke`, `stream`, etc.).
    * **`load_llm`**: Function to handle loading.
    * **Path Validation:** Checks if the model file specified in `settings.LLM_MODEL_PATH` actually exists before attempting to load.
    * **Parameters:** We pass configuration values from `settings.py` (like `model_path`, `n_ctx`, `max_tokens`). `n_gpu_layers=0` ensures CPU-only operation initially. `verbose=False` keeps the console cleaner.
    * **Caching (`_llm_instance`):** Similar to the embedding model, we cache the loaded LLM instance to avoid reloading it repeatedly.
    * **Error Handling:** Catches exceptions during loading and logs them.
    * **`if __name__ == '__main__':` block**: Allows basic testing of model loading and invocation, *provided you have downloaded the model and updated `settings.py`*.

### Step 3: Implement RAG Pipeline (`app/retrieval_augmentation/rag_pipeline.py`)

This module brings everything together: it takes a user query, uses the appropriate document retriever, constructs a prompt with retrieved context, and gets the LLM to generate an answer.

1.  **Action:** Open the file `app/retrieval_augmentation/rag_pipeline.py` in VS Code.
2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/retrieval_augmentation/rag_pipeline.py
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from typing import Optional, Tuple, List

    # Import components from other modules
    from app.semantic_indexing.vector_store import get_vector_store_retriever
    from app.retrieval_augmentation.llm_interface import load_llm
    from app.utils.logging_config import get_logger

    logger = get_logger(__name__)

    # --- Custom Prompt Template ---
    # Define a prompt template to guide the LLM
    # This template explicitly tells the LLM to use the provided context.
    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks based on research papers. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer or the context doesn't contain the answer, just say that you don't know. 
    Do not try to make up an answer. Keep the answer concise and relevant to the context provided.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    rag_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Cache the RAG chains to avoid rebuilding them for the same document ID unnecessarily
    _rag_chains = {}

    def create_or_get_rag_chain(document_id: str):
        """
        Creates or retrieves a cached RetrievalQA chain for a specific document ID.

        Args:
            document_id: The unique identifier for the document's vector store collection.

        Returns:
            A RetrievalQA chain instance, or None if components fail to load.
        """
        global _rag_chains
        
        if document_id in _rag_chains:
            logger.info(f"Returning cached RAG chain for document ID: {document_id}")
            return _rag_chains[document_id]

        logger.info(f"Creating new RAG chain for document ID: {document_id}")

        # 1. Load the LLM
        llm = load_llm()
        if llm is None:
            logger.error("Failed to load LLM. Cannot create RAG chain.")
            return None

        # 2. Get the document-specific retriever
        # Use default k=4 from get_vector_store_retriever definition, or customize here
        retriever = get_vector_store_retriever(document_id, search_k=4) 
        if retriever is None:
            logger.error(f"Failed to get retriever for document ID '{document_id}'. Cannot create RAG chain.")
            return None
            
        # 3. Create the RetrievalQA chain
        try:
            # Chain Type "stuff": Puts all retrieved context chunks directly into the prompt.
            # Good for smaller contexts, might fail if context exceeds LLM window.
            # Alternatives: "map_reduce", "refine", "map_rerank" for handling larger context.
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True, # Crucial for showing evidence
                chain_type_kwargs={"prompt": rag_prompt} # Use our custom prompt
            )
            logger.info(f"Successfully created RAG chain for document ID: {document_id}")
            _rag_chains[document_id] = qa_chain # Cache the chain
            return qa_chain
            
        except Exception as e:
            logger.error(f"Failed to create RetrievalQA chain: {e}", exc_info=True)
            return None


    def get_rag_response(chain: RetrievalQA, query: str) -> Tuple[Optional[str], Optional[List[Document]]]:
        """
        Executes the RAG chain with a given query.

        Args:
            chain: The initialized RetrievalQA chain.
            query: The user's question.

        Returns:
            A tuple containing:
            - The generated answer string (or None if failed).
            - A list of source Document objects used (or None if failed or not returned).
        """
        if not chain:
            logger.error("Received invalid chain object.")
            return None, None
            
        logger.info(f"Executing RAG chain with query: '{query}'")
        try:
            # Use chain.invoke for the standard LangChain Expression Language interface
            response = chain.invoke({"query": query}) 
            
            answer = response.get("result")
            source_docs = response.get("source_documents")
            
            if answer:
                 logger.info("RAG chain executed successfully.")
            else:
                 logger.warning("RAG chain executed but returned no answer.")
                 
            return answer, source_docs

        except Exception as e:
            logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
            return None, None

    # Example Usage (can be run directly for testing)
    if __name__ == '__main__':
        # This test assumes previous steps worked and a document was indexed
        print("\n--- Running RAG Pipeline Test ---")
        
        # Use the same document ID as used in vector_store.py test
        test_pdf_path_rag = Path("path/to/your/test.pdf") # <--- IMPORTANT: CHANGE THIS PATH
        test_doc_id_rag = test_pdf_path_rag.name 

        if not test_pdf_path_rag.exists():
             print(f"Test PDF file not found at: {test_pdf_path_rag}. Cannot run RAG test.")
        else:
            print(f"Attempting to create RAG chain for document: {test_doc_id_rag}")
            rag_chain_instance = create_or_get_rag_chain(test_doc_id_rag)

            if rag_chain_instance:
                print("RAG chain created successfully.")
                
                test_query_rag = "What methodology was used in the study?" # Example query relevant to a research paper
                print(f"\nTesting RAG response with query: '{test_query_rag}'")
                
                answer_rag, sources_rag = get_rag_response(rag_chain_instance, test_query_rag)
                
                if answer_rag:
                    print("\n--- Generated Answer ---")
                    print(answer_rag)
                    
                    if sources_rag:
                        print("\n--- Source Documents Used ---")
                        for i, doc in enumerate(sources_rag):
                             print(f"  Source {i+1}: (Metadata: {doc.metadata})")
                             # print(f"    Content Preview: {doc.page_content[:150].replace(chr(10), ' ')}...")
                    else:
                        print("\n(No source documents were returned by the chain)")
                else:
                    print("\nFailed to get an answer from the RAG chain.")
                    
                # Test caching
                print("\nAttempting to get RAG chain again (should be cached)...")
                rag_chain_instance_2 = create_or_get_rag_chain(test_doc_id_rag)
                if rag_chain_instance is rag_chain_instance_2:
                    print("Successfully retrieved cached RAG chain instance.")
                else:
                    print("Warning: RAG chain was created again.")

            else:
                print("Failed to create RAG chain. Check logs for errors (LLM or Retriever loading?).")

    ```
3.  **Action:** Save the file (`app/retrieval_augmentation/rag_pipeline.py`).
4.  **Explanation:**
    * **`PROMPT_TEMPLATE`**: We define a custom prompt to explicitly instruct the LLM on how to behave. It tells the LLM to base its answer *only* on the provided `Context`, how to handle cases where the answer isn't found, and to be concise. This is crucial for reliable RAG.
    * **`rag_prompt = PromptTemplate(...)`**: Creates a LangChain `PromptTemplate` object from our string template.
    * **`create_or_get_rag_chain`**:
        * Takes a `document_id` to specify which document's retriever to use.
        * Loads the shared LLM instance using `load_llm()`.
        * Gets the specific retriever for the `document_id` using `get_vector_store_retriever()`.
        * Handles cases where the LLM or retriever fails to load.
        * Uses `RetrievalQA.from_chain_type`: This is a standard LangChain way to build a RAG pipeline.
            * `llm=llm`: Specifies the language model.
            * `chain_type="stuff"`: Tells the chain to "stuff" all retrieved documents directly into the context section of the prompt. Simple and effective if the context fits within the LLM's window.
            * `retriever=retriever`: Provides the mechanism to fetch relevant documents.
            * `return_source_documents=True`: Makes the chain return the specific `Document` objects used to generate the answer, allowing us to show evidence to the user.
            * `chain_type_kwargs={"prompt": rag_prompt}`: Injects our custom `PromptTemplate`.
        * Caches the created chain in the `_rag_chains` dictionary using the `document_id` as the key to avoid rebuilding it every time for the same document.
    * **`get_rag_response`**:
        * Takes the created `chain` and the user's `query`.
        * Executes the chain using `chain.invoke({"query": query})`. The input is a dictionary as required by this chain type.
        * Parses the output dictionary to extract the `result` (the answer) and the `source_documents`.
        * Returns the answer and sources.
    * **`if __name__ == '__main__':` block**: Tests the chain creation and execution process using the same document ID as the vector store test. It performs a sample query relevant to research papers and prints the answer and source document metadata. **Remember to update `test_pdf_path_rag`**.

### Step 4: Update `requirements.txt`

Record the `llama-cpp-python` dependency.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run:
    ```powershell
    pip freeze > requirements.txt
    ```

### Step 5: Commit Changes to Git

Save your progress.

1.  **Action:** In your PowerShell terminal, run the following commands:
    ```powershell
    git add app/retrieval_augmentation/llm_interface.py app/retrieval_augmentation/rag_pipeline.py requirements.txt
    git commit -m "feat: Implement LLM loading and RAG pipeline modules"
    ```

---

Phase 3 is complete! This was a significant phase. We now have:
* A way to load and interact with our local LLaMA 2 model.
* A complete RAG pipeline (`RetrievalQA` chain) that:
    * Takes a user query and document ID.
    * Retrieves relevant chunks from that document's vector store.
    * Uses a custom prompt to feed the query and context to the LLM.
    * Generates an answer grounded in the retrieved context.
    * Returns the answer and the source chunks used.

The core backend logic for our research chatbot!