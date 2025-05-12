# RAG orchestration logic

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