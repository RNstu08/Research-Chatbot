# Text chunking strategies

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document class

# Import configuration and logger
from config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def chunk_text(full_text: str, file_name: str = "source_document") -> List[Document]:
    """
    Splits a large text into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        full_text: The entire text content extracted from the document.
        file_name: The original name of the file (for metadata).

    Returns:
        A list of LangChain Document objects, each representing a chunk.
        Returns an empty list if the input text is empty or chunking fails.
    """
    if not full_text:
        logger.warning("Received empty text for chunking.")
        return []

    logger.info(f"Starting chunking process for document '{file_name}'...")
    logger.info(f"Using chunk size: {settings.CHUNK_SIZE}, overlap: {settings.CHUNK_OVERLAP}")

    try:
        # Initialize the splitter
        # Separators are prioritized: try paragraphs, then lines, then spaces, then chars.
        # length_function=len uses character count by default for size calculation.
        # You could potentially use a tokenizer's length function here for token-based chunking,
        # but character-based is simpler to start with and common for RecursiveCharacterTextSplitter.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, # Adds character start index metadata to chunks
            separators=["\n\n", "\n", " ", ""], # Default separators
        )

        # Use create_documents which handles metadata assignment well
        # It takes a list of texts, but we provide our single full_text in a list
        # We also provide common metadata to be added to all resulting chunks.
        chunks = text_splitter.create_documents(
            texts=[full_text], 
            metadatas=[{"source": file_name}] # Add source filename to metadata
        )


        # Alternative: split_text returns list of strings, less ideal if using LC Documents later
        # chunks_text = text_splitter.split_text(full_text)
        # chunks = [Document(page_content=ct, metadata={"source": file_name}) for ct in chunks_text]

        logger.info(f"Successfully split text into {len(chunks)} chunks.")

        if chunks:
            logger.debug(f"Example chunk 1 metadata: {chunks[0].metadata}")
            # logger.debug(f"Example chunk 1 content preview: {chunks[0].page_content[:100]}...")

        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk text for document '{file_name}': {e}", exc_info=True)
        return [] # Return empty list on failure

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    # Combine text from the parser example (or use dummy text)
    # Assuming pdf_parser.py is in the same directory for this direct test
    try:
        from pdf_parser import parse_pdf
        test_pdf_path = Path("path/to/your/test.pdf") # <--- IMPORTANT: CHANGE THIS PATH

        if test_pdf_path.exists():
            print(f"\n--- Running Text Chunker Test ---")
            page_texts = parse_pdf(test_pdf_path)
            if page_texts:
                # Join text from all pages into one large string
                # Add double newline between pages to help splitter recognize page breaks
                full_document_text = "\n\n".join(page_texts) 
                print(f"Total characters extracted: {len(full_document_text)}")

                chunked_documents = chunk_text(full_document_text, file_name=test_pdf_path.name)

                if chunked_documents:
                    print(f"Text successfully chunked into {len(chunked_documents)} documents.")
                    print(f"Chunk size (chars) target: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")
                    # Print info about the first few chunks
                    for i, doc in enumerate(chunked_documents[:3]):
                        print(f"\nChunk {i+1}:")
                        print(f"  Metadata: {doc.metadata}")
                        print(f"  Content length: {len(doc.page_content)} chars")
                        # print(f"  Preview: {doc.page_content[:150].replace(chr(10), ' ')}...") # Replace newlines for preview
                else:
                    print("Chunking process resulted in zero chunks.")
            else:
                print("Failed to extract text using pdf_parser for chunker test.")
        else:
             print(f"Test PDF file not found at: {test_pdf_path}")
             print("Please update the test_pdf_path variable in text_chunker.py with a valid path to test.")

    except ImportError:
        print("Could not import pdf_parser. Make sure it's in the same directory for direct testing.")
    except Exception as e:
        print(f"An error occurred during the chunker test: {e}")