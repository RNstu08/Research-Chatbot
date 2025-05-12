

Let's begin **Phase 1: Data Ingestion & Preprocessing Module**. Our goal here is to create the Python code that can take an uploaded PDF, extract its text content, and then split that text into manageable chunks suitable for our AI model.

We will work primarily within the `app/data_processing/` directory.

### Step 1: Install Necessary Libraries

First, we need to install the libraries required for PDF parsing and text chunking.

1.  **Action:** In your PowerShell terminal (which should still show `(.venv) PS D:\...`), run the following `pip` commands:
    ```powershell
    # Library for efficient PDF handling
    pip install PyMuPDF

    # Core LangChain library and specific text splitters module
    pip install langchain langchain-text-splitters
    ```
2.  **Observe:** Pip will download and install these packages and their dependencies into your `.venv`.

### Step 2: Implement PDF Parsing (`app/data_processing/pdf_parser.py`)

This module will handle opening a PDF file and extracting its text content.

1.  **Action:** Open the file `app/data_processing/pdf_parser.py` in VS Code.

2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/data_processing/pdf_parser.py
    import fitz  # PyMuPDF
    from pathlib import Path
    from typing import Union, List
    import io

    # Import our logger
    from app.utils.logging_config import get_logger

    logger = get_logger(__name__)

    def parse_pdf(file_source: Union[str, Path, bytes]) -> List[str]:
        """
        Parses a PDF file and extracts text content page by page.

        Args:
            file_source: Either a file path (as string or Path object) 
                         or file content as bytes.

        Returns:
            A list of strings, where each string is the text content of a page.
            Returns an empty list if parsing fails.
            
        Raises:
            FileNotFoundError: If the file path provided does not exist.
            Exception: For other potential PyMuPDF errors during parsing.
        """
        page_texts: List[str] = []
        document = None # Initialize document to None for finally block

        try:
            if isinstance(file_source, (str, Path)):
                file_path = Path(file_source)
                if not file_path.is_file():
                    logger.error(f"File not found at path: {file_path}")
                    raise FileNotFoundError(f"No PDF file found at {file_path}")
                logger.info(f"Opening PDF from path: {file_path}")
                document = fitz.open(file_path)
            elif isinstance(file_source, bytes):
                logger.info("Opening PDF from bytes stream.")
                # Use io.BytesIO to handle the byte stream
                stream = io.BytesIO(file_source)
                document = fitz.open(stream=stream, filetype="pdf")
            else:
                logger.error(f"Unsupported file source type: {type(file_source)}")
                return [] # Return empty list for unsupported types

            logger.info(f"Successfully opened PDF. Number of pages: {len(document)}")

            for page_num in range(len(document)):
                page = document.load_page(page_num)
                text = page.get_text("text") # Extract text as plain text
                page_texts.append(text)
                logger.debug(f"Extracted text from page {page_num + 1}/{len(document)}")

            logger.info(f"Successfully extracted text from all {len(document)} pages.")
            return page_texts

        except FileNotFoundError as fnf_error:
            # Re-raise file not found specifically
            raise fnf_error
        except Exception as e:
            # Catch other potential errors from fitz.open() or page processing
            logger.error(f"Failed to parse PDF: {e}", exc_info=True) # exc_info=True logs traceback
            # Depending on desired behavior, you might return empty list or re-raise
            # For robustness in a pipeline, returning empty might be preferable
            # but raising allows upstream handling. Let's re-raise for now.
            raise Exception(f"PDF parsing failed. Reason: {e}") from e
        finally:
            if document:
                logger.debug("Closing PDF document.")
                document.close()

    # Example Usage (can be run directly for testing)
    if __name__ == '__main__':
        # Create a dummy PDF path for testing (replace with a real PDF path on your system)
        # You might need to create a 'test_pdfs' directory and put a sample PDF there
        # For example: D:\Machine_learning\Projects\genAI\research_chatbot\test_pdfs\sample.pdf
        test_pdf_path = Path("path/to/your/test.pdf") # <--- IMPORTANT: CHANGE THIS PATH
        
        # Check if the test file exists before trying to parse
        if test_pdf_path.exists():
            print(f"Attempting to parse PDF: {test_pdf_path}")
            try:
                pages = parse_pdf(test_pdf_path)
                if pages:
                    print(f"Successfully parsed {len(pages)} pages.")
                    # print("\n--- Text from Page 1 ---")
                    # print(pages[0][:500] + "..." if len(pages[0]) > 500 else pages[0]) # Print first 500 chars
                else:
                    print("Parsing returned no pages (or failed silently).")
            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during parsing: {e}")
        else:
            print(f"Test PDF file not found at: {test_pdf_path}")
            print("Please update the test_pdf_path variable in pdf_parser.py with a valid path to test.")

    ```

3.  **Action:** Save the file (`app/data_processing/pdf_parser.py`).

4.  **Explanation:**

      * **`import fitz`**: This imports the PyMuPDF library, which is conventionally imported as `fitz`.
      * **`get_logger(__name__)`**: We get a logger instance specific to this module for clear log messages. `__name__` automatically resolves to `"app.data_processing.pdf_parser"`.
      * **`parse_pdf` function**:
          * Accepts `file_source` which can be a file path (string or `Path` object) or raw `bytes` (useful for handling file uploads in memory).
          * Uses a `try...except...finally` block for robust error handling.
          * Opens the PDF using `fitz.open()`. It handles both file paths and byte streams.
          * Iterates through each `page` in the `document`.
          * `page.get_text("text")` extracts the plain text from the page. PyMuPDF offers other formats (like HTML, XML) which might preserve more structure but are complex to parse reliably; plain text is the most common starting point.
          * Appends the text of each page to the `page_texts` list.
          * Logs information about the process and errors using our configured logger.
          * Crucially includes a `finally` block to ensure `document.close()` is always called, releasing the file handle.
          * Handles `FileNotFoundError` specifically and raises other exceptions for upstream handling.
      * **`if __name__ == '__main__':` block**: This allows you to run this file directly (`python app/data_processing/pdf_parser.py`) for basic testing, provided you update `test_pdf_path` to point to an actual PDF file on your system.

### Step 3: Implement Text Chunking (`app/data_processing/text_chunker.py`)

This module takes the extracted text (likely joined from pages) and splits it into chunks.

1.  **Action:** Open the file `app/data_processing/text_chunker.py` in VS Code.

2.  **Action:** Replace the placeholder content with the following Python code:

    ```python
    # app/data_processing/text_chunker.py
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

    ```

3.  **Action:** Save the file (`app/data_processing/text_chunker.py`).

4.  **Explanation:**

      * **Imports:** We import `RecursiveCharacterTextSplitter` from `langchain_text_splitters` and `Document` from `langchain_core.documents` (which represents a piece of text with associated metadata). We also import our `settings` and `get_logger`.
      * **`chunk_text` function**:
          * Takes the `full_text` string and the `file_name` (for metadata).
          * Initializes `RecursiveCharacterTextSplitter` using `CHUNK_SIZE` and `CHUNK_OVERLAP` from `config/settings.py`.
          * `length_function=len`: Tells the splitter to measure chunk size by character count.
          * `add_start_index=True`: This is useful metadata, recording where each chunk started in the original text.
          * `separators=["\n\n", "\n", " ", ""]`: Defines the order in which the splitter tries to break the text (paragraphs -\> lines -\> spaces -\> characters).
          * `text_splitter.create_documents()`: This is the preferred LangChain method. It takes a list of texts (we provide our single `full_text`) and optionally a list of corresponding metadata dictionaries. It returns a list of `Document` objects, where each `Document` has `page_content` (the chunk text) and `metadata`. We add the source filename to the metadata here.
          * Includes logging and error handling.
      * **`if __name__ == '__main__':` block**: Provides a way to test the chunker directly, assuming `pdf_parser.py` is accessible and you provide a valid `test_pdf_path`. It parses the PDF, joins the pages, chunks the full text, and prints information about the resulting chunks.

### Step 4: Update `requirements.txt`

Now that we've installed new packages, let's record them.

1.  **Action:** In your PowerShell terminal (with `.venv` active), run:
    ```powershell
    pip freeze > requirements.txt
    ```
2.  **Observe:** This command overwrites `requirements.txt` with all the packages currently installed in your virtual environment, including `PyMuPDF`, `langchain`, `langchain-text-splitters`, and their dependencies.

### Step 5: Commit Changes to Git

Let's save our progress in version control.

1.  **Action:** In your PowerShell terminal, run the following commands:
    ```powershell
    git add app/data_processing/pdf_parser.py app/data_processing/text_chunker.py requirements.txt
    git commit -m "feat: Implement PDF parsing and text chunking modules"
    ```

-----

Phase 1 is now complete\! We have created the core components for processing input documents:

  * A function (`parse_pdf`) to reliably extract text from PDFs.
  * A function (`chunk_text`) to split that text into appropriately sized, semantically-aware chunks using a standard, robust strategy (`RecursiveCharacterTextSplitter`).
  * Dependencies are tracked in `requirements.txt`.

Our code is modular, includes logging and error handling, and uses configuration values. We are now ready to take these text chunks and prepare them for semantic search.
