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