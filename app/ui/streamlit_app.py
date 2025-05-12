# app/ui/streamlit_app.py
import sys
from pathlib import Path
# when running Streamlit from the project root.
project_root = Path(__file__).resolve().parents[2] # Go up two levels: streamlit_app.py -> ui -> app -> research_chatbot
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


import streamlit as st
import hashlib
import time # For simulating processing delays if needed

# Import backend modules (ensure these paths are correct relative to project root)
from app.data_processing.pdf_parser import parse_pdf
from app.data_processing.text_chunker import chunk_text
from app.semantic_indexing.vector_store import add_documents_to_vector_store, get_vector_store_retriever, _generate_collection_name # Allow importing helper
from app.retrieval_augmentation.rag_pipeline import create_or_get_rag_chain, get_rag_response
from app.services import history_manager # Will import functions from history_manager
from app.utils.logging_config import get_logger
import hashlib
import time


# --- Page Configuration and Initial Setup ---
st.set_page_config(page_title="Research Paper Chatbot", layout="wide")
logger = get_logger(__name__) # Get logger for this UI module
# Logging/DB Init will be handled by run.py before streamlit starts

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculates SHA256 hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()

# --- Initialize Session State ---
# Ensure all necessary keys are initialized
default_session_state = {
    "messages": [],              # Chat messages for the current session/document
    "current_doc_id": None,      # DB ID of the currently selected processed file
    "current_doc_name": None,    # Filename of current doc
    "current_doc_hash": None,    # Hash of the current doc (used for RAG chain ID)
    "rag_chain": None,           # Cached RAG chain for the current doc
    "processed_files_list": [],  # Cache for processed files dropdown
    "file_uploader_key": 0       # Key to reset file uploader
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Utility Functions for UI ---
def load_processed_files():
    """Loads/refreshes processed files from the database into session state."""
    st.session_state.processed_files_list = history_manager.get_all_processed_files()
    logger.debug(f"Refreshed processed files list: {len(st.session_state.processed_files_list)} items")

def select_document(file_id: int):
    """Sets the state when a document is selected."""
    if file_id == st.session_state.current_doc_id:
        return # No change

    logger.info(f"Selecting document with DB ID: {file_id}")
    doc_details = history_manager.get_processed_file_details(file_id)
    if doc_details:
        st.session_state.current_doc_id = doc_details['id']
        st.session_state.current_doc_name = doc_details['file_name']
        st.session_state.current_doc_hash = doc_details['file_hash'] # Store the hash
        st.session_state.messages = history_manager.get_chat_history(file_id) # Load chat history

        # Load RAG chain using the FILE HASH as the document_id for the chain/collection
        with st.spinner(f"Loading AI model for {st.session_state.current_doc_name}..."):
            st.session_state.rag_chain = create_or_get_rag_chain(st.session_state.current_doc_hash)

        if st.session_state.rag_chain is None:
             st.error(f"Failed to load the RAG chain for {st.session_state.current_doc_name}. It might not have been processed correctly.")
             # Reset selection partially?
             st.session_state.current_doc_id = None
             st.session_state.current_doc_name = None
             st.session_state.current_doc_hash = None
        else:
             logger.info(f"Successfully loaded RAG chain and chat history for {st.session_state.current_doc_name}")

    else:
        st.error(f"Could not find details for selected document ID {file_id}.")
        st.session_state.current_doc_id = None
        st.session_state.current_doc_name = None
        st.session_state.current_doc_hash = None
        st.session_state.rag_chain = None
        st.session_state.messages = []
    # We don't rerun here, selection logic in main body handles UI update

def display_chat_history():
    """Displays chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Check if 'sources' exists and is not None or empty
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources"):
                    for i, source_doc_data in enumerate(message["sources"]):
                        # Adapt based on how sources are stored (dict vs Document obj)
                        # Assuming sources are stored as list of dicts from history_manager
                        metadata = source_doc_data.get('metadata', {})
                        page_content = source_doc_data.get('page_content', '[Content not available]')
                        st.write(f"**Source {i+1}:** (Page: {metadata.get('page', 'N/A')}, Chunk start index: {metadata.get('start_index', 'N/A')})")
                        st.caption(page_content)

# --- Main Application Logic ---
st.title("📚 Research Paper Chatbot")

# Load processed files list initially if empty
if not st.session_state.processed_files_list:
    load_processed_files()

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Management")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a new research paper (PDF)", 
        type="pdf",
        key=f"file_uploader_{st.session_state.file_uploader_key}" # Use key to allow reset
    )

    # Process Button - Enabled only if a file is uploaded
    process_button_disabled = uploaded_file is None
    if st.button("Process Uploaded PDF", disabled=process_button_disabled):
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}... This may take a moment."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    file_hash = calculate_file_hash(file_bytes)
                    file_name = uploaded_file.name

                    logger.info(f"Processing uploaded file: {file_name}, Hash: {file_hash[:8]}...")

                    # Check if file hash already exists in DB
                    existing_file_id = history_manager.get_processed_file_id_by_hash(file_hash)

                    if existing_file_id:
                        st.info(f"This file ('{file_name}') has already been processed (ID: {existing_file_id}). Selecting it.")
                        newly_selected_id = existing_file_id
                    else:
                        logger.info(f"Parsing PDF: {file_name}")
                        page_texts = parse_pdf(file_bytes) # Pass bytes directly
                        if not page_texts:
                            raise ValueError("Could not extract text from the PDF.")

                        full_text = "\n\n".join(page_texts)
                        logger.info(f"Chunking text for: {file_name}")
                        chunks = chunk_text(full_text, file_name=file_name)
                        if not chunks:
                            raise ValueError("Could not chunk the extracted text.")

                        # Use file hash as the document_id for vector store and RAG chain consistency
                        vector_store_doc_id = file_hash 
                        collection_name = _generate_collection_name(vector_store_doc_id) 

                        logger.info(f"Adding {len(chunks)} chunks to vector store for {file_name} (ID: {vector_store_doc_id[:8]})")
                        add_documents_to_vector_store(chunks, vector_store_doc_id) 

                        # Add to history DB
                        logger.info(f"Adding file entry to history DB: {file_name}")
                        db_file_id = history_manager.add_processed_file(
                            file_name=file_name, 
                            file_hash=file_hash, 
                            collection_name=collection_name, # Store the generated collection name
                            total_chunks=len(chunks)
                        )
                        if db_file_id:
                            st.success(f"'{file_name}' processed successfully (ID: {db_file_id}).")
                            newly_selected_id = db_file_id
                            load_processed_files() # Refresh list in state
                        else:
                            raise ValueError("Failed to add file to processing history database.")

                    # Select the newly processed/found document
                    select_document(newly_selected_id)

                    # Reset the file uploader widget
                    st.session_state.file_uploader_key += 1
                    st.rerun() # Rerun to clear the uploader and update UI based on new selection

                except Exception as e:
                    logger.error(f"Error processing uploaded file {uploaded_file.name}: {e}", exc_info=True)
                    st.error(f"An error occurred during processing: {e}")
                    # Reset potentially inconsistent state
                    st.session_state.current_doc_id = None 
                    st.session_state.current_doc_name = None
                    st.session_state.current_doc_hash = None
                    st.session_state.rag_chain = None

        else:
            st.warning("Please upload a PDF file first.")

    st.markdown("---")

    # Dropdown for selecting processed files
    st.subheader("Select Processed Paper")
    if not st.session_state.processed_files_list:
         st.info("No papers processed yet.")
    else:
        # Create display names and map back to file_id
        options_map = { f"{f['file_name']} (ID: {f['id']})": f['id'] 
                       for f in st.session_state.processed_files_list }
        display_options = ["<Select a paper>"] + list(options_map.keys()) # Add a placeholder

        # Find current selection's display name
        current_selection_display = "<Select a paper>"
        if st.session_state.current_doc_id:
             for disp_name, f_id in options_map.items():
                if f_id == st.session_state.current_doc_id:
                    current_selection_display = disp_name
                    break

        selected_display_option = st.selectbox(
            "Choose a paper to chat with:",
            options=display_options,
            index=display_options.index(current_selection_display), # Set current index
            key="processed_doc_selector" 
        )

        # Handle selection change if a valid paper (not placeholder) is chosen
        if selected_display_option != "<Select a paper>":
            selected_file_id = options_map[selected_display_option]
            if selected_file_id != st.session_state.current_doc_id:
                select_document(selected_file_id)
                st.rerun() # Rerun to load chat for the selected document
        elif st.session_state.current_doc_id is not None and selected_display_option == "<Select a paper>":
            # User explicitly selected the placeholder, deselect current doc
            logger.info("User deselected document.")
            st.session_state.current_doc_id = None
            st.session_state.current_doc_name = None
            st.session_state.current_doc_hash = None
            st.session_state.rag_chain = None
            st.session_state.messages = []
            st.rerun()


# --- Main Chat Area ---
st.header("💬 Chat Interface")

if st.session_state.current_doc_id and st.session_state.current_doc_name:
    st.info(f"Currently chatting with: **{st.session_state.current_doc_name}** (ID: {st.session_state.current_doc_id})")

    # Display existing chat messages
    display_chat_history()

    # Chat Input - only shown if RAG chain is loaded
    if st.session_state.rag_chain:
        if prompt := st.chat_input(f"Ask a question about '{st.session_state.current_doc_name}'..."):
            # Add user message to state and DB
            st.session_state.messages.append({"role": "user", "content": prompt})
            history_manager.add_chat_message(st.session_state.current_doc_id, "user", prompt)

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response
            with st.spinner("Assistant is thinking..."):
                try:
                    answer, sources = get_rag_response(st.session_state.rag_chain, prompt)

                    # Prepare assistant message data
                    assistant_message = {"role": "assistant"}
                    if answer:
                        assistant_message["content"] = answer
                        if sources:
                            # Prepare sources for storage/display (convert Document objects to dicts)
                            formatted_sources = [
                                {"page_content": doc.page_content, "metadata": doc.metadata} 
                                for doc in sources
                            ]
                            assistant_message["sources"] = formatted_sources
                        else:
                            assistant_message["sources"] = None
                    else:
                        assistant_message["content"] = "Sorry, I couldn't generate an answer based on the document."
                        assistant_message["sources"] = None

                    # Add assistant message to state and DB
                    st.session_state.messages.append(assistant_message)
                    history_manager.add_chat_message(
                        file_id=st.session_state.current_doc_id, 
                        role="assistant", 
                        content=assistant_message["content"],
                        sources=assistant_message.get("sources") # Pass the list of dicts
                    )

                    # Rerun to display the new assistant message along with sources
                    st.rerun()

                except Exception as e:
                    logger.error(f"Error getting RAG response: {e}", exc_info=True)
                    st.error(f"An error occurred while generating the response: {e}")
                    # Add error message to chat (optional)
                    # error_msg = {"role": "assistant", "content": f"Sorry, an error occurred: {e}"}
                    # st.session_state.messages.append(error_msg)
                    # history_manager.add_chat_message(st.session_state.current_doc_id, "assistant", error_msg["content"])
                    # st.rerun()

    else:
         # This case handles if RAG chain failed to load after selection
         st.warning(f"The AI model for '{st.session_state.current_doc_name}' could not be loaded. Please try re-processing the document or check the logs.")

else:
    st.info("Upload and process a PDF, or select one from the sidebar to begin chatting.")