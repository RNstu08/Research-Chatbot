# SQLite for file/chat history

import sqlite3
from typing import List, Tuple, Optional, Dict, Any
import json
from datetime import datetime

from config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = settings.SQLITE_DB_PATH

def _get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to SQLite database at {DB_PATH}: {e}", exc_info=True)
        raise

def init_db():
    """Initializes the database schema if it doesn't already exist."""
    logger.info(f"Initializing database at {DB_PATH} if not already set up.")
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()

            # Table for processed files
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL UNIQUE, /* SHA256 hash of file content */
                    collection_name TEXT NOT NULL UNIQUE, /* ChromaDB collection name */
                    total_chunks INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.debug("Table 'processed_files' checked/created.")

            # Table for chat messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL, /* Foreign key to processed_files */
                    role TEXT NOT NULL, /* 'user' or 'assistant' */
                    content TEXT NOT NULL,
                    sources_json TEXT, /* JSON string of source document metadata */
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES processed_files (id)
                )
            """)
            logger.debug("Table 'chat_messages' checked/created.")

            conn.commit()
            logger.info("Database initialization complete.")
    except sqlite3.Error as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        # It's critical if DB can't be initialized, so re-raise
        raise RuntimeError("Failed to initialize the application database.") from e

def add_processed_file(file_name: str, file_hash: str, collection_name: str, total_chunks: int) -> Optional[int]:
    """Adds a record of a processed file to the database."""
    logger.info(f"Adding processed file to DB: {file_name}, hash: {file_hash[:8]}..., collection: {collection_name}")
    sql = """
        INSERT INTO processed_files (file_name, file_hash, collection_name, total_chunks)
        VALUES (?, ?, ?, ?)
    """
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_name, file_hash, collection_name, total_chunks))
            conn.commit()
            file_id = cursor.lastrowid
            logger.info(f"File '{file_name}' added to DB with ID: {file_id}")
            return file_id
    except sqlite3.IntegrityError: # Handles UNIQUE constraint violations
        logger.warning(f"File '{file_name}' or hash '{file_hash[:8]}' or collection '{collection_name}' likely already exists in DB.")
        # Retrieve the existing file_id if it already exists based on hash or name
        return get_processed_file_id_by_hash(file_hash) or get_processed_file_id_by_name(file_name)
    except sqlite3.Error as e:
        logger.error(f"Error adding processed file '{file_name}' to DB: {e}", exc_info=True)
        return None

def get_processed_file_id_by_hash(file_hash: str) -> Optional[int]:
    """Retrieves the ID of a processed file by its hash."""
    sql = "SELECT id FROM processed_files WHERE file_hash = ?"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_hash,))
            row = cursor.fetchone()
            return row['id'] if row else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching file by hash '{file_hash[:8]}': {e}", exc_info=True)
        return None

def get_processed_file_id_by_name(file_name: str) -> Optional[int]:
    """Retrieves the ID of a processed file by its name."""
    sql = "SELECT id FROM processed_files WHERE file_name = ?"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_name,))
            row = cursor.fetchone()
            return row['id'] if row else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching file by name '{file_name}': {e}", exc_info=True)
        return None


def get_all_processed_files() -> List[Dict[str, Any]]:
    """Retrieves all processed files from the database."""
    logger.debug("Fetching all processed files from DB.")
    sql = "SELECT id, file_name, collection_name, processed_at, total_chunks FROM processed_files ORDER BY processed_at DESC"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            # Convert rows to dictionaries for easier use
            files = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Retrieved {len(files)} processed files.")
            return files
    except sqlite3.Error as e:
        logger.error(f"Error fetching all processed files: {e}", exc_info=True)
        return []

def get_processed_file_details(file_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves details for a specific processed file by its ID."""
    logger.debug(f"Fetching details for processed file ID: {file_id}")
    sql = "SELECT id, file_name, file_hash, collection_name, total_chunks, processed_at FROM processed_files WHERE id = ?"
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            logger.warning(f"No processed file found with ID: {file_id}")
            return None
    except sqlite3.Error as e:
        logger.error(f"Error fetching processed file ID {file_id}: {e}", exc_info=True)
        return None

def add_chat_message(file_id: int, role: str, content: str, sources: Optional[List[Dict]] = None):
    """Adds a chat message to the database for a given file_id."""
    logger.debug(f"Adding chat message to DB for file_id {file_id}, role: {role}")
    sources_json = json.dumps(sources) if sources else None
    sql = """
        INSERT INTO chat_messages (file_id, role, content, sources_json)
        VALUES (?, ?, ?, ?)
    """
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_id, role, content, sources_json))
            conn.commit()
            logger.debug(f"Chat message added for file_id {file_id}, role {role}.")
    except sqlite3.Error as e:
        logger.error(f"Error adding chat message for file_id {file_id}: {e}", exc_info=True)

def get_chat_history(file_id: int) -> List[Dict[str, Any]]:
    """Retrieves chat history for a given file_id."""
    logger.debug(f"Fetching chat history for file_id {file_id}.")
    sql = "SELECT role, content, sources_json, timestamp FROM chat_messages WHERE file_id = ? ORDER BY timestamp ASC"
    messages: List[Dict[str, Any]] = []
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (file_id,))
            for row in cursor.fetchall():
                message = dict(row)
                if message['sources_json']:
                    try:
                        message['sources'] = json.loads(message['sources_json'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode sources_json for a message in file_id {file_id}")
                        message['sources'] = None # Or some error indicator
                else:
                    message['sources'] = None
                del message['sources_json'] # Remove raw JSON string
                messages.append(message)
            logger.debug(f"Retrieved {len(messages)} chat messages for file_id {file_id}.")
            return messages
    except sqlite3.Error as e:
        logger.error(f"Error fetching chat history for file_id {file_id}: {e}", exc_info=True)
        return []

# Call init_db() when the module is first imported to ensure tables exist.
# This is a simple way to handle initialization. For more complex apps,
# you might have an explicit setup step.
# init_db() # Be cautious with calling this at import time in some frameworks, 
            # for Streamlit this is generally okay for simple cases.
            # We'll call it explicitly from the app setup.

if __name__ == '__main__':
    print("Running History Manager Tests...")
    init_db() # Ensure DB is set up for testing

    # Test adding a file
    print("\nTesting add_processed_file...")
    test_file_id = add_processed_file("test_paper.pdf", "testhash123", "collection_test_paper", 120)
    if test_file_id:
        print(f"Added/found test_paper.pdf with ID: {test_file_id}")

        # Test retrieving the file
        print("\nTesting get_all_processed_files...")
        all_files = get_all_processed_files()
        print(f"All processed files: {all_files}")
        assert any(f['file_name'] == "test_paper.pdf" for f in all_files)

        print("\nTesting get_processed_file_details...")
        details = get_processed_file_details(test_file_id)
        print(f"Details for ID {test_file_id}: {details}")
        assert details is not None and details['file_name'] == "test_paper.pdf"

        # Test adding chat messages
        print("\nTesting add_chat_message...")
        add_chat_message(test_file_id, "user", "What is this paper about?")
        source_example = [{"page_content": "This is context from page 1.", "metadata": {"source": "test_paper.pdf", "page": 1}}]
        add_chat_message(test_file_id, "assistant", "It's about AI.", sources=source_example)

        # Test retrieving chat history
        print("\nTesting get_chat_history...")
        history = get_chat_history(test_file_id)
        print(f"Chat history for file ID {test_file_id}:")
        for msg in history:
            print(msg)
        assert len(history) >= 2
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'
        assert history[1]['sources'] is not None

        print("\nHistory Manager tests completed.")
    else:
        print("Failed to add test_paper.pdf for testing.")