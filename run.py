# run.py (Modified for Setup Only)
import sys
import os
from pathlib import Path
import logging # Use standard logging if setup fails

# Ensure the 'app' directory is in the Python path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

# Import setup functions
try:
    from app.utils.logging_config import setup_logging
    from app.services.history_manager import init_db
    SETUP_IMPORTS_SUCCESS = True
except ImportError as e:
    logging.error(f"Failed to import setup modules: {e}", exc_info=True)
    SETUP_IMPORTS_SUCCESS = False


def perform_setup():
    """Performs initial application setup (logging and DB init)."""
    print("--- Performing Initial Setup ---")

    # 1. Setup Logging
    try:
        setup_logging()
        # Get logger instance after setup
        logger = logging.getLogger(__name__) 
        logger.info("Logging configured successfully via run.py.")
    except Exception as e:
        logging.error(f"Critical error during logging setup: {e}", exc_info=True)
        print(f"ERROR: Logging setup failed: {e}")
        # Decide if you want to exit or try to continue
        # sys.exit("Failed to initialize logging. Setup aborted.")

    # 2. Initialize Database
    try:
        init_db()
        # Use logger if available
        log_func = logger.info if 'logger' in locals() and logger else print
        log_func("Database initialization check complete via run.py.")
    except Exception as e:
        log_func = logger.error if 'logger' in locals() and logger else print
        log_func(f"Critical error during database initialization: {e}")
        print(f"ERROR: Database initialization failed: {e}")
        sys.exit("Failed to initialize database. Setup aborted.")

    print("--- Initial Setup Complete ---")
    print("\nTo run the application, use the following command:")
    print("streamlit run app/ui/streamlit_app.py")


if __name__ == "__main__":
    if SETUP_IMPORTS_SUCCESS:
        perform_setup()
    else:
        print("ERROR: Could not import necessary setup modules. Please check file structure and imports.")
        print("Aborting setup.")
        sys.exit(1)