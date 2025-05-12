**Phase 5: Orchestration, Execution & MLOps Considerations**

This phase involves:

1.  **Testing the Full Workflow:** Running the app and testing the end-to-end process (upload, process, select, chat).
2.  **Identifying Potential Issues & Edge Cases:** Thinking about what could go wrong or where the current implementation might fall short.
3.  **MLOps Lite & Best Practices:** Discussing how to maintain and improve the project using good software engineering and MLOps principles, even for a local application.
4.  **Future Enhancements:** Brainstorming ways to make the application more powerful and robust.

Let's break these down:

**1. Testing the Full Workflow**

We've already started doing this while debugging. The key steps to test systematically are:

* **Upload & Process:** Does uploading a *new* PDF work correctly? Does it appear in the dropdown? Are logs showing successful parsing, chunking, embedding, and DB insertion?
* **Re-Upload:** Does uploading the *same* PDF correctly identify it as already processed?
* **Selection:** Can you select different processed PDFs from the dropdown? Does the chat history and context switch correctly?
* **Chatting:**
    * Ask relevant questions. Are the answers sensible and seemingly derived from the document?
    * Check the "View Sources" expander. Are the retrieved chunks relevant?
    * Ask irrelevant questions. Does the bot correctly state it doesn't know?
* **Persistence:** Stop the Streamlit app (`Ctrl+C` in the terminal) and restart it (`python run.py` then `streamlit run app/ui/streamlit_app.py`). Are the previously processed files still listed in the dropdown? Is the chat history for a selected document retained?

**2. Potential Issues & Edge Cases (Recap & Expansion)**

* **PDF Quality:** Scanned PDFs (no selectable text), complex layouts (multi-column, tables crossing pages), mathematical formulas, figures/images. *Current Limitation: Best for single-column, text-based PDFs.*
* **Chunking:** Optimal size/overlap is document-dependent. Recursive splitting might break logical sections (e.g., separating a heading from its paragraph). *Mitigation: Experiment with settings, consider layout-aware or semantic chunking later.*
* **Embedding Quality:** The chosen model (`all-MiniLM-L6-v2`) is good general-purpose, but might not be optimal for highly specialized scientific domains. *Mitigation: Experiment with other embedding models (e.g., BAAI/bge-..., domain-specific models).*
* **Retrieval:** Sometimes the most semantically similar chunk according to embeddings isn't the *best* answer context. *Mitigation: Re-ranking, query transformation (advanced).*
* **LLM Performance:** Slow inference, high resource usage (RAM). *Mitigation: Quantized models, GPU offloading (requires setup).*
* **LLM Accuracy/Hallucination:** LLM might still occasionally generate plausible but incorrect info, or fail to follow instructions perfectly. *Mitigation: Prompt engineering, ensuring good context retrieval.*
* **Scalability (Local):** Processing very large PDFs (> hundreds of pages) might strain memory or take a long time. ChromaDB is efficient locally but might slow down with millions of vectors (unlikely for single-user research paper use).
* **Error Handling:** Ensure errors during processing (e.g., corrupted PDF, failed LLM load) are handled gracefully and reported clearly to the user.

**3. MLOps Lite & Best Practices**

* **Version Control (Git):** Continue using it diligently. Commit frequently with clear messages. Use branches for new features.
* **Configuration (`config/settings.py`):** Keep all tunable parameters here. Avoid hardcoding paths or model names in the main code.
* **Logging (`logs/app.log`):** Use logs extensively to diagnose issues. Add more detailed logging if needed during debugging.
* **Dependency Management (`requirements.txt`):** Keep it updated (`pip freeze > requirements.txt`).
* **Testing:** Add automated tests (`pytest`) for core functions (parsing, chunking, maybe basic RAG response checks with mock data) to catch regressions when you make changes.
* **Code Quality:** Consider using linters (`flake8`, `ruff`) and formatters (`black`) to maintain clean, consistent code.
* **Modularity:** Stick to the modular design. If adding a new feature (like OCR), create new functions/modules rather than mixing logic into existing ones.

**4. Future Enhancements**

* **Advanced Chunking/Parsing:** Integrate layout-aware parsing (`unstructured.io`) or semantic chunking. Add OCR (`pytesseract`).
* **Advanced Retrieval:** Implement re-ranking or query transformations.
* **Multi-Document Chat:** Allow selecting multiple PDFs and asking questions across them (requires changes to how retrievers are combined).
* **Improved UI:** Streaming responses, source highlighting, feedback buttons.
* **Different Models:** Allow selecting different embedding or LLM models via the config or UI.
* **GPU Acceleration:** Configure `llama-cpp-python` for GPU offloading if hardware allows.
* **Evaluation:** Integrate a framework (RAGAs, TruLens) to systematically measure performance.
* **Containerization (Docker):** Package the app for easier deployment/sharing.

This phase is about consolidating your understanding, ensuring the application works reliably, and planning for its future. We've built a complex system with many moving parts!

