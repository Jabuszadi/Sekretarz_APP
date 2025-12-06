import agent_db
from summarizer import generate_minutes_of_meeting
from qdrant_handler import get_qdrant_client, get_embedding_model
from utils import save_minutes_to_file, get_relevant_date_for_file
from models import MinutesResponse, ProcessingResult
from datetime import datetime
import logging
import torch
from typing import Optional, Tuple, List, Callable, Dict
from pathlib import Path
from config import QDRANT_ENDPOINT, QDRANT_API_KEY, EMBEDDING_MODEL_NAME, QDRANT_SEARCH_RESULTS_LIMIT
import config
from langchain_core.documents import Document
import json
import asyncio # DODANY IMPORT: Do obsługi asynchroniczności
from starlette.concurrency import run_in_threadpool

# qdrant_client = AsyncQdrantClient(url=config.QDRANT_ENDPOINT, api_key=config.QDRANT_API_KEY)

async def generate_and_save_minutes(
    collection_name: str,
    original_source_path: Optional[Path],
    custom_prompt: Optional[str] = None,
    output_name: Optional[str] = None,
    progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    file_id: Optional[int] = None,
    target_date_override: Optional[str] = None,
    reuse_minutes_id: Optional[int] = None,
    gemini_api_key: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[MinutesResponse], Optional[str], Optional[int]]:
    metadata = metadata or {}

    """Generate and save meeting minutes."""
    target_date: Optional[str] = None
    if original_source_path:
        try:
            exists = await asyncio.to_thread(original_source_path.exists)
            if exists:
                target_date = await get_relevant_date_for_file(original_source_path)
        except Exception as date_error:  # pragma: no cover - diagnostyczne
            logging.warning("Could not determine target date from source file %s: %s", original_source_path, date_error)
    if not target_date and target_date_override:
        target_date = target_date_override
    logging.info(f"🔍 Determining target date for filtering: {target_date}")

    relevant_docs = []
    minutes_id = None

    try:
        # Pobierz zainicjalizowane obiekty za pomocą funkcji getterów
        current_qdrant_client = get_qdrant_client()
        current_embedding_model = get_embedding_model()

        if current_qdrant_client is None:
            logging.error("Qdrant client not initialized for minutes generation!")
            raise RuntimeError("Qdrant client is not initialized.")
        if current_embedding_model is None:
            logging.error("Embedding model not loaded for minutes generation!")
            raise RuntimeError("Embedding model is not loaded.")

        query_vector = await asyncio.to_thread(current_embedding_model.embed_query, "meeting minutes summary") # Użyj pobranego modelu

        # Zawsze wykonuj wyszukiwanie w Qdrant BEZ FILTRA DATY
        logging.info("Searching Qdrant for relevant documents without date filter (will filter in Python)...\n")
        hits = await current_qdrant_client.search( # Użyj pobranego klienta
            collection_name=collection_name,
            query_vector=query_vector,
            limit=config.QDRANT_SEARCH_RESULTS_LIMIT * 5 # Pobierz więcej, jeśli będziesz filtrować!
                                                      # Np. 5x więcej niż ostateczny limit
        )

        # Przekonwertuj wyniki na format LangChain Document
        all_retrieved_docs = []
        for hit in hits:
            # Upewnij się, że 'text' jest kluczem do treści dokumentu
            doc_content = hit.payload.get("text", "")
            if not doc_content: # Fallback, jeśli 'text' nie istnieje lub jest puste
                doc_content = hit.payload.get("page_content", "")
                
            doc = Document(page_content=doc_content, metadata=hit.payload)
            all_retrieved_docs.append(doc)
            logging.debug(f"DEBUG: Document created with metadata: {doc.metadata}")

        logging.debug(f"Target date for filtering: {target_date}")

        # Filter documents by date if a target_date is provided and valid
        if target_date:
            temp_relevant_docs = []
            for doc in all_retrieved_docs:
                doc_metadata_field = doc.metadata.get("metadata", {})
                doc_date = doc_metadata_field.get("date")
                if doc_date == target_date:
                    temp_relevant_docs.append(doc)
            relevant_docs = temp_relevant_docs
            if not relevant_docs:
                logging.warning(
                    f"No relevant documents found in Qdrant collection '{collection_name}' for date '{target_date}'. Cannot generate minutes."
                )
                return None, "No relevant documents found.", None

        else:
            relevant_docs = all_retrieved_docs
            logging.warning("No target date provided or valid, using all retrieved documents for summarization.")

        if not relevant_docs:
            logging.warning(f"No relevant documents found in Qdrant collection '{collection_name}' for date '{target_date}'. Cannot generate minutes.")
            return None, "No relevant documents found.", None

        # Aggregate content for summarization
        context = " ".join([doc.page_content for doc in relevant_docs])
        logging.info(f"Aggregated {len(relevant_docs)} documents into context of {len(context.split())} words for summarization.")

        # Generuj minuty
        logging.info("Generating meeting minutes summary with LLM...")
        minutes_content_dict = await generate_minutes_of_meeting(
            context,
            custom_prompt,
            progress_callback,
            api_key_override=gemini_api_key,
        )
        logging.info("LLM summarization complete.")

        # Extract metadata
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        llm_model = metadata.get("llm_model") or config.GEMINI_MODEL_NAME
        embeddings_model = metadata.get("embeddings_model") or config.EMBEDDING_MODEL_NAME
        chunking_method = metadata.get("chunking_method") or f"segmenty {config.CHUNK_DURATION}s z nałożeniem {config.CHUNK_OVERLAP}s"
        transcription_model_display = (
            metadata.get("transcription_model_display")
            or metadata.get("transcription_model")
            or config.WHISPER_MODEL_SIZE
        )
        diarization_model_display = metadata.get("diarization_model") or config.PYANNOTE_PIPELINE

        # Formatuj minuty (dla pliku tekstowego)
        final_minutes_text = "Raport ze Spotkania\n\n"
        for section, content in minutes_content_dict.items():
            final_minutes_text += f"**{section}:**\n{content}\n\n"
        final_minutes_text += f"\n---\nWygenerowano: {generated_at}\n"
        final_minutes_text += f"Model LLM: {llm_model}\n"
        final_minutes_text += f"Model Embeddings: {embeddings_model}\n"
        final_minutes_text += f"Metoda chunkowania: {chunking_method}\n"
        final_minutes_text += f"Model transkrypcji: {transcription_model_display}\n"
        final_minutes_text += f"Model diarizacji: {diarization_model_display}\n"

        try:
            # Zapisz minuty do pliku
            saved_file_path = await save_minutes_to_file(final_minutes_text, collection_name, target_date)
            
            if saved_file_path is None:
                logging.error(f"❌ Failed to save minutes to file. save_minutes_to_file returned None.")
                return None, "Failed to save minutes file.", None

            # NEW: Add minutes to database
            minutes_id = await run_in_threadpool(
                agent_db.add_meeting_minutes_with_update,
                summary_text=final_minutes_text,
                minutes_path=str(saved_file_path),
                generated_at=generated_at,
                llm_model=llm_model,
                embeddings_model=embeddings_model,
                chunking_method=chunking_method,
                transcription_model=transcription_model_display,
                diarization_model=diarization_model_display,
                meeting_date_str=target_date,
                existing_minutes_id=None,
                file_id=file_id,
            )
            logging.info(f"✅ Meeting minutes added to database with ID: {minutes_id}")

            return MinutesResponse(minutes=final_minutes_text), saved_file_path, minutes_id
        except IOError as e:
            logging.error(f"❌ Error saving minutes or metadata file: {e}")
            return None, f"Error saving file: {e}", None
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during minutes generation: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}", None