from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from models import ProcessingRequest, ProcessingResult
from file_handlers import cleanup_temp_dir, job_temp_storage, save_uploaded_file_temp, save_file_to_temp_and_convert_if_needed
from processing_service import process_audio
from ingest import ingest_transcription
from minutes_service import generate_and_save_minutes
import config

# Import StreamingResponse for SSE
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import asyncio # Needed for async generators
import uuid
import logging
from pathlib import Path
import json # Moved import json to the top
from typing import List, Optional, Tuple, Dict, Any # Dodaj Dict i Any
import torch  # Dodaj na górze pliku
import time
import httpx
import threading
import queue
from qdrant_handler import initialize_qdrant_resources
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from datetime import datetime
import shutil # NOWY IMPORT: Do czyszczenia tymczasowych plików/katalogów
import torchaudio # NOWY IMPORT: Do ładowania i zapisywania audio segmentów
from pyannote.core import Segment # NOWY IMPORT: Do pracy z segmentami diarization
from starlette.concurrency import run_in_threadpool # NOWY IMPORT: Do obsługi blokujących operacji I/O

from processor import process_mkv_files # Upewnij się, że to jest zaimportowane
from summarizer import generate_minutes_of_meeting, gemini_model # Upewnij się, że to jest zaimportowane, DODANO gemini_model
from utils import save_minutes_to_file, load_prompt_config, save_prompt_config, delete_prompt_config, get_file_creation_date, get_relevant_date_for_file, extract_date_from_filename
from utils import enroll_speaker_from_audio, load_enrolled_speakers, _enrolled_speakers_profiles, get_diarization_models, get_current_ram_usage_gb, get_current_vram_usage_gb # DODANY IMPORT
import tempfile # NOWY IMPORT: Do bezpośredniego zapisu plików audio
import agent_db # NEW: Import agent_db

# NOWE IMPORTY DLA FUNKCJONALNOŚCI CZATU
from qdrant_handler import search_all_collections, get_all_collection_names, initialize_qdrant_resources # DODANO
from typing import List # Upewnij się, że jest zaimportowane na górze, ale dodaj na wszelki wypadek
import asyncio # Upewnij się, że jest zaimportowane na górze, ale dodaj na wszelki wypadek

# Ustawienie podstawowej konfiguracji logowania
# Zmieniono poziom logowania z DEBUG na INFO, aby zmniejszyć gadatliwość
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Opcjonalnie: stłumienie logów DEBUG dla bibliotek, które generują dużo szumu
# httpx jest używany przez klienta Qdrant i potrafi być bardzo gadatliwy
logging.getLogger("httpx").setLevel(logging.WARNING)
# Opcjonalnie: stłumienie logów DEBUG dla bibliotek, które generują dużo szumu
# httpx jest używany przez klienta Qdrant i potrafi być bardzo gadatliwy
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # Nowy log dla httpcore
logging.getLogger("matplotlib").setLevel(logging.WARNING) # Nowy log dla matplotlib
logging.getLogger("speechbrain").setLevel(logging.WARNING) # Nowy log dla speechbrain
logging.getLogger("torchaudio").setLevel(logging.WARNING) # Nowy log dla torchaudio

app = FastAPI(
    title="Audio/Video Transcription & Summarization API",
    description="API to process audio/video files and generate meeting minutes.",
    version="1.0.0",
)

# Modele Pydantic dla zapytania i odpowiedzi czatu (PRZENIESIONE Z CHAT_AGENT_APP.PY)
class ChatQuery(BaseModel):
    query: str
    collection_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

# Funkcja do generowania odpowiedzi LLM na podstawie kontekstu i zapytania (PRZENIESIONA Z CHAT_AGENT_APP.PY)
async def generate_chat_response(user_query: str, context_documents: List[str]) -> str:
    """
    Generuje odpowiedź Agenta na podstawie zapytania użytkownika i dostarczonych dokumentów kontekstowych.
    """
    if not gemini_model:
        logging.error("Gemini model not initialized. Cannot generate chat response.")
        return "Przepraszam, model AI nie jest dostępny. Spróbuj ponownie później."

    context_str = "\n".join(context_documents)
    if not context_str:
        return "Nie znalazłem żadnych informacji w bazie danych, które odpowiadałyby na Twoje pytanie."

    prompt = f"""
Jesteś pomocnym asystentem AI. Twoim zadaniem jest odpowiadanie na pytania użytkownika,
wykorzystując *wyłącznie* informacje zawarte w podanym fragmencie tekstu.
Jeśli informacja nie znajduje się w tekście, odpowiedz, że nie możesz znaleźć odpowiedzi na podstawie dostępnych danych.
Nie wymyślaj informacji. Odpowiadaj zwięźle i na temat.

**Zapytanie Użytkownika:**
{user_query}

**Dostępny Kontekst (fragmenty transkrypcji/protokołów):**
---
{context_str}
---

**Twoja Odpowiedź (oparta wyłącznie na Kontekście):**
"""
    try:
        response = await asyncio.to_thread(gemini_model.invoke, prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error generating chat response with Gemini: {e}")
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."


# NEW: Mount a static directory to serve index.html (Już jest)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# NOWY ENDPOINT: Do serwowania interfejsu czatu
@app.get("/chat_ui", response_class=HTMLResponse)
async def serve_chat_interface():
    html_file_path = Path("chat_interface.html")
    if not html_file_path.exists():
        raise HTTPException(status_code=404, detail="Chat interface HTML file not found.")
    with open(html_file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


# NOWY ENDPOINT: Przeniesiony z chat_agent_app.py
@app.post("/chat/query", response_model=ChatResponse)
async def chat_query_endpoint(chat_query: ChatQuery):
    """
    Endpoint do wysyłania zapytań do Agenta Czatowego.
    """
    user_query = chat_query.query
    collection_name = chat_query.collection_name
    logging.info(f"Received chat query: '{user_query}' for collection: {collection_name}")

    try:
        relevant_documents = await search_all_collections(
            user_query,
            limit_per_collection=config.QDRANT_SEARCH_LIMIT_PER_COLLECTION,
            total_limit=config.QDRANT_SEARCH_TOTAL_LIMIT,
            target_collection=collection_name
        )
        
        agent_response = await generate_chat_response(user_query, [doc.content for doc in relevant_documents])
        
        return ChatResponse(response=agent_response)

    except Exception as e:
        logging.error(f"Error in chat_query_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas przetwarzania Twojego zapytania: {e}")

# NOWY ENDPOINT: Przeniesiony z chat_agent_app.py
@app.get("/get_qdrant_collections/")
async def get_qdrant_collections_endpoint():
    """Endpoint do pobierania listy nazw kolekcji Qdrant."""
    try:
        # Zapewnij inicjalizację Qdrant przed próbą pobrania kolekcji
        await initialize_qdrant_resources() # Upewnij się, że klient Qdrant jest zainicjalizowany
        collection_names = await get_all_collection_names()
        return JSONResponse(content=collection_names)
    except Exception as e:
        logging.error(f"Error getting Qdrant collection names: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Qdrant collections: {e}")

@app.post("/process_file/", response_model=ProcessingResult)
async def process_uploaded_file(
    file: UploadFile = File(...),
    whisper_model_size: str = Form(config.WHISPER_MODEL_SIZE),
    output_name: Optional[str] = Form(None),
):
    file_id = None # Initialize file_id
    file_hash = None # Initialize file_hash
    try:
        job_id = str(uuid.uuid4())
        job_temp_storage[job_id] = {'type': 'file', 'batch_job_id': None, 'status': 'handling_upload'}
        logging.info(f"[{job_id}] Received file upload request for '{file.filename}'.")

        input_file_temp_path = await save_file_to_temp_and_convert_if_needed(file, job_id)
        if not input_file_temp_path:
            logging.error(f"[{job_id}] File upload or conversion failed for '{file.filename}'.")
            raise HTTPException(status_code=500, detail="File upload failed.")
        logging.info(f"[{job_id}] File '{file.filename}' saved to temporary path: {input_file_temp_path}")

        # NEW: Compute file hash and add to processed_files table
        try:
            file_hash = await run_in_threadpool(agent_db.compute_file_hash, input_file_temp_path) # Użyj run_in_threadpool
            logging.info(f"[{job_id}] Computed file hash: {file_hash}")
            file_id = await run_in_threadpool(agent_db.add_file, file.filename, str(input_file_temp_path), file_hash, "processing", "") # Użyj run_in_threadpool
            logging.info(f"[{job_id}] File '{file.filename}' added to processed_files with ID: {file_id}")
        except Exception as e:
            logging.error(f"[{job_id}] Error adding file record to processed_files table: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database error during file record creation: {e}")


        logging.info(f"[{job_id}] Starting audio processing for '{file.filename}'...")
        transcription_segments, transcription_filename, transcript_id = await process_audio(input_file_temp_path, job_id, whisper_model_size) # NEW: Capture transcript_id
        if not transcription_segments:
            logging.error(f"[{job_id}] Audio processing failed or produced no segments for '{file.filename}'.")
            if file_hash:
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Audio processing failed") # Użyj run_in_threadpool
            raise HTTPException(status_code=500, detail="Audio processing failed.")
        logging.info(f"[{job_id}] Audio processing complete. Transcript ID: {transcript_id}")

        # DODAJ: Zapis segmentów do bazy
        if transcript_id is not None and transcription_segments:
            for seg in transcription_segments:
                agent_db.add_transcript_segment(
                    transcript_id,
                    seg.start,
                    seg.end,
                    seg.speaker,
                    seg.text
                )
            logging.info(f"[{job_id}] Added {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")

        logging.info(f"[{job_id}] Starting ingestion to Qdrant for '{file.filename}'...")
        collection_name_for_file = await ingest_transcription(
            transcription_segments,
            input_file_temp_path,
            job_id,
            transcript_id, # Dodano transcript_id
            config.CHUNK_DURATION, # Teraz te argumenty są poprawne
            config.CHUNK_OVERLAP   # względem definicji
        )
        if not collection_name_for_file:
            logging.error(f"[{job_id}] Ingestion to Qdrant failed for '{file.filename}'.")
            if file_hash:
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Ingestion to Qdrant failed") # Użyj run_in_threadpool
            collection_name = None
            raise HTTPException(status_code=500, detail="Ingestion to Qdrant failed.")
        else:
            collection_name = collection_name_for_file
        logging.info(f"[{job_id}] Ingested to Qdrant collection: {collection_name}")

        logging.info(f"[{job_id}] Generating and saving minutes for '{file.filename}'...")
        # Po transkrypcji, wygeneruj i zapisz minuty ze spotkania
        # generate_and_save_minutes zwraca teraz MinutesResponse, filename, i minutes_db_id
        minutes_content_response, minutes_filename, minutes_db_id = await generate_and_save_minutes( # <-- DODANO minutes_db_id
            collection_name=collection_name,
            original_source_path=Path(input_file_temp_path),
            custom_prompt=None,
            output_name=output_name
        )

        if minutes_content_response:
            logging.info(f"[{job_id}] Minutes generated successfully for '{file.filename}'. Minutes ID: {minutes_db_id}")
            if file_hash:
                await run_in_threadpool(agent_db.update_file_status, file_hash, "completed", "Processing successful") # Użyj run_in_threadpool
            # ... (usunięto duplikację kodu zwracającego ProcessingResult, zostawiając jedną wersję na końcu) ...
        else:
            logging.error(f"[{job_id}] Minutes generation failed for '{file.filename}'.")
            if file_hash:
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Minutes generation failed") # Użyj run_in_threadpool
            # minutes_content = None # Te linie są redundantne, jeśli rzucamy wyjątek
            # minutes_filename = None
            minutes_db_id = None # NEW: Ensure minutes_id is None on failure
            raise HTTPException(status_code=500, detail="Minutes generation failed.")

        # NEW: Add entry to meeting_data table
        try:
            meeting_date = get_relevant_date_for_file(input_file_temp_path)
            if file_id is not None and transcript_id is not None and minutes_db_id is not None and collection_name is not None:
                await run_in_threadpool(agent_db.add_meeting_data, file_id, transcript_id, minutes_db_id, meeting_date, collection_name) # Użyj run_in_threadpool
                logging.info(f"[{job_id}] Meeting data added to database for file_id: {file_id}")
            else:
                missing_info = []
                if file_id is None: missing_info.append("File ID")
                if transcript_id is None: missing_info.append("Transcript ID")
                if minutes_db_id is None: missing_info.append("Minutes ID")
                if collection_name is None: missing_info.append("Collection Name")
                logging.warning(f"[{job_id}] Could not add meeting data to database due to missing information: {', '.join(missing_info)}")
                if file_hash: # Jeśli brakuje danych do meeting_data, oznacz jako błąd
                    await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Incomplete data for meeting_data entry") # Użyj run_in_threadpool
                raise ValueError("Incomplete data for meeting_data entry.")
        except Exception as e:
            logging.error(f"[{job_id}] Error adding meeting data to database: {e}", exc_info=True)
            if file_hash:
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", f"Database error during meeting data creation: {e}") # Użyj run_in_threadpool
            raise HTTPException(status_code=500, detail=f"Database error during meeting data creation: {e}")

        # Finalna aktualizacja statusu i zwrócenie wyniku po pomyślnym zakończeniu wszystkich operacji
        if file_hash: # Upewnij się, że file_hash jest dostępne
            await run_in_threadpool(agent_db.update_file_status, file_hash, "completed", "Processing successful") # Użyj run_in_threadpool
            logging.info(f"[{job_id}] File '{file.filename}' processing completed successfully. Status updated to 'completed'.")
        
        return ProcessingResult(
            status="success",
            message="File processed successfully",
            transcription=transcription_segments,
            minutes=minutes_content_response.minutes, # Użyj minutes_content_response.minutes
            unique_collection_name=collection_name,
            minutes_filename=minutes_filename,
            transcription_filename=transcription_filename
        )
    except HTTPException as he:
        logging.error(f"[{job_id}] HTTPException during processing: {he.detail}", exc_info=True)
        if file_id and file_hash: # Upewnij się, że file_id i file_hash są dostępne
            current_status = await run_in_threadpool(agent_db.get_file_status, file_hash) # Użyj run_in_threadpool
            if current_status != "completed": # Zmieniono na get_file_status(file_hash)
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", f"HTTP Exception: {he.detail}") # Użyj run_in_threadpool
        raise he
    except Exception as e:
        logging.error(f"[{job_id}] Unhandled Exception during processing: {e}", exc_info=True)
        # Import traceback inside the except block to avoid unused import warnings if not needed
        import traceback
        traceback.print_exc()
        if file_id and file_hash: # Upewnij się, że file_id i file_hash są dostępne
            current_status = await run_in_threadpool(agent_db.get_file_status, file_hash) # Użyj run_in_threadpool
            if current_status != "completed": # Zmieniono na get_file_status(file_hash)
                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", f"Unhandled Exception: {str(e)}") # Użyj run_in_threadpool
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Dodano: Logowanie zużycia RAM po zakończeniu przetwarzania pojedynczego pliku
        ram_gb = get_current_ram_usage_gb()
        if ram_gb > 0:
            logging.info(f"[{job_id}] Final RAM usage after processing single file: {ram_gb:.2f} GB")

        current_vram_usage = get_current_vram_usage_gb()
        if current_vram_usage > 0:
            logging.info(f"[{job_id}] Final VRAM usage after processing single file: {current_vram_usage:.2f} GB")
        else:
            logging.warning(f"[{job_id}] VRAM usage could not be retrieved after processing single file. See warnings above.")


@app.post("/upload_multiple/")
async def upload_multiple_files_for_processing(
    files: List[UploadFile] = File(...),
    whisper_model_size: str = Form(config.WHISPER_MODEL_SIZE),
    custom_prompt: Optional[str] = Form(None),
    chunk_duration: int = Form(config.CHUNK_DURATION),
    chunk_overlap: int = Form(config.CHUNK_OVERLAP),
    output_name: Optional[str] = Form(None),
):
    batch_job_id = str(uuid.uuid4())
    job_temp_storage[batch_job_id] = {
        'type': 'batch',
        'file_job_ids': [],
        'status': 'uploaded',
        'params': {
            'whisper_model_size': whisper_model_size,
            'custom_prompt': custom_prompt,
            'chunk_duration': chunk_duration,
            'chunk_overlap': chunk_overlap,
            'output_name': output_name
        }
    }
    logging.info(f"DEBUG: Batch job {batch_job_id} added to job_temp_storage at /upload_multiple/: {list(job_temp_storage.keys())}")
    print(f"Received upload request for batch job: {batch_job_id}")

    uploaded_file_info = []

    try:
        for file in files:
            file_job_id = str(uuid.uuid4())
            job_temp_storage[batch_job_id]['file_job_ids'].append(file_job_id)
            job_temp_storage[file_job_id] = {'type': 'file', 'batch_job_id': batch_job_id, 'status': 'uploaded'}

            print(f"  Processing file '{file.filename}' with file_job_id: {file_job_id}")

            temp_file_to_process_path = await save_file_to_temp_and_convert_if_needed(file, file_job_id)

            job_temp_storage[file_job_id]['params'] = {
                 'whisper_model_size': whisper_model_size,
                 'custom_prompt': custom_prompt,
                 'chunk_duration': chunk_duration,
                 'chunk_overlap': chunk_overlap,
                 'output_name': output_name
            }
            job_temp_storage[file_job_id]['file_path'] = temp_file_to_process_path

            uploaded_file_info.append({
                "filename": file.filename,
                "file_job_id": file_job_id,
                "temp_path": str(temp_file_to_process_path)
            })

            print(f"  File '{file.filename}' saved and job {file_job_id} details stored.")

        return JSONResponse(
            status_code=202,
            content={
                "message": f"Received {len(files)} files, ready for processing batch.",
                "batch_job_id": batch_job_id,
                "uploaded_files": uploaded_file_info
            }
        )
        logging.info(f"DEBUG: Returning 202 Accepted for batch job {batch_job_id} from /upload_multiple/. Current job_temp_storage keys: {list(job_temp_storage.keys())}")

    except Exception as e:
        print(f"Error during upload for batch job {batch_job_id}: {str(e)}")
        if batch_job_id in job_temp_storage:
            for fj_id in job_temp_storage[batch_job_id].get('file_job_ids', []):
                 cleanup_temp_dir(fj_id)
            if batch_job_id in job_temp_storage:
                del job_temp_storage[batch_job_id]

        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Multiple file upload or initial processing failed for batch {batch_job_id}: {str(e)}"
        )

# Modified endpoint for streaming status updates, now taking batch_job_id
# This endpoint will now orchestrate the processing of multiple files
@app.get("/process_batch_status/stream/{batch_job_id}/")
async def process_batch_status_stream(batch_job_id: str):
    logging.info(f"Received request for batch status stream for batch_job_id: {batch_job_id}")
    logging.info(f"DEBUG: Current job_temp_storage keys at stream start: {list(job_temp_storage.keys())}") # Upewnij się, że używasz job_temp_storage (bez _)
    try:
        batch_job_info = job_temp_storage.get(batch_job_id)

        if not batch_job_info or batch_job_info.get('type') != 'batch':
            logging.error(f"Batch job with ID {batch_job_id} not found or is not a batch type. Returning 404.")
            return JSONResponse(status_code=404, content={"detail": f"Batch job with ID {batch_job_id} not found."})

        file_job_ids = batch_job_info.get('file_job_ids', [])
        if not file_job_ids:
            logging.warning(f"Batch job {batch_job_id} has no associated file_job_ids. Returning 400.")
            return JSONResponse(status_code=400, content={"detail": f"Batch job {batch_job_id} has no files to process."})

        batch_params = batch_job_info.get('params', {})
        batch_whisper_model_size = batch_params.get('whisper_model_size', config.WHISPER_MODEL_SIZE)
        batch_custom_prompt = batch_params.get('custom_prompt', None)
        batch_output_name = batch_params.get('output_name', None)
        batch_chunk_duration = batch_params.get('chunk_duration', config.CHUNK_DURATION)
        batch_chunk_overlap = batch_params.get('chunk_overlap', config.CHUNK_OVERLAP)

        async def batch_event_generator():
            try:
                total_files = len(file_job_ids)
                completed_files = 0
                all_transcription_segments = []
                all_minutes_content = []
                all_minutes_filenames = []
                all_transcription_filenames = []
                collection_names = []

                for file_job_id in file_job_ids:
                    file_info = job_temp_storage.get(file_job_id)
                    if not file_info:
                        logging.error(f"File job {file_job_id} not found in storage. Skipping.")
                        continue

                    original_filename = file_info.get('filename', file_job_id)
                    file_path = Path(file_info['file_path'])
                    file_params = file_info['params']

                    processing_message = {
                        "message": f"Processing file: {original_filename}",
                        "status_type": "info",
                        "batch_progress": {
                            "completed_files": completed_files,
                            "total_files": total_files,
                            "current_file_name": original_filename
                        }
                    }
                    yield "event: message\\ndata: " + json.dumps(processing_message) + "\\n\\n"
                    await asyncio.sleep(0.05)

                    file_id = None # Inicjalizacja file_id dla każdego pliku w partii
                    file_hash = None # Inicjalizacja file_hash dla każdego pliku w partii
                    transcript_id = None # Inicjalizacja transcript_id
                    minutes_db_id = None # Inicjalizacja minutes_db_id
                    collection_name_for_file = None # Inicjalizacja collection_name_for_file


                    try:
                        # KROK 1: Dodanie pliku do processed_files
                        try:
                            file_hash = await run_in_threadpool(agent_db.compute_file_hash, file_path) # Użyj run_in_threadpool
                            logging.info(f"[{file_job_id}] Computed file hash: {file_hash}")
                            file_id = await run_in_threadpool(agent_db.add_file, original_filename, str(file_path), file_hash, "processing", "") # Użyj run_in_threadpool
                            logging.info(f"[{file_job_id}] File '{original_filename}' added to processed_files with ID: {file_id}")
                        except Exception as e:
                            logging.error(f"[{file_job_id}] Error adding file record to processed_files table: {e}", exc_info=True)
                            raise ValueError(f"Database error during file record creation: {e}")


                        transcription_segments, transcription_filename, transcript_id = await process_audio(\
                            file_path,\
                            file_job_id,\
                            file_params.get('whisper_model_size', config.WHISPER_MODEL_SIZE)\
                        )
                        if not transcription_segments:
                            logging.error(f"[{file_job_id}] Audio processing failed or produced no segments for file {file_job_id}.")
                            if file_hash:
                                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Audio processing failed") # Użyj run_in_threadpool
                            raise ValueError(f"Audio processing failed or produced no segments for file {file_job_id}.")
                        logging.info(f"[{file_job_id}] Audio processing complete. Transcript ID: {transcript_id}")

                        # DODAJ: Zapis segmentów do bazy
                        if transcript_id is not None and transcription_segments:
                            for seg in transcription_segments:
                                agent_db.add_transcript_segment(
                                    transcript_id,
                                    seg.start,
                                    seg.end,
                                    seg.speaker,
                                    seg.text
                                )
                            logging.info(f"[{file_job_id}] Added {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")


                        collection_name_for_file = await ingest_transcription(
                            transcription_segments,
                            file_path,
                            file_job_id,
                            transcript_id, # Dodano transcript_id
                            file_params.get('chunk_duration', config.CHUNK_DURATION), # Jest już częścią definicji funkcji
                            file_params.get('chunk_overlap', config.CHUNK_OVERLAP)   # Jest już częścią definicji funkcji
                        )
                        if not collection_name_for_file:
                            logging.error(f"[{file_job_id}] Ingestion to Qdrant failed for file {file_job_id}.")
                            if file_hash:
                                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Ingestion to Qdrant failed") # Użyj run_in_threadpool
                            raise ValueError(f"Ingestion to Qdrant failed for file {file_job_id}.")
                        logging.info(f"[{file_job_id}] Ingested to Qdrant collection: {collection_name_for_file}")

                        relevant_date = get_relevant_date_for_file(file_path)
                        minutes_content_response, minutes_filename, minutes_db_id = await generate_and_save_minutes(\
                            collection_name_for_file,\
                            file_path,\
                            custom_prompt=file_params.get('custom_prompt', None),\
                            output_name=file_params.get('output_name', None),\
                        )
                        if minutes_content_response is None:
                            logging.error(f"[{file_job_id}] Minutes generation failed for file {file_job_id}.")
                            if file_hash:
                                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Minutes generation failed") # Użyj run_in_threadpool
                            raise ValueError(f"Minutes generation failed for file {file_job_id}.")
                        logging.info(f"[{file_job_id}] Minutes generated successfully. Minutes ID: {minutes_db_id}")

                        # KROK 2: Dodanie wpisu do meeting_data (po wszystkich ID)
                        try:
                            if file_id is not None and transcript_id is not None and minutes_db_id is not None and collection_name_for_file is not None:
                                await run_in_threadpool(agent_db.add_meeting_data, file_id, transcript_id, minutes_db_id, relevant_date, collection_name_for_file) # Użyj run_in_threadpool
                                logging.info(f"[{file_job_id}] Meeting data added to database for file_id: {file_id}")
                            else:
                                missing_info = []
                                if file_id is None: missing_info.append("File ID")
                                if transcript_id is None: missing_info.append("Transcript ID")
                                if minutes_db_id is None: missing_info.append("Minutes ID")
                                if collection_name_for_file is None: missing_info.append("Collection Name")
                                logging.warning(f"[{file_job_id}] Could not add meeting data to database due to missing information: {', '.join(missing_info)}")
                                if file_hash: # Jeśli brakuje danych do meeting_data, oznacz jako błąd
                                    await run_in_threadpool(agent_db.update_file_status, file_hash, "error", "Incomplete data for meeting_data entry") # Użyj run_in_threadpool
                                raise ValueError("Incomplete data for meeting_data entry.")
                        except Exception as e_db_add:
                            logging.error(f"[{file_job_id}] Error adding meeting data to database: {e_db_add}", exc_info=True)
                            if file_hash:
                                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", f"Database error during meeting data creation: {e_db_add}") # Użyj run_in_threadpool
                            raise ValueError(f"Database error during meeting data creation: {e_db_add}")

                        completed_files += 1

                        # KROK 3: Finalna aktualizacja statusu pliku na "completed"
                        if file_hash: # Upewnij się, że file_hash jest dostępne
                            await run_in_threadpool(agent_db.update_file_status, file_hash, "completed", "Processing successful") # Użyj run_in_threadpool
                            logging.info(f"[{file_job_id}] File '{original_filename}' processing completed successfully. Status updated to 'completed'.")


                        file_complete_message = {
                            "message": f"File {original_filename} processed successfully!",
                            "status_type": "success",
                            "transcription_filename": str(transcription_filename),
                            "minutes_filename": str(minutes_filename),
                            "transcription": [seg.model_dump() for seg in transcription_segments] if transcription_segments else [],
                            "minutes": minutes_content_response.model_dump() if hasattr(minutes_content_response, 'model_dump') else minutes_content_response,
                            "original_filename": original_filename,
                            "batch_progress": {
                                "completed_files": completed_files,
                                "total_files": total_files,
                                "current_file_name": original_filename
                            }
                        }
                        yield "event: message\\ndata: " + json.dumps(file_complete_message) + "\\n\\n"
                        await asyncio.sleep(0.05) # Zwiększono czas oczekiwania po każdym pliku

                    except Exception as e:
                        logging.error(f"[{file_job_id}] Processing failed for file job {file_job_id} in batch {batch_job_id}: {e}", exc_info=True)
                        # Aktualizacja statusu na 'error' w przypadku jakiegokolwiek błędu w tej pętli
                        if file_hash:
                            # Upewnij się, że status nie jest już 'completed'
                            current_status = await run_in_threadpool(agent_db.get_file_status, file_hash) # Użyj run_in_threadpool
                            if current_status != "completed":
                                await run_in_threadpool(agent_db.update_file_status, file_hash, "error", f"Processing failed: {str(e)}") # Użyj run_in_threadpool
                                logging.info(f"[{file_job_id}] File '{original_filename}' status updated to 'error' due to exception.")

                        error_message = {
                            "message": f"Processing failed for {original_filename}: {str(e)}",
                            "status_type": "error",
                            "original_filename": original_filename,
                            "batch_progress": {
                                "completed_files": completed_files,
                                "total_files": total_files,
                                "current_file_name": original_filename
                            }
                        }
                        yield "event: message\\ndata: " + json.dumps(error_message) + "\\n\\n"
                        await asyncio.sleep(0.05) # Zwiększono czas oczekiwania po komunikacie o błędzie
                    finally:
                        logging.info(f"[{file_job_id}] Cleaning up temporary directory for file job.") # Nowy log
                        cleanup_temp_dir(file_job_id)
                        job_temp_storage.pop(file_job_id, None)

                # Przeniesione logowanie RAM/VRAM, aby było przed finalnym wysłaniem wiadomości "batch_complete"
                ram_gb = get_current_ram_usage_gb()
                if ram_gb > 0:
                    logging.info(f"Final RAM usage after processing batch job {batch_job_id}: {ram_gb:.2f} GB")
                else:
                    logging.warning(f"RAM usage could not be retrieved after batch job {batch_job_id}.")

                print("DEBUG: Punkt kontrolny przed VRAM log (w generatorze batch).") # NOWY Wiersz dla debugowania

                current_vram_usage = get_current_vram_usage_gb()
                if current_vram_usage > 0:
                    logging.info(f"Final VRAM usage after processing batch job {batch_job_id}: {current_vram_usage:.2f} GB")
                else:
                    logging.warning(f"VRAM usage could not be retrieved after batch job {batch_job_id}. See warnings above.")

                final_message_data = {
                    "batch_complete": True,
                    "message": "Batch processing complete!",
                    "status_type": "success",
                    "batch_progress": {
                        "completed_files": total_files,
                        "total_files": total_files
                    }
                }
                json_data_str = json.dumps(final_message_data)
                yield "event: message\\ndata: " + json_data_str + "\\n\\n"
                await asyncio.sleep(2) # Zwiększono czas oczekiwania, aby dać frontendowi więcej czasu na odebranie wiadomości
                job_temp_storage.pop(batch_job_id, None) # To usunie batch_job_id z tymczasowego storage
                logging.info(f"DEBUG: Batch job {batch_job_id} removed from job_temp_storage after batch_complete. Stream closing soon.") # Nowy log przed zamknięciem strumienia

            except Exception as e:
                logging.error(f"FATAL ERROR in batch_event_generator for batch {batch_job_id}: {e}", exc_info=True)
                error_message = {
                    "batch_complete": True,
                    "message": f"Fatal error during batch processing: {str(e)}",
                    "status_type": "error",
                    "batch_progress": {
                        "completed_files": completed_files if 'completed_files' in locals() else 0,
                        "total_files": total_files if 'total_files' in locals() else 0
                    }
                }
                yield "event: message\\ndata: " + json.dumps(error_message) + "\\n\\n"
                await asyncio.sleep(0.5) # Zwiększono czas oczekiwania po fatalnym błędzie, zanim strumień się zamknie

        logging.info(f"Successfully prepared StreamingResponse for batch_job_id: {batch_job_id}")
        return StreamingResponse(batch_event_generator(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Unhandled exception in process_batch_status_stream for batch {batch_job_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error during batch stream setup: {str(e)}"}
        )

def retry_errors(handler):
    error_files = get_files_with_error()
    if not error_files:
        print("Brak plików z błędem do ponownego przetworzenia.")
        return
    print(f"Retry dla {len(error_files)} plików z błędem...")
    for filename, filepath in error_files:
        file_path = Path(filepath)
        if file_path.exists():
            print(f"Ponowna próba przetworzenia: {file_path}")
            handler.process_file(file_path, force_retry=True)
        else:
            print(f"Plik {file_path} już nie istnieje – pomijam.")

def process_existing_files(watch_dir):
    for file_path in Path(watch_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {".wav", ".mp3", ".mkv", ".mp4", ".flac", ".ogg", ".m4a"}:
            print(f"Znaleziono istniejący plik: {file_path}, dodaję do kolejki...")
            file_queue.put(file_path)

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Loading models and enrolled speakers...")
    try:
        # USUNIĘTO: agent_db.init_db() - Inicjalizacja bazy danych SQLite będzie zarządzana przez Database Agent.
        # logging.info("Attempting to initialize SQLite database...")
        # agent_db.init_db()
        # logging.info("SQLite database initialized successfully during startup.")

        logging.info("Attempting to initialize Qdrant client and embedding model...")
        await initialize_qdrant_resources()
        logging.info("Qdrant client and embedding model initialized successfully during startup.")

        logging.info("Attempting to load diarization and enrolled speakers models...")
        from utils import get_diarization_models, load_enrolled_speakers, _enrolled_speakers_profiles # Ensure _enrolled_speakers_profiles is imported here as well
        get_diarization_models()
        logging.info("Diarization and embedding models loaded.")

        load_enrolled_speakers() # This will populate _enrolled_speakers_profiles
        logging.info(f"DEBUG: _enrolled_speakers_profiles after initial startup load: {list(_enrolled_speakers_profiles.keys())} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO
        logging.info("Enrolled speakers profiles initialized during startup.")

        logging.info("All startup tasks completed.")

    except Exception as e:
        logging.error(f"FATAL ERROR during startup: {e}", exc_info=True)
        # Re-raise the exception to make sure the app doesn't start in a broken state
        raise

class PromptSet(BaseModel):
    name: str
    prompts: Dict[str, str]

@app.get("/prompts/list")
async def list_prompts_endpoint():
    try:
        prompts = load_prompt_config()
        return JSONResponse(content=prompts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prompts: {e}")

@app.post("/prompts/save")
async def save_prompts_endpoint(prompt_set: PromptSet):
    try:
        save_prompt_config(prompt_set.name, prompt_set.prompts)
        return {"message": "Prompt set saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prompts: {e}")

@app.delete("/prompts/delete/{prompt_name}")
async def delete_prompts_endpoint(prompt_name: str):
    try:
        delete_prompt_config(prompt_name)
        return {"message": "Prompt set deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prompts: {e}")

# --- NOWE ENDPOINTY DLA IDENYTFIKACJI MÓWCÓW ---

@app.post("/enroll_speaker_direct/")
async def enroll_speaker_direct_endpoint(
    audio_file: UploadFile = File(...),
    speaker_name_form: Optional[str] = Form(None) # Nowe pole do przyjmowania nazwy mówcy z formularza
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    # Użyj nazwy mówcy z formularza, jeśli podano, w przeciwnym razie z nazwy pliku
    speaker_name = speaker_name_form if speaker_name_form else Path(audio_file.filename).stem

    if not speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty, please provide a name or ensure the audio file has a valid name.")

    enrollment_job_id = f"enroll_{uuid.uuid4().hex}"
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    try:
        audio_path = await save_uploaded_file_temp(audio_file, enrollment_job_id, temp_dir_obj, temp_dir_path)
        # Zmieniono wywołanie funkcji, aby użyć await bezpośrednio, bo enroll_speaker_from_audio jest już async
        enrollment_success = await enroll_speaker_from_audio(audio_path, speaker_name)

        if enrollment_success:
            logging.info(f"Speaker '{speaker_name}' enrolled successfully from {audio_path}")
            # Przeładowanie mówców po zapisaniu, aby lista na froncie była aktualna
            from utils import load_enrolled_speakers, _enrolled_speakers_profiles # Import here to ensure we get the latest
            await run_in_threadpool(load_enrolled_speakers)
            logging.info(f"DEBUG: _enrolled_speakers_profiles in api_app after enroll and reload: {list(_enrolled_speakers_profiles.keys())} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO
            return JSONResponse(status_code=200, content={"message": f"Speaker '{speaker_name}' enrolled successfully."})
        else:
            logging.error(f"Failed to enroll speaker '{speaker_name}' from {audio_path}")
            raise HTTPException(status_code=500, detail=f"Failed to enroll speaker '{speaker_name}'. Check logs for details.")

    except Exception as e:
        logging.error(f"Error during direct speaker enrollment for '{speaker_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker enrollment: {str(e)}")
    finally:
        if temp_dir_obj:
            temp_dir_obj.cleanup()
            logging.info(f"Temporary directory for enrollment job {enrollment_job_id} cleaned up.")

@app.delete("/delete_speaker/{speaker_name}")
async def delete_speaker_endpoint(speaker_name: str):
    try:
        from utils import delete_speaker, load_enrolled_speakers, _enrolled_speakers_profiles # Ensure imports here
        success = await run_in_threadpool(delete_speaker, speaker_name)
        if success:
            # Przeładowanie mówców po usunięciu, aby lista na froncie była aktualna
            await run_in_threadpool(load_enrolled_speakers)
            logging.info(f"DEBUG: _enrolled_speakers_profiles in api_app after delete and reload: {list(_enrolled_speakers_profiles.keys())} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO
            return JSONResponse(status_code=200, content={"message": f"Speaker '{speaker_name}' deleted successfully."})
        else:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found or could not be deleted.")
    except Exception as e:
        logging.error(f"Error deleting speaker '{speaker_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker deletion: {str(e)}")

@app.get("/list_processed_files/")
async def list_processed_files_endpoint():
    processed_files_info = {}
    output_base_dir = Path(config.OUTPUT_DIR_API_TRANSCRIPTS)
    if not output_base_dir.exists():
        return JSONResponse(content={})

    for job_id_dir in output_base_dir.iterdir():
        if job_id_dir.is_dir():
            transcription_files = list(job_id_dir.glob("*_diarized_transcription.txt"))
            if transcription_files:
                transcription_file = transcription_files[0]
                parts = transcription_file.stem.split('_', 2)
                original_filename_stem = parts[2] if len(parts) > 2 else transcription_file.stem

                processed_files_info[job_id_dir.name] = {
                    "display_name": f"{original_filename_stem} (ID: {job_id_dir.name[:8]}...)",
                    "file_job_id": job_id_dir.name,
                    "transcription_path": str(transcription_file)
                }
    return JSONResponse(content=processed_files_info)

@app.get("/get_file_speakers/{file_job_id}")
async def get_file_speakers_endpoint(file_job_id: str):
    transcription_file_path = Path(config.OUTPUT_DIR_API_TRANSCRIPTS) / file_job_id / f"*_diarized_transcription.txt"
    found_files = list(Path(config.OUTPUT_DIR_API_TRANSCRIPTS).glob(f"{file_job_id}/*_diarized_transcription.txt"))

    if not found_files:
        raise HTTPException(status_code=404, detail=f"Transcription file for job ID {file_job_id} not found.")

    file_path = found_files[0]

    unique_speakers = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    speaker_label = line.split(":", 1)[0].strip()
                    if speaker_label.startswith("SPEAKER_"):
                        unique_speakers.add(speaker_label)
        return JSONResponse(content=sorted(list(unique_speakers)))
    except Exception as e:
        logging.error(f"Error reading transcription file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read speakers from file: {e}")

@app.get("/get_enrolled_speakers/")
async def get_enrolled_speakers_endpoint():
    from utils import _enrolled_speakers_profiles # Ensure import here
    current_enrolled_speakers = list(_enrolled_speakers_profiles.keys())
    logging.info(f"DEBUG: get_enrolled_speakers_endpoint about to return: {current_enrolled_speakers} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO
    return JSONResponse(content=current_enrolled_speakers)

@app.exception_handler(404)
async def custom_404_handler(request, exc):
    logging.warning(f"404 Not Found: {request.url}")
    if not request.url.path.startswith("/static/"):
        return HTMLResponse(content=open("index.html", "r", encoding="utf-8").read(), status_code=200)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Loading models...")
    try:
        from utils import get_diarization_models
        get_diarization_models()
        logging.info("All models loaded and enrolled speakers profiles initialized.")
    except Exception as e:
        logging.error(f"Failed to load models during startup: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}
