from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, Query, Body
from models import ProcessingResult
from file_handlers import cleanup_temp_dir, job_temp_storage, save_uploaded_file_temp, save_file_to_temp_and_convert_if_needed
from processor import process_audio, TranscriptionProviderError # Poprawiony import z processor.py
from ingest import ingest_transcription
from minutes_service import generate_and_save_minutes
import config

# Import StreamingResponse for SSE
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uuid
import logging
import sys
import warnings # DODANO

_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
else:
    _root_logger.setLevel(logging.INFO)

_uvicorn_logger = logging.getLogger("uvicorn.error")
if _uvicorn_logger.handlers:
    for handler in _uvicorn_logger.handlers:
        if handler not in _root_logger.handlers:
            _root_logger.addHandler(handler)

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module") # DODANO
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.checkpoints") # DODANO
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.processing.features") # DODANO

from pathlib import Path
import json # Moved import json to the top
from typing import List, Optional, Tuple, Dict, Any, Set, Deque, Callable, Awaitable, AsyncGenerator # Dodaj dodatkowe typy
from dataclasses import dataclass, field
from collections import deque
import torch  # Dodaj na górze pliku
import contextlib
import time
import httpx
import threading
import queue
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from datetime import datetime
import shutil # NOWY IMPORT: Do czyszczenia tymczasowych plików/katalogów
import torchaudio # NOWY IMPORT: Do ładowania i zapisywania audio segmentów
from pyannote.core import Segment # NOWY IMPORT: Do pracy z segmentami diarization
from starlette.concurrency import run_in_threadpool # NOWY IMPORT: Do obsługi blokujących operacji I/O
from fastapi.middleware.cors import CORSMiddleware # NOWY IMPORT DLA CORS
from fastapi import Request # PRZYWRÓCONO - potrzebne do ręcznego parsowania

from summarizer import generate_minutes_of_meeting, gemini_model, generate_chat_response # Upewnij się, że to jest zaimportowane, DODANO gemini_model
from utils import save_minutes_to_file, load_prompt_config, save_prompt_config, delete_prompt_config, get_file_creation_date, get_relevant_date_for_file, extract_date_from_filename, get_default_prompt_names
from utils import (
    enroll_speaker_from_audio,
    load_enrolled_speakers,
    list_enrolled_speakers,
    delete_speaker,
    get_diarization_models,
    get_current_ram_usage_gb,
    get_current_vram_usage_gb,
    get_transcription_provider_catalog,
    build_transcription_metadata,
)
import tempfile # NOWY IMPORT: Do bezpośredniego zapisu plików audio
import agent_db # NEW: Import agent_db
from queue_service import start_boss, stop_boss, enqueue_message, enqueue_batch_processing, enqueue_single_file_processing, get_job_status, register_message_handler, get_queue_stats, list_active_jobs

# NOWE IMPORTY DLA FUNKCJONALNOŚCI CZATU
from qdrant_handler import (
    search_all_collections,
    get_all_collection_names,
    initialize_qdrant_resources,
    delete_collection_if_exists,
) # DODANO
from typing import List # Upewnij się, że jest zaimportowane na górze, ale dodaj na wszelki wypadek
import asyncio # DODANY IMPORT: Do obsługi asynchroniczności (tylko jeden raz)

# NOWE IMPORTY DLA AUTORYZACJI JWT
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # NOWE IMPORTY
from jose import JWTError, jwt # NOWE IMPORTY
from passlib.context import CryptContext # NOWY IMPORT
from datetime import datetime, timedelta # NOWY IMPORT
from typing import Optional # Upewnij się, że Optional jest zaimportowane

app = FastAPI(
    title="Audio/Video Transcription & Summarization API",
    description="API to process audio/video files and generate meeting minutes.",
    version="1.0.0",
)

# DODANO: Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Zezwól na połączenia z Twojego frontendu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class BatchStreamState:
    listeners: Set["asyncio.Queue[str]"] = field(default_factory=set)
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=500))
    task: Optional["asyncio.Task[None]"] = None
    completed: bool = False
    error: Optional[str] = None


batch_stream_states: Dict[str, BatchStreamState] = {}


# Pomocnicze funkcje do obsługi publikacji zdarzeń batcha
def _enqueue_batch_message(state: BatchStreamState, payload: str) -> None:
    """Dodaj komunikat do historii oraz do wszystkich aktywnych słuchaczy."""
    state.history.append(payload)
    stale_listeners = []
    listeners_count = len(state.listeners)
    if listeners_count > 0:
        logging.debug(f"Wysyłanie wiadomości do {listeners_count} listenerów")
    else:
        logging.debug("Brak aktywnych listenerów - wiadomość zostanie zapisana w historii")
    
    for listener_queue in list(state.listeners):
        try:
            listener_queue.put_nowait(payload)
        except asyncio.QueueFull:
            stale_listeners.append(listener_queue)
            logging.warning(f"Listener queue pełny, usuwam z listy")
    for listener_queue in stale_listeners:
        state.listeners.discard(listener_queue)


async def publish_batch_message(batch_job_id: str, message: Dict[str, Any]) -> None:
    """Publikuj komunikat postępu dla danego batcha."""
    state = batch_stream_states.get(batch_job_id)
    if not state:
        logging.debug(f"[{batch_job_id}] publish_batch_message invoked without active state.")
        return
    payload = f"event: message\ndata: {json.dumps(message)}\n\n"
    _enqueue_batch_message(state, payload)


def finalize_batch_stream(batch_job_id: str) -> None:
    """Oznacz batch jako zakończony i powiadom wszystkich słuchaczy."""
    state = batch_stream_states.get(batch_job_id)
    if not state:
        return
    state.completed = True
    for listener_queue in list(state.listeners):
        listener_queue.put_nowait(None)


def initialize_batch_stream_state(batch_job_id: str) -> BatchStreamState:
    state = batch_stream_states.get(batch_job_id)
    if state is None:
        state = BatchStreamState()
        batch_stream_states[batch_job_id] = state
    else:
        state.completed = False
        state.error = None
        state.history.clear()
    return state


async def _consume_batch_generator(
    batch_job_id: str,
    generator_factory: Callable[[], AsyncGenerator[str, None]],
    on_complete: Optional[Callable[[], Awaitable[None]]] = None,
) -> None:
    state = batch_stream_states.get(batch_job_id)
    if state is None:
        state = batch_stream_states.get(batch_job_id)
        if state is None:
            state = initialize_batch_stream_state(batch_job_id)
    
    logging.info(f"[{batch_job_id}] _consume_batch_generator uruchomiony. Liczba listenerów: {len(state.listeners)}")

    try:
        async for payload in generator_factory():
            logging.debug(f"[{batch_job_id}] Otrzymano payload z batch_event_generator, liczba listenerów: {len(state.listeners)}")
            _enqueue_batch_message(state, payload)
    except Exception as e:
        logging.error(f"[{batch_job_id}] Exception while consuming batch generator: {e}", exc_info=True)
        state.error = str(e)
        error_message = {
            "batch_complete": True,
            "message": f"Fatal error during batch processing: {str(e)}",
            "status_type": "error"
        }
        payload = f"event: message\ndata: {json.dumps(error_message)}\n\n"
        _enqueue_batch_message(state, payload)
    finally:
        logging.info(f"[{batch_job_id}] _consume_batch_generator finally - finalizuję batch stream")
        finalize_batch_stream(batch_job_id)
        if on_complete:
            try:
                logging.info(f"[{batch_job_id}] Wywołuję on_complete (finalize_batch_status_for_db)")
                await on_complete()
                logging.info(f"[{batch_job_id}] ✅ on_complete zakończony pomyślnie")
            except Exception as finalize_error:
                logging.error(f"[{batch_job_id}] ❌ Error during batch completion callback: {finalize_error}", exc_info=True)
        else:
            logging.warning(f"[{batch_job_id}] ⚠️  on_complete nie jest ustawiony - status batcha NIE zostanie zaktualizowany!")


def _sanitize_provider_token_mapping(data: Any) -> Dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError("Expected object.")
    cleaned: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        normalized_key = str(key).strip().lower()
        if normalized_key not in {"gemini", "assemblyai", "openai"}:
            continue
        sanitized_value = value.strip()
        if sanitized_value:
            cleaned[normalized_key] = sanitized_value
    return cleaned


def _parse_provider_tokens_json(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return _sanitize_provider_token_mapping(parsed)
    except (json.JSONDecodeError, ValueError) as parse_error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider_tokens payload: {parse_error}",
        )


_REMOTE_TRANSCRIPTION_PROVIDERS = {"gemini", "assemblyai"}
_REMOTE_DIARIZATION_LABEL = "Modal remote diarization"


def _compose_transcription_metadata(
    provider: Optional[str],
    model: Optional[str],
) -> Dict[str, str]:
    metadata = build_transcription_metadata(provider, model)
    provider_id = metadata.get("transcription_provider", "")
    metadata["diarization_model"] = (
        _REMOTE_DIARIZATION_LABEL if provider_id in _REMOTE_TRANSCRIPTION_PROVIDERS else config.PYANNOTE_PIPELINE
    )
    return metadata


def create_batch_event_generator(batch_job_id: str, file_job_ids: List[str]) -> Callable[[], AsyncGenerator[str, None]]:
    async def batch_event_generator():
        logging.info(f"[{batch_job_id}] Starting batch_event_generator.")
        progress_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        message_queue: asyncio.Queue[str] = asyncio.Queue()

        async def queue_listener_coroutine():
            while True:
                message = await progress_queue.get()
                if message is None:
                    break
                await message_queue.put(f"event: message\ndata: {json.dumps(message)}\n\n")
                progress_queue.task_done()

        listener_task = asyncio.create_task(queue_listener_coroutine())

        try:
            total_files = len(file_job_ids)
            completed_files = 0
            all_transcription_segments = []
            all_minutes_content = []
            all_minutes_filenames = []
            all_transcription_filenames = []
            collection_names = []

            async def flush_message_queue():
                while not message_queue.empty():
                    yield await message_queue.get()
                    message_queue.task_done()

            for file_job_id in file_job_ids:
                async for msg in flush_message_queue():
                    yield msg

                logging.info(f"[{batch_job_id}] Processing file_job_id: {file_job_id}")
                file_info = job_temp_storage.get(file_job_id)
                if not file_info:
                    logging.error(f"[{batch_job_id}] File job {file_job_id} not found in storage. Skipping.")
                    continue
                logging.debug(
                    "[%s] Retrieved file_info keys for %s: %s",
                    batch_job_id,
                    file_job_id,
                    list(file_info.keys()),
                )

                original_filename = file_info.get('filename')
                if not original_filename or original_filename == file_job_id:
                    db_file_id = file_info.get('db_file_id')
                    db_record = None
                    try:
                        if db_file_id is not None:
                            db_record = await run_in_threadpool(agent_db.get_file_record_by_id, int(db_file_id))
                        if not db_record:
                            db_record = await run_in_threadpool(agent_db.get_file_record_by_filehash, file_job_id)
                    except Exception as lookup_error:
                        logging.warning(f"[{file_job_id}] Could not retrieve filename from DB: {lookup_error}")
                    if db_record and db_record.get('filename'):
                        original_filename = db_record['filename']
                        file_info['filename'] = original_filename
                        job_temp_storage[file_job_id] = file_info
                    else:
                        original_filename = file_job_id

                file_path = Path(file_info['file_path'])
                file_params = file_info['params']
                provider_metadata = _compose_transcription_metadata(
                    file_params.get('transcription_provider', config.DEFAULT_TRANSCRIPTION_PROVIDER),
                    file_params.get('transcription_model'),
                )
                file_params['transcription_metadata'] = provider_metadata

                processing_message = {
                    "message": f"Processing file: {original_filename}",
                    "status_type": "info",
                    "file_job_id": file_job_id,
                    "batch_progress": {
                        "completed_files": completed_files,
                        "total_files": total_files,
                        "current_file_name": original_filename,
                        "file_job_id": file_job_id
                    }
                }
                yield f"event: message\ndata: {json.dumps(processing_message)}\n\n"
                await asyncio.sleep(0.05)

                file_id = None
                file_hash = None
                transcript_id = None
                minutes_db_id = None
                collection_name_for_file = None

                async def set_file_status(new_status: str, message: str = "") -> None:
                    if file_id is not None:
                        await run_in_threadpool(
                            agent_db.update_file_status_by_id,
                            file_id,
                            new_status,
                            message,
                        )
                    elif file_hash:
                        await run_in_threadpool(
                            agent_db.update_file_status,
                            file_hash,
                            new_status,
                            message,
                        )

                try:
                    logging.info(f"[{file_job_id}] Starting processing for {original_filename}. Steps: hash, audio, ingest, minutes, meeting_data.")
                    try:
                        logging.debug(f"[{file_job_id}] Computing file hash for {file_path}.")
                        file_hash = await run_in_threadpool(agent_db.compute_file_hash, file_path)
                        logging.info(f"[{file_job_id}] Computed file hash: {file_hash}")
                        file_id = file_info.get('db_file_id')
                        if file_id is None:
                            logging.error(f"[{file_job_id}] db_file_id not found in job_temp_storage. Creating new record.")
                            file_id = await run_in_threadpool(agent_db.add_file, original_filename, str(file_path), file_hash, "processing", "")
                        logging.info(f"[{file_job_id}] Retrieved DB file_id: {file_id}")
                        await run_in_threadpool(agent_db.update_file_hash, file_id, file_hash)
                        await set_file_status("processing", "Audio processing started")

                        def whisper_progress_update_callback(percentage: float):
                            try:
                                logging.debug(f"[{file_job_id}] Transcription progress: {percentage:.1f}%")
                            except Exception:
                                logging.debug(f"[{file_job_id}] Transcription progress: {percentage}%")

                        yield f'event: message\ndata: {json.dumps({"message": f"Audio processing started for {original_filename}", "status_type": "info", "file_job_id": file_job_id, "current_file_progress_percentage": 10})}\n\n'
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        logging.error(f"[{file_job_id}] Error in initial file record handling: {e}", exc_info=True)
                        raise ValueError(f"Database error during file record creation/retrieval: {e}")

                    logging.info(f"[{file_job_id}] Starting audio processing for {original_filename}.")
                    current_file_job_id_str = str(file_job_id)
                    try:
                        transcription_segments, transcription_filename, transcript_id = await process_audio(
                            audio_path=file_path,
                            file_job_id=current_file_job_id_str,
                            transcription_model=file_params.get('transcription_model'),
                            transcription_provider=file_params.get('transcription_provider', config.DEFAULT_TRANSCRIPTION_PROVIDER),
                            progress_callback=whisper_progress_update_callback,
                            username=file_params.get('username'),
                            file_id=file_id,
                            provider_tokens=file_params.get('provider_tokens'),
                        )
                    except TranscriptionProviderError as provider_exc:
                        logging.error(f"[{file_job_id}] Transcription provider error: {provider_exc}")
                        await set_file_status("error", str(provider_exc))
                        raise HTTPException(status_code=400, detail=str(provider_exc))
                    async for msg in flush_message_queue():
                        yield msg

                    if not transcription_segments:
                        logging.error(f"[{file_job_id}] Audio processing failed or produced no segments for '{original_filename}'.")
                        await set_file_status("error", "Audio processing failed")
                        raise HTTPException(status_code=500, detail="Audio processing failed.")
                    logging.info(f"[{file_job_id}] Audio processing complete. Transcript ID: {transcript_id}")
                    await set_file_status("transcribed", "Audio transcribed successfully")
                    yield f'event: message\ndata: {json.dumps({"message": f"Audio transcribed successfully for {original_filename}", "status_type": "info", "file_job_id": file_job_id, "current_file_progress_percentage": 30})}\n\n'
                    await asyncio.sleep(0.01)
                    async for msg in flush_message_queue():
                        yield msg

                    if transcript_id is not None and transcription_segments:
                        logging.info(f"[{file_job_id}] Adding {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")
                        for seg in transcription_segments:
                            await run_in_threadpool(
                                agent_db.add_transcript_segment,
                                transcript_id,
                                seg.start,
                                seg.end,
                                seg.speaker,
                                seg.text
                            )
                        logging.info(f"[{file_job_id}] Added {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")

                    logging.info(f"[{file_job_id}] Starting ingestion to Qdrant for '{original_filename}'...")
                    await set_file_status("ingesting", "Ingesting to Qdrant")
                    yield f'event: message\ndata: {json.dumps({"message": f"Ingesting to Qdrant for {original_filename}", "status_type": "info", "file_job_id": file_job_id, "current_file_progress_percentage": 50})}\n\n'
                    await asyncio.sleep(0.01)
                    async for msg in flush_message_queue():
                        yield msg

                    collection_name_for_file = await ingest_transcription(
                        transcription_segments,
                        file_path,
                        file_job_id,
                        transcript_id,
                        file_params.get('chunk_duration', config.CHUNK_DURATION),
                        file_params.get('chunk_overlap', config.CHUNK_OVERLAP)
                    )
                    if not collection_name_for_file:
                        logging.error(f"[{file_job_id}] Ingestion to Qdrant failed for '{original_filename}'.")
                        await set_file_status("error", "Ingestion to Qdrant failed")
                        collection_name = None
                        raise HTTPException(status_code=500, detail="Ingestion to Qdrant failed.")
                    else:
                        collection_name = collection_name_for_file
                    logging.info(f"[{file_job_id}] Ingested to Qdrant collection: {collection_name}")
                    await set_file_status("indexed", "Qdrant indexing complete")
                    yield f'event: message\ndata: {json.dumps({"message": f"Qdrant indexing complete for {original_filename}", "status_type": "info", "file_job_id": file_job_id, "current_file_progress_percentage": 70})}\n\n'
                    await asyncio.sleep(0.01)
                    async for msg in flush_message_queue():
                        yield msg

                    summary_delay = max(0.0, getattr(config, "GEMINI_SUMMARY_DELAY_SECONDS", 0.0))
                    if summary_delay > 0:
                        logging.info(
                            f"[{file_job_id}] Waiting {summary_delay:.1f}s before generating minutes to reduce Gemini load."
                        )
                        await asyncio.sleep(summary_delay)

                    logging.info(f"[{file_job_id}] Generating and saving minutes for '{original_filename}'...")
                    await set_file_status("summarizing", "Generating minutes")

                    def minutes_progress_update_callback(percentage: float, section_title: Optional[str] = None):
                        clamped_percentage = max(0.0, min(percentage, 100.0))
                        mapped_percentage = 80 + (clamped_percentage * 0.15)
                        message_text = f"Generowanie minut dla {original_filename}"
                        if section_title:
                            message_text += f" – sekcja: {section_title}"
                        progress_message = {
                            "message": message_text,
                            "status_type": "info",
                            "file_job_id": file_job_id,
                            "current_file_name": original_filename,
                            "current_file_progress_percentage": mapped_percentage,
                            "current_file_minutes_progress_percentage": clamped_percentage
                        }
                        progress_message["current_file_minutes_section"] = section_title
                        logging.info(f"[{file_job_id}] Minutes progress: {clamped_percentage:.1f}% ({'sekcja: ' + section_title if section_title else 'sekcja zakończona'})")
                        asyncio.create_task(progress_queue.put(progress_message))

                    minutes_task = asyncio.create_task(
                        generate_and_save_minutes(
                            collection_name=collection_name,
                            original_source_path=Path(file_path),
                            custom_prompt=file_params.get('custom_prompt', None),
                            output_name=file_params.get('output_name', None),
                            progress_callback=minutes_progress_update_callback,
                            file_id=file_id,
                            gemini_api_key=(file_params.get('provider_tokens') or {}).get('gemini'),
                            metadata=file_params.get('transcription_metadata'),
                        )
                    )

                    try:
                        while not minutes_task.done():
                            async for msg in flush_message_queue():
                                yield msg
                            await asyncio.sleep(0.1)

                        minutes_content_response, minutes_filename, minutes_db_id = await minutes_task
                    except Exception:
                        if not minutes_task.done():
                            minutes_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await minutes_task
                        raise

                    async for msg in flush_message_queue():
                        yield msg

                    if minutes_content_response:
                        logging.info(f"[{file_job_id}] Minutes generated successfully for '{original_filename}'. Minutes ID: {minutes_db_id}")
                    else:
                        logging.error(f"[{file_job_id}] Minutes generation failed for '{original_filename}'.")
                        await set_file_status("error", "Minutes generation failed")
                        minutes_db_id = None
                        raise HTTPException(status_code=500, detail="Minutes generation failed.")

                    try:
                        logging.info(f"[{file_job_id}] Attempting to add meeting data for file_id: {file_id}")
                        await set_file_status("finalizing", "Adding meeting data")
                        yield f'event: message\ndata: {json.dumps({"message": f"Adding meeting data for {original_filename}", "status_type": "info", "current_file_progress_percentage": 95})}\n\n'
                        await asyncio.sleep(0.01)
                        async for msg in flush_message_queue():
                            yield msg

                        meeting_date = await get_relevant_date_for_file(file_path)
                        if file_id is not None and transcript_id is not None and minutes_db_id is not None and collection_name is not None:
                            await run_in_threadpool(agent_db.add_meeting_data, file_id, transcript_id, minutes_db_id, meeting_date, collection_name)
                            logging.info(f"[{file_job_id}] Meeting data added to database for file_id: {file_id}")
                        else:
                            missing_info = []
                            if file_id is None:
                                missing_info.append("File ID")
                            if transcript_id is None:
                                missing_info.append("Transcript ID")
                            if minutes_db_id is None:
                                missing_info.append("Minutes ID")
                            if collection_name is None:
                                missing_info.append("Collection Name")
                            logging.warning(f"[{file_job_id}] Could not add meeting data to database due to missing information: {', '.join(missing_info)}")
                            await set_file_status("error", "Incomplete data for meeting_data entry")
                            raise ValueError("Incomplete data for meeting_data entry.")
                    except Exception as e:
                        logging.error(f"[{file_job_id}] Error adding meeting data to database: {e}", exc_info=True)
                        await set_file_status("error", f"Database error during meeting data creation: {e}")
                        raise HTTPException(status_code=500, detail=f"Database error during meeting data creation: {e}")

                    completed_files += 1

                    await set_file_status("completed", "Processing successful")
                    logging.info(f"[{file_job_id}] File '{original_filename}' processing completed successfully. Status updated to 'completed'.")

                    file_complete_message = {
                        "message": f"File {original_filename} processed successfully!",
                        "status_type": "success",
                        "file_job_id": file_job_id,
                        "file_id": file_id,
                        "transcription_filename": str(transcription_filename),
                        "minutes_filename": str(minutes_filename),
                        "transcription": [seg.model_dump() for seg in transcription_segments] if transcription_segments else [],
                        "minutes": minutes_content_response.model_dump() if hasattr(minutes_content_response, 'model_dump') else minutes_content_response,
                        "original_filename": original_filename,
                        "batch_progress": {
                            "completed_files": completed_files,
                            "total_files": total_files,
                            "current_file_name": original_filename,
                            "file_job_id": file_job_id
                        },
                        "current_file_progress_percentage": 100
                    }
                    yield f'event: message\ndata: {json.dumps(file_complete_message)}\n\n'
                    await asyncio.sleep(0.05)
                    async for msg in flush_message_queue():
                        yield msg

                except Exception as e:
                    logging.error(f"[{file_job_id}] Processing failed for file job {file_job_id} in batch {batch_job_id}: {e}", exc_info=True)
                    if file_id is not None:
                        current_status = await run_in_threadpool(agent_db.get_file_status_by_id, file_id)
                    elif file_hash:
                        current_status = await run_in_threadpool(agent_db.get_file_status, file_hash)
                    else:
                        current_status = None
                    if current_status != "completed":
                        await set_file_status("error", f"Processing failed: {str(e)}")
                        logging.info(f"[{file_job_id}] File '{original_filename}' status updated to 'error' due to exception.")

                    error_message = {
                        "message": f"Processing failed for {original_filename}: {str(e)}",
                        "status_type": "error",
                        "file_job_id": file_job_id,
                        "original_filename": original_filename,
                        "batch_progress": {
                            "completed_files": completed_files,
                            "total_files": total_files,
                            "current_file_name": original_filename,
                            "file_job_id": file_job_id
                        },
                        "current_file_progress_percentage": 0
                    }
                    yield f'event: message\ndata: {json.dumps(error_message)}\n\n'
                    await asyncio.sleep(0.05)
                    async for msg in flush_message_queue():
                        yield msg

                finally:
                    logging.info(f"[{file_job_id}] Cleaning up temporary directory for file job.")
                    await cleanup_temp_dir(file_job_id)

            await progress_queue.put(None)
            async for msg in flush_message_queue():
                yield msg

            ram_gb = await get_current_ram_usage_gb()
            if ram_gb > 0:
                logging.info(f"Final RAM usage after processing batch job {batch_job_id}: {ram_gb:.2f} GB")
            else:
                logging.warning(f"RAM usage could not be retrieved after batch job {batch_job_id}.")

            current_vram_usage = await get_current_vram_usage_gb()
            if current_vram_usage > 0:
                logging.info(f"Final VRAM usage after processing batch job {batch_job_id}: {current_vram_usage:.2f} GB")
            else:
                logging.warning(f"VRAM usage could not be retrieved after batch job {batch_job_id}.")

            final_message_data = {
                "batch_complete": True,
                "message": "Batch processing complete!",
                "status_type": "success",
                "batch_progress": {
                    "completed_files": total_files,
                    "total_files": total_files
                }
            }
            yield f"event: message\ndata: {json.dumps(final_message_data)}\n\n"
            await asyncio.sleep(2)
            job_temp_storage.pop(batch_job_id, None)
            logging.info(f"DEBUG: Batch job {batch_job_id} removed from job_temp_storage after batch_complete. Stream closing soon.")

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
            yield f"event: message\ndata: {json.dumps(error_message)}\n\n"
            await asyncio.sleep(0.5)
        finally:
            if listener_task:
                listener_task.cancel()
            logging.info(f"[{batch_job_id}] Batch event generator finished.")

    return batch_event_generator


async def run_batch_processing(batch_job_id: str) -> None:
    logging.info(f"[{batch_job_id}] run_batch_processing invoked.")
    state = initialize_batch_stream_state(batch_job_id)

    # WAŻNE: Zaktualizuj status batcha na "processing" na początku
    # To pozwoli endpointowi streamingu wykryć, że batch jest przetwarzany przez Dramatiq worker
    try:
        await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, "processing")
        logging.info(f"[{batch_job_id}] Batch status zaktualizowany na 'processing' w bazie danych.")
    except Exception as e:
        logging.warning(f"[{batch_job_id}] Nie udało się zaktualizować statusu batcha na 'processing': {e}")

    batch_job_info = job_temp_storage.get(batch_job_id)
    if not batch_job_info:
        logging.error(f"[{batch_job_id}] Batch job info not found in job_temp_storage during background start.")
        error_message = {
            "batch_complete": True,
            "message": f"Batch job {batch_job_id} not found.",
            "status_type": "error"
        }
        payload = f"event: message\ndata: {json.dumps(error_message)}\n\n"
        _enqueue_batch_message(state, payload)
        finalize_batch_stream(batch_job_id)
        return

    file_job_ids = batch_job_info.get('file_job_ids', [])
    if not file_job_ids:
        logging.error(f"[{batch_job_id}] No file_job_ids found for batch during background start.")
        error_message = {
            "batch_complete": True,
            "message": f"Batch job {batch_job_id} has no files to process.",
            "status_type": "error"
        }
        payload = f"event: message\ndata: {json.dumps(error_message)}\n\n"
        _enqueue_batch_message(state, payload)
        finalize_batch_stream(batch_job_id)
        return

    generator_factory = create_batch_event_generator(batch_job_id, file_job_ids)

    async def finalize_batch_status_for_db():
        logging.info(f"[{batch_job_id}] Background processing finished. Updating final batch status in DB.")
        final_batch_status_for_db = "completed"
        any_file_error = False
        any_file_processing = False
        batch_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)
        file_ids_for_status: List[int] = []
        if batch_record and batch_record.get("file_ids"):
            file_ids_for_status = batch_record["file_ids"]
            logging.info(f"[{batch_job_id}] Znaleziono {len(file_ids_for_status)} file_ids w batch_record")
        else:
            logging.info(f"[{batch_job_id}] Brak file_ids w batch_record, szukam w job_temp_storage")
            for file_job_id in file_job_ids:
                file_info = job_temp_storage.get(file_job_id)
                if file_info and file_info.get("db_file_id") is not None:
                    file_ids_for_status.append(int(file_info["db_file_id"]))
                else:
                    record = await run_in_threadpool(agent_db.get_file_record_by_hash, file_job_id)
                    if record and record.get("id") is not None:
                        file_ids_for_status.append(int(record["id"]))

        logging.info(f"[{batch_job_id}] Sprawdzam status {len(file_ids_for_status)} plików")
        for file_id_for_status in file_ids_for_status:
            file_record_from_db = await run_in_threadpool(agent_db.get_file_record_by_id, file_id_for_status)
            if file_record_from_db:
                file_status = file_record_from_db.get('status', 'unknown')
                logging.info(f"[{batch_job_id}] Plik {file_id_for_status} ma status: {file_status}")
                if file_status == "error":
                    any_file_error = True
                    logging.warning(f"[{batch_job_id}] Plik {file_id_for_status} ma status 'error'")
                    break
                elif file_status != "completed":
                    any_file_processing = True
                    logging.info(f"[{batch_job_id}] Plik {file_id_for_status} jeszcze nie zakończony (status: {file_status})")
            else:
                logging.warning(f"[{batch_job_id}] Nie znaleziono rekordu pliku {file_id_for_status} w bazie")

        if any_file_error:
            final_batch_status_for_db = "error"
            logging.warning(f"[{batch_job_id}] Batch zakończy się statusem 'error' (co najmniej jeden plik ma błąd)")
        elif any_file_processing:
            final_batch_status_for_db = "processing"
            logging.warning(f"[{batch_job_id}] Batch pozostanie w statusie 'processing' (nie wszystkie pliki są completed)")
        else:
            logging.info(f"[{batch_job_id}] Wszystkie pliki są completed, batch będzie 'completed'")

        await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, final_batch_status_for_db)
        logging.info(f"[{batch_job_id}] ✅ Final batch status updated to '{final_batch_status_for_db}' in DB (background).")

    try:
        await _consume_batch_generator(batch_job_id, generator_factory, finalize_batch_status_for_db)
    except Exception as e:
        logging.error(f"[{batch_job_id}] Exception in run_batch_processing: {e}", exc_info=True)


# Inicjalizacja kontekstu do hash'owania haseł
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto") # ZMIENIONO NA PBKDF2_SHA256

# Schemat OAuth2 dla tokenów Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Endpoint, gdzie klient może uzyskać token

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    logging.debug(f"[AUTH] Created JWT token for user: {data.get('sub')}, token: {encoded_jwt}") # DODANO
    return encoded_jwt

# Zaktualizowana funkcja get_current_user, która faktycznie weryfikuje token JWT (dla nagłówków)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    logging.debug(f"[AUTH] Attempting to get current user. Received token (truncated): {token[:10]}...") # DODANO
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logging.warning("[AUTH] Token payload did not contain 'sub' (username).") # DODANO
            raise credentials_exception
        logging.debug(f"[AUTH] Successfully decoded token for user: {username}") # DODANO
        return {"username": username}
    except JWTError as e:
        logging.warning(f"[AUTH] JWTError during token decoding: {e}", exc_info=True) # DODANO
        raise credentials_exception

# NOWA FUNKCJA: do uzyskiwania aktualnego użytkownika z tokena z parametru zapytania (dla SSE)
async def get_current_user_from_query(token: str = Query(..., alias="token")):
    logging.debug(f"[AUTH-QUERY] Attempting to get current user from query token (truncated): {token[:10]}...") # DODANO
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logging.warning("[AUTH-QUERY] Token payload did not contain 'sub' (username).") # DODANO
            raise credentials_exception
        logging.debug(f"[AUTH-QUERY] Successfully decoded token for user: {username}") # DODANO
        return {"username": username}
    except JWTError as e:
        logging.warning(f"[AUTH-QUERY] JWTError during token decoding: {e}", exc_info=True) # DODANO
        raise credentials_exception


async def _resolve_request_user(request: Request, token_query: Optional[str]) -> Dict[str, str]:
    if token_query:
        return await get_current_user_from_query(token_query)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        raw_token = auth_header.split(" ", 1)[1].strip()
        if raw_token:
            return await get_current_user(raw_token)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get(
    "/transcription/providers/",
    dependencies=[Depends(get_current_user)],
)
async def get_transcription_providers() -> List[Dict[str, Any]]:
    """
    Zwraca katalog dostępnych dostawców i modeli transkrypcji.
    """
    return get_transcription_provider_catalog()

@app.get("/", dependencies=[Depends(get_current_user)]) # Przywrócono zależność
async def read_root():
    return {"message": "Welcome to the API!"}

# NOWY ENDPOINT: do uzyskiwania tokena JWT
@app.post("/token")
async def login_for_access_token(request: Request): # ZMIENIONO: Przyjmuje obiekt Request
    logging.debug("DEBUG: /token endpoint reached with raw Request object.") # DODANO
    try:
        form = await request.form() # Odczyt danych formularza
        username = form.get("username")
        password = form.get("password")
        logging.debug(f"[LOGIN-RAW] Received username: {username}, password: {password[:5]}...") # Przywrócono skrócone hasło

        if not username or not password:
            logging.warning("[LOGIN-RAW] Missing username or password in form data.") # DODANO
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing username or password",
            )

        user = agent_db.get_user(username)
        if not user:
            logging.warning(f"[LOGIN] User {username} not found in DB.") # DODANO
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logging.debug(f"[LOGIN] User found: {user['username']}, Hashed DB password: {user['hashed_password'][:10]}...") # Przywrócono skrócone hasło hashowane
        if not pwd_context.verify(password, user["hashed_password"]):
            logging.warning(f"[LOGIN] Password verification failed for user: {username}") # DODANO
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logging.debug(f"[LOGIN] Password verification successful for user: {username}") # DODANO
        access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logging.error(f"[LOGIN-RAW] Error during login processing: {e}", exc_info=True) # DODANO
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during login") # DODANO

@app.post("/register")
async def register_user(form_data: OAuth2PasswordRequestForm = Depends()):
    password_bytes = form_data.password.encode('utf-8')
    truncated_password_bytes = password_bytes[:72]
    hashed_password = pwd_context.hash(truncated_password_bytes) # Przekazujemy bajty bezpośrednio
    user_id = agent_db.create_user(form_data.username, hashed_password)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    return {"message": "User registered successfully", "user_id": user_id}

# Ustawienie podstawowej konfiguracji logowania (NIE RUSZAĆ! Zarządzane przez run.py)
logging.getLogger("uvicorn").setLevel(logging.WARNING) # Ogólny logger Uvicorn
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # Często źródło logów DEBUG z HTTPX
logging.getLogger("qdrant_client").setLevel(logging.INFO)
logging.getLogger("passlib").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("speechbrain").setLevel(logging.WARNING) # Zmieniono na WARNING
logging.getLogger("transformers").setLevel(logging.WARNING) # Zmieniono na WARNING
logging.getLogger("huggingface_hub").setLevel(logging.INFO)
logging.getLogger("torchaudio").setLevel(logging.WARNING) # Dla logów FFmpeg

# Modele Pydantic dla zapytania i odpowiedzi czatu
class ChatQuery(BaseModel):
    query: str
    collection_name: Optional[str] = None
    username: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

    # Funkcja do generowania odpowiedzi LLM na podstawie kontekstu i zapytania (PRZENIESIONA Z CHAT_AGENT_APP.PY)
    # async def generate_chat_response(user_query: str, context_documents: List[str]) -> str:
    #     """
    #     Generuje odpowiedź Agenta na podstawie zapytania użytkownika i dostarczonych dokumentów kontekstowych.
    #     """
    #     if not gemini_model:
    #         logging.error("Gemini model not initialized. Cannot generate chat response.")
    #         return "Przepraszam, model AI nie jest dostępny. Spróbuj ponownie później."
    #
    #     context_str = "\n".join(context_documents)
    #     if not context_str:
    #         return "Nie znalazłem żadnych informacji w bazie danych, które odpowiadałyby na Twoje pytanie."
    #
    #     prompt = f"""
    # Jesteś pomocnym asystentem AI. Twoim zadaniem jest odpowiadanie na pytania użytkownika,
    # wykorzystując *wyłącznie* informacje zawarte w podanym fragmencie tekstu.
    # Jeśli informacja nie znajduje się w tekście, odpowiedz, że nie możesz znaleźć odpowiedzi na podstawie dostępnych danych.
    # Nie wymyślaj informacji. Odpowiadaj zwięźle i na temat.
    #
    # **Zapytanie Użytkownika:**
    # {user_query}
    #
    # **Dostępny Kontekst (fragmenty transkrypcji/protokołów):**
    # ---
    # {context_str}
    # ---
    #
    # **Twoja Odpowiedź (oparta wyłącznie na Kontekście):**
    # """
    #     try:
    #         response = await asyncio.to_thread(gemini_model.invoke, prompt)
    #         return response.content.strip()
    #     except Exception as e:
    #         logging.error(f"Error generating chat response with Gemini: {e}")
    #         return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
    #
    #

# Zmieniony endpoint, aby serwować index.html dla wszystkich ścieżek SPA,
# które nie są ścieżkami API. Upewnij się, że katalog 'dist' istnieje po zbudowaniu frontendu.
# @app.get("/{full_path:path}", response_class=HTMLResponse)
# async def serve_spa(full_path: str):
#     html_file = Path("frontend/dist/index.html")
#     if not html_file.exists():
#         raise HTTPException(status_code=404, detail="Frontend index.html not found. Did you build the frontend?")
#     with open(html_file, "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read(), status_code=200)


# NEW: Mount a static directory to serve index.html (Już jest)
app.mount("/static", StaticFiles(directory="."), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     with open("index.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read(), status_code=200)

# NOWY ENDPOINT: Do serwowania interfejsu czatu
# @app.get("/chat_ui", response_class=HTMLResponse)
# async def serve_chat_interface():
#     html_file_path = Path("chat_interface.html")
#     if not html_file_path.exists():
#         raise HTTPException(status_code=404, detail="Chat interface HTML file not found.")
#     with open(html_file_path, "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read(), status_code=200)


# NOWY ENDPOINT: Przeniesiony z chat_agent_app.py
# @app.post("/chat/query", response_model=ChatResponse)
# async def chat_query_endpoint(chat_query: ChatQuery):
#     """
#     Endpoint do wysyłania zapytań do Agenta Czatowego.
#     """
#     user_query = chat_query.query
#     collection_name = chat_query.collection_name
#     logging.info(f"Received chat query: '{user_query}' for collection: {collection_name}")
#
#     try:
#         relevant_documents = await search_all_collections(
#             user_query,
#             limit_per_collection=config.QDRANT_SEARCH_LIMIT_PER_COLLECTION,
#             total_limit=config.QDRANT_SEARCH_TOTAL_LIMIT,
#             target_collection=collection_name
#         )
#
#         agent_response = await generate_chat_response(user_query, [doc.content for doc in relevant_documents])
#
#         return ChatResponse(response=agent_response)
#
#     except Exception as e:
#         logging.error(f"Error in chat_query_endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas przetwarzania Twojego zapytania: {e}")
#
# NOWY ENDPOINT: Przeniesiony z chat_agent_app.py
# @app.get("/get_qdrant_collections/")
# async def get_qdrant_collections_endpoint():
#     """Endpoint do pobierania listy nazw kolekcji Qdrant."""
#     try:
#         # Zapewnij inicjalizację Qdrant przed próbą pobrania kolekcji
#         await initialize_qdrant_resources() # Upewnij się, że klient Qdrant jest zainicjalizowany
#         collection_names = await get_all_collection_names()
#         return JSONResponse(content=collection_names)
#     except Exception as e:
#         logging.error(f"Error getting Qdrant collection names: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve Qdrant collections: {e}")

async def process_single_file_internal(
    file_job_id: str,
    file_path: Path,
    filename: str,
    username: str,
    transcription_provider: str,
    transcription_model: Optional[str] = None,
    output_name: Optional[str] = None,
    provider_tokens: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Wewnętrzna funkcja do przetwarzania pojedynczego pliku.
    Używana przez aktora Dramatiq.
    """
    file_id = None
    file_hash = None
    
    async def set_file_status(new_status: str, message: str = "") -> None:
        if file_id is not None:
            await run_in_threadpool(
                agent_db.update_file_status_by_id,
                file_id,
                new_status,
                message,
            )
        elif file_hash:
            await run_in_threadpool(
                agent_db.update_file_status,
                file_hash,
                new_status,
                message,
            )

    try:
        logging.info(f"[{file_job_id}] Starting internal processing for '{filename}'")

        # Compute file hash and add to processed_files table
        try:
            logging.info(f"[{file_job_id}] Attempting to compute file hash for {file_path}")
            file_hash = await run_in_threadpool(agent_db.compute_file_hash, file_path)
            logging.info(f"[{file_job_id}] Computed file hash: {file_hash}")
            logging.info(f"[{file_job_id}] Attempting to add file record to DB: {filename}, status 'processing'")
            file_id = await run_in_threadpool(
                agent_db.add_file,
                filename,
                str(file_path),
                file_hash,
                "processing",
                "",
                username,
            )
            logging.info(f"[{file_job_id}] File '{filename}' added to processed_files with ID: {file_id}")
        except Exception as e:
            logging.error(f"[{file_job_id}] Error adding file record to processed_files table: {e}", exc_info=True)
            raise

        logging.info(f"[{file_job_id}] Starting audio processing for '{filename}'...")
        def whisper_progress_update_callback(percentage: float):
            try:
                logging.debug(f"[{file_job_id}] Transcription progress: {percentage:.1f}%")
            except Exception:
                logging.debug(f"[{file_job_id}] Transcription progress: {percentage}%")
        
        transcription_metadata = _compose_transcription_metadata(transcription_provider, transcription_model)

        try:
            transcription_segments, transcription_filename, transcript_id = await process_audio(
                audio_path=file_path,
                file_job_id=file_job_id,
                transcription_model=transcription_model,
                transcription_provider=transcription_provider,
                progress_callback=whisper_progress_update_callback,
                username=username,
                file_id=file_id,
                provider_tokens=provider_tokens or {},
            )
        except TranscriptionProviderError as provider_exc:
            logging.error(f"[{file_job_id}] Transcription provider error: {provider_exc}")
            await set_file_status("error", f"Transcription provider error: {provider_exc}")
            raise
        if not transcription_segments:
            logging.error(f"[{file_job_id}] Audio processing failed or produced no segments for '{filename}'.")
            await set_file_status("error", "Audio processing failed")
            raise Exception("Audio processing failed.")
        logging.info(f"[{file_job_id}] Audio processing complete. Transcript ID: {transcript_id}")

        # Zapis segmentów do bazy
        if transcript_id is not None and transcription_segments:
            logging.info(f"[{file_job_id}] Adding {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")
            for seg in transcription_segments:
                await run_in_threadpool(agent_db.add_transcript_segment,
                    transcript_id,
                    seg.start,
                    seg.end,
                    seg.speaker,
                    seg.text
                )
            logging.info(f"[{file_job_id}] Added {len(transcription_segments)} segments to transcript_segments for transcript_id={transcript_id}")

        logging.info(f"[{file_job_id}] Starting ingestion to Qdrant for '{filename}'...")
        collection_name_for_file = await ingest_transcription(
            transcription_segments,
            file_path,
            file_job_id,
            transcript_id,
            config.CHUNK_DURATION,
            config.CHUNK_OVERLAP
        )
        if not collection_name_for_file:
            logging.error(f"[{file_job_id}] Ingestion to Qdrant failed for '{filename}'.")
            await set_file_status("error", "Ingestion to Qdrant failed")
            raise Exception("Ingestion to Qdrant failed.")
        collection_name = collection_name_for_file
        logging.info(f"[{file_job_id}] Ingested to Qdrant collection: {collection_name}")

        logging.info(f"[{file_job_id}] Generating and saving minutes for '{filename}'...")
        minutes_content_response, minutes_filename, minutes_db_id = await generate_and_save_minutes(
            collection_name=collection_name,
            original_source_path=file_path,
            custom_prompt=None,
            output_name=output_name,
            file_id=file_id,
            gemini_api_key=(provider_tokens or {}).get('gemini'),
            metadata=transcription_metadata,
        )

        if minutes_content_response:
            logging.info(f"[{file_job_id}] Minutes generated successfully for '{filename}'. Minutes ID: {minutes_db_id}")
            await set_file_status("completed", "Processing successful")
        else:
            logging.error(f"[{file_job_id}] Minutes generation failed for '{filename}'.")
            await set_file_status("error", "Minutes generation failed")
            raise Exception("Minutes generation failed.")

        # Add entry to meeting_data table
        try:
            logging.info(f"[{file_job_id}] Attempting to add meeting data for file_id: {file_id}")
            meeting_date = await get_relevant_date_for_file(file_path)
            if file_id is not None and transcript_id is not None and minutes_db_id is not None and collection_name is not None:
                await run_in_threadpool(agent_db.add_meeting_data, file_id, transcript_id, minutes_db_id, meeting_date, collection_name)
                logging.info(f"[{file_job_id}] Meeting data added to database for file_id: {file_id}")
            else:
                missing_info = []
                if file_id is None: missing_info.append("File ID")
                if transcript_id is None: missing_info.append("Transcript ID")
                if minutes_db_id is None: missing_info.append("Minutes ID")
                if collection_name is None: missing_info.append("Collection Name")
                logging.warning(f"[{file_job_id}] Could not add meeting data to database due to missing information: {', '.join(missing_info)}")
                await set_file_status("error", "Incomplete data for meeting_data entry")
                raise ValueError("Incomplete data for meeting_data entry.")
        except Exception as e:
            logging.error(f"[{file_job_id}] Error adding meeting data to database: {e}", exc_info=True)
            await set_file_status("error", f"Database error during meeting data creation: {e}")
            raise

        # Finalna aktualizacja statusu
        logging.info(f"[{file_job_id}] Final update: file {filename} status to 'completed'.")
        await set_file_status("completed", "Processing successful")
        logging.info(f"[{file_job_id}] File '{filename}' processing completed successfully.")
        
        return {
            "status": "success",
            "file_id": file_id,
            "collection_name": collection_name,
        }
    except Exception as e:
        logging.error(f"[{file_job_id}] Error during processing: {e}", exc_info=True)
        if file_id is not None:
            current_status = await run_in_threadpool(agent_db.get_file_status_by_id, file_id)
        elif file_hash:
            current_status = await run_in_threadpool(agent_db.get_file_status, file_hash)
        else:
            current_status = None
        if current_status != "completed":
            await set_file_status("error", f"Processing error: {str(e)}")
        raise


@app.post("/process_file/")
async def process_uploaded_file(
    current_user: Dict[str, str] = Depends(get_current_user),
    file: UploadFile = File(...),
    transcription_provider: str = Form(config.DEFAULT_TRANSCRIPTION_PROVIDER),
    transcription_model: Optional[str] = Form(None),
    output_name: Optional[str] = Form(None),
    provider_tokens_json: Optional[str] = Form(None),
):
    """
    Przetwarza pojedynczy plik audio/video.
    Teraz używa Dramatiq dla niezawodności - zadanie jest w kolejce i będzie przetworzone nawet po restarcie serwera.
    """
    logging.info(f"Starting process_uploaded_file for {file.filename}")

    try:
        job_id = str(uuid.uuid4())
        job_temp_storage[job_id] = {'type': 'file', 'batch_job_id': None, 'status': 'handling_upload'}
        logging.info(f"[{job_id}] Received file upload request for '{file.filename}'.")

        # Zapisz plik tymczasowo
        input_file_temp_path = await save_file_to_temp_and_convert_if_needed(file, job_id)
        if not input_file_temp_path:
            logging.error(f"[{job_id}] File upload or conversion failed for '{file.filename}'.")
            raise HTTPException(status_code=500, detail="File upload failed.")
        logging.info(f"[{job_id}] File '{file.filename}' saved to temporary path: {input_file_temp_path}")

        # Użyj Dramatiq do przetwarzania pliku
        try:
            provider_tokens = _parse_provider_tokens_json(provider_tokens_json)
            provider_tokens_json_str = json.dumps(provider_tokens) if provider_tokens else ""
            
            job_id_dramatiq = await enqueue_single_file_processing(
                file_job_id=job_id,
                file_path=str(input_file_temp_path),
                filename=file.filename,
                username=current_user["username"],
                transcription_provider=transcription_provider,
                transcription_model=transcription_model,
                output_name=output_name,
                provider_tokens_json=provider_tokens_json_str,
            )
            logging.info(f"[{job_id}] File dodany do kolejki Dramatiq: {job_id_dramatiq}")
            
            # Zwróć odpowiedź z job_id
            return JSONResponse(
                status_code=202,
                content={
                    "message": f"File '{file.filename}' received and queued for processing.",
                    "file_job_id": job_id,
                    "dramatiq_job_id": job_id_dramatiq,
                    "status": "queued"
                }
            )
        except Exception as e:
            logging.error(f"[{job_id}] Błąd podczas dodawania pliku do kolejki Dramatiq: {e}", exc_info=True)
            # Fallback - przetwórz synchronicznie jeśli Dramatiq nie działa
            logging.warning(f"[{job_id}] Używam fallback: synchroniczne przetwarzanie")
            try:
                provider_tokens = _parse_provider_tokens_json(provider_tokens_json)
                result = await process_single_file_internal(
                    file_job_id=job_id,
                    file_path=input_file_temp_path,
                    filename=file.filename,
                    username=current_user["username"],
                    transcription_provider=transcription_provider,
                    transcription_model=transcription_model,
                    output_name=output_name,
                    provider_tokens=provider_tokens,
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "File processed successfully (fallback mode)",
                        "file_job_id": job_id,
                        "file_id": result.get("file_id"),
                        "status": "completed"
                    }
                )
            except Exception as fallback_error:
                logging.error(f"[{job_id}] Fallback processing also failed: {fallback_error}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(fallback_error)}")
                
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[{job_id}] Unhandled Exception during file upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )


@app.get("/process_file/status/{file_job_id}")
async def get_file_processing_status(
    file_job_id: str,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    """
    Sprawdza status przetwarzania pojedynczego pliku.
    
    Zwraca:
        - status: 'queued', 'processing', 'completed', 'error'
        - file_id: ID pliku w bazie (jeśli dostępne)
        - dramatiq_status: Status zadania w Dramatiq (jeśli dostępne)
    """
    try:
        # Sprawdź status w bazie danych przez file_job_id (hash)
        file_record = await run_in_threadpool(agent_db.get_file_record_by_hash, file_job_id)
        
        if file_record:
            return {
                "file_job_id": file_job_id,
                "file_id": file_record.get("id"),
                "filename": file_record.get("filename"),
                "status": file_record.get("status", "unknown"),
                "message": file_record.get("message", ""),
            }
        
        # Jeśli nie ma w bazie, sprawdź czy jest w job_temp_storage
        file_info = job_temp_storage.get(file_job_id)
        if file_info:
            return {
                "file_job_id": file_job_id,
                "status": file_info.get("status", "unknown"),
                "message": "File is being processed",
            }
        
        # Jeśli nie ma nigdzie, sprawdź czy może być w Dramatiq
        # (możemy sprawdzić przez get_job_status jeśli mamy dramatiq_job_id)
        # Na razie zwróć not found
        raise HTTPException(status_code=404, detail=f"File job {file_job_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting file status for {file_job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting file status: {e}")


@app.post("/upload_multiple/")
async def upload_multiple_files_for_processing(
    current_user: Dict[str, str] = Depends(get_current_user),
    files: List[UploadFile] = File(...),
    transcription_provider: str = Form(config.DEFAULT_TRANSCRIPTION_PROVIDER),
    transcription_model: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    chunk_duration: int = Form(config.CHUNK_DURATION),
    chunk_overlap: int = Form(config.CHUNK_OVERLAP),
    output_name: Optional[str] = Form(None),
    provider_tokens_json: Optional[str] = Form(None),
):
    batch_job_id = str(uuid.uuid4())
    username = current_user["username"]

    provider_tokens = _parse_provider_tokens_json(provider_tokens_json)

    job_params_for_memory = {
        'transcription_provider': transcription_provider,
        'transcription_model': transcription_model,
        'custom_prompt': custom_prompt,
        'chunk_duration': chunk_duration,
        'chunk_overlap': chunk_overlap,
        'output_name': output_name,
        'username': username,
        'provider_tokens': dict(provider_tokens),
    }

    job_temp_storage[batch_job_id] = {
        'type': 'batch',
        'file_job_ids': [],
        'status': 'uploaded',
        'params': job_params_for_memory,
    }
    logging.info(f"Starting upload_multiple_files_for_processing for {len(files)} files. Batch ID: {batch_job_id}") # NOWY LOG
    
    # NOWE: Dodanie wpisu do tabeli batch_jobs
    try:
        params_for_db = {key: value for key, value in job_params_for_memory.items() if key != 'provider_tokens'}
        await run_in_threadpool(
            agent_db.add_batch_job,
            batch_job_id,
            "uploaded",
            params_for_db,
        )
        logging.info(f"[{batch_job_id}] Batch job added to database with status 'uploaded'.")
    except Exception as e:
        logging.error(f"[{batch_job_id}] Error adding batch job to database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error during batch job creation: {e}")

    uploaded_file_info = []

    try:
        for file in files:
            file_job_id = str(uuid.uuid4())
            job_temp_storage[batch_job_id]['file_job_ids'].append(file_job_id)
            job_temp_storage[file_job_id] = {
                'type': 'file',
                'batch_job_id': batch_job_id,
                'status': 'uploaded',
                'filename': file.filename
            }

            logging.info(f"[{file_job_id}] Processing file '{file.filename}' for batch {batch_job_id}.") # NOWY LOG

            temp_file_to_process_path = await save_file_to_temp_and_convert_if_needed(file, file_job_id)
            if not temp_file_to_process_path:
                logging.error(f"[{file_job_id}] File upload or conversion failed for '{file.filename}' in batch {batch_job_id}.") # NOWY LOG
                # Zaktualizuj status batcha na 'error'
                await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, "error")
                raise HTTPException(status_code=500, detail="File upload failed in batch.")
            logging.info(f"[{file_job_id}] File '{file.filename}' saved to temporary path: {temp_file_to_process_path} for batch {batch_job_id}.") # NOWY LOG

            # NOWE: Dodanie pliku do processed_files i zapisanie jego numerycznego ID
            logging.info(f"[{file_job_id}] Attempting to add file '{file.filename}' to processed_files table with status 'uploaded'.") # NOWY LOG
            file_id_db = await run_in_threadpool(
                agent_db.add_file,
                file.filename,
                str(temp_file_to_process_path),
                file_job_id,
                "uploaded",
                "",
                username,
            )
            job_temp_storage[file_job_id]['db_file_id'] = file_id_db # Zapisz numeryczne ID z bazy danych
            logging.info(f"[{file_job_id}] File '{file.filename}' added to processed_files with DB ID: {file_id_db} for batch {batch_job_id}.") # NOWY LOG

            job_temp_storage[file_job_id]['params'] = {
                 'transcription_provider': transcription_provider,
                 'transcription_model': transcription_model,
                 'custom_prompt': custom_prompt,
                 'chunk_duration': chunk_duration,
                 'chunk_overlap': chunk_overlap,
                 'output_name': output_name,
                 'username': username,
                 'provider_tokens': dict(provider_tokens),
            }
            job_temp_storage[file_job_id]['file_path'] = temp_file_to_process_path

            uploaded_file_info.append({
                "filename": file.filename,
                "file_job_id": file_job_id,
                "temp_path": str(temp_file_to_process_path)
            })
        
        # Aktualizuj listę file_job_ids w bazie danych dla batcha
        await run_in_threadpool(agent_db.update_batch_job_file_ids_json,
            batch_job_id,
            json.dumps(job_temp_storage[batch_job_id]['file_job_ids'])
        )
        logging.info(f"[{batch_job_id}] Updated file_job_ids in database for batch.")

        # Użyj Dramatiq zamiast asyncio.create_task()
        logging.info(f"[{batch_job_id}] Próbuję dodać batch do kolejki Dramatiq...")
        try:
            job_id = await enqueue_batch_processing(batch_job_id)
            logging.info(f"[{batch_job_id}] ✅ Batch dodany do kolejki Dramatiq: {job_id}")
        except Exception as e:
            logging.error(f"[{batch_job_id}] ❌ Błąd podczas dodawania batcha do kolejki Dramatiq: {e}", exc_info=True)
            # Fallback do starego sposobu jeśli Dramatiq nie działa
            logging.warning(f"[{batch_job_id}] Używam fallback: asyncio.create_task()")
            state = initialize_batch_stream_state(batch_job_id)
            if state.task and not state.task.done():
                logging.warning(f"[{batch_job_id}] Previous processing task still running; cancelling before starting a new one.")
                state.task.cancel()
            state.task = asyncio.create_task(run_batch_processing(batch_job_id))
            logging.info(f"[{batch_job_id}] Background processing task started (fallback).")

        logging.info(f"[{batch_job_id}] All files uploaded and ready for batch processing. Returning 202 Accepted.") # NOWY LOG
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
        logging.error(f"Error during upload for batch job {batch_job_id}: {str(e)}", exc_info=True)
        # Zaktualizuj status batcha na 'error' w przypadku ogólnego błędu
        await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, "error")
        if batch_job_id in job_temp_storage:
            logging.warning(f"[{batch_job_id}] Cleaning up temporary directories after batch upload failure.") # NOWY LOG
            for fj_id in job_temp_storage[batch_job_id].get('file_job_ids', []):
                 await cleanup_temp_dir(fj_id) # Asynchroniczne wywołanie
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
@app.get("/process_batch_status/stream/{batch_job_id}/", dependencies=[Depends(get_current_user_from_query)])
async def process_batch_status_stream(batch_job_id: str):
    logging.info(f"[{batch_job_id}] Initializing batch status stream.") # NOWY LOG
    try:
        batch_job_info = job_temp_storage.get(batch_job_id)
        if not batch_job_info:
            logging.error(f"[{batch_job_id}] Batch job info not found in job_temp_storage.") # NOWY LOG
            return JSONResponse(status_code=404, content={"detail": f"Batch job with ID {batch_job_id} not found."})
        logging.debug(
            "[%s] Retrieved batch_job_info keys: %s",
            batch_job_id,
            list(batch_job_info.keys()),
        ) # NOWY LOG

        file_job_ids = batch_job_info.get('file_job_ids', [])
        if not file_job_ids:
            logging.warning(f"[{batch_job_id}] No file_job_ids found for batch. Returning 400.") # NOWY LOG
            return JSONResponse(status_code=400, content={"detail": f"Batch job {batch_job_id} has no files to process."})
        logging.info(f"[{batch_job_id}] Found {len(file_job_ids)} files to process in batch.") # NOWY LOG

        batch_params = batch_job_info.get('params', {})
        batch_transcription_provider = batch_params.get('transcription_provider', config.DEFAULT_TRANSCRIPTION_PROVIDER)
        batch_transcription_model = batch_params.get('transcription_model')
        batch_custom_prompt = batch_params.get('custom_prompt', None)
        batch_output_name = batch_params.get('output_name', None)
        batch_chunk_duration = batch_params.get('chunk_duration', config.CHUNK_DURATION)
        batch_chunk_overlap = batch_params.get('chunk_overlap', config.CHUNK_OVERLAP)

        state = initialize_batch_stream_state(batch_job_id)

        generator_factory = create_batch_event_generator(batch_job_id, file_job_ids)

        async def finalize_batch_status_for_db():
            logging.info(f"[{batch_job_id}] Batch event generator finished. Updating final batch status in DB.") # NOWY LOG
            final_batch_status_for_db = "completed"
            any_file_error = False
            any_file_processing = False

            batch_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)
            file_ids_for_status: List[int] = []
            if batch_record and batch_record.get("file_ids"):
                file_ids_for_status = batch_record["file_ids"]
            else:
                for file_job_id in file_job_ids:
                    file_info = job_temp_storage.get(file_job_id)
                    if file_info and file_info.get("db_file_id") is not None:
                        file_ids_for_status.append(int(file_info["db_file_id"]))
                    else:
                        record = await run_in_threadpool(agent_db.get_file_record_by_hash, file_job_id)
                        if record and record.get("id") is not None:
                            file_ids_for_status.append(int(record["id"]))

            for file_id_for_status in file_ids_for_status:
                file_record_from_db = await run_in_threadpool(agent_db.get_file_record_by_id, file_id_for_status)
                if file_record_from_db:
                    if file_record_from_db['status'] == "error":
                        any_file_error = True
                        break
                    elif file_record_from_db['status'] != "completed":
                        any_file_processing = True
            
            if any_file_error:
                final_batch_status_for_db = "error"
            elif any_file_processing:
                final_batch_status_for_db = "processing"
            
            await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, final_batch_status_for_db)
            logging.info(f"[{batch_job_id}] Final batch status updated to '{final_batch_status_for_db}' in DB.") # NOWY LOG

        client_queue: asyncio.Queue[str] = asyncio.Queue()
        # WAŻNE: Dodaj client_queue do state.listeners PRZED uruchomieniem przetwarzania
        # To zapewni, że wszystkie wiadomości z batch_event_generator będą streamowane
        state.listeners.add(client_queue)
        logging.info(f"[{batch_job_id}] Client queue dodany do state.listeners. Liczba listenerów: {len(state.listeners)}")

        # Wyślij historię wiadomości (jeśli są)
        for payload in list(state.history):
            client_queue.put_nowait(payload)
            logging.debug(f"[{batch_job_id}] Wysłano wiadomość z historii do client_queue")

        if state.completed:
            client_queue.put_nowait(None)
            logging.info(f"[{batch_job_id}] Batch już zakończony, zamykam stream")

        # WAŻNE: Endpoint streamingu NIE uruchamia przetwarzania bezpośrednio!
        # Przetwarzanie jest wykonywane przez Dramatiq worker (z kolejki batch_processing).
        # 
        # Problem: Obecna implementacja uruchamia batch_event_generator, który przetwarza pliki bezpośrednio.
        # To oznacza, że przetwarzanie NIE idzie przez Dramatiq i nie jest odzyskiwane po restarcie.
        #
        # Rozwiązanie: Endpoint streamingu powinien monitorować status z bazy danych
        # (aktualizowany przez Dramatiq worker), a nie uruchamiać przetwarzanie.
        #
        # TODO: Refaktoryzacja - stworzyć nowy generator, który:
        # 1. Monitoruje status z bazy danych (polling)
        # 2. Streamuje aktualizacje statusu do klienta
        # 3. Dramatiq worker przetwarza batch i aktualizuje status w bazie
        #
        # Na razie: Endpoint NIE uruchamia przetwarzania, tylko informuje że batch jest w kolejce Dramatiq
        if (not state.task or state.task.done()) and not state.completed:
            logging.warning(f"[{batch_job_id}] ⚠️  UWAGA: Endpoint streamingu NIE uruchamia przetwarzania bezpośrednio.")
            logging.warning(f"[{batch_job_id}] ⚠️  Batch powinien być przetwarzany przez Dramatiq worker z kolejki.")
            logging.info(f"[{batch_job_id}] Sprawdzam status batcha w bazie danych...")
            
            # Sprawdź status batcha w bazie danych
            batch_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)
            if batch_record:
                batch_status = batch_record.get('status', 'unknown')
                logging.info(f"[{batch_job_id}] Batch status w bazie: {batch_status}")
                
                if batch_status == "uploaded":
                    logging.info(f"[{batch_job_id}] Batch czeka na przetworzenie przez Dramatiq worker.")
                    # Wyślij wiadomość do klienta, że batch jest w kolejce
                    queued_message = {
                        "message": f"Batch {batch_job_id} jest w kolejce Dramatiq i będzie przetworzony przez worker.",
                        "status_type": "info",
                        "batch_status": "queued"
                    }
                    _enqueue_batch_message(state, f"event: message\ndata: {json.dumps(queued_message)}\n\n")
                elif batch_status == "processing":
                    logging.info(f"[{batch_job_id}] Batch jest przetwarzany przez Dramatiq worker.")
                elif batch_status == "completed":
                    logging.info(f"[{batch_job_id}] Batch został już przetworzony.")
                    state.completed = True
                elif batch_status == "error":
                    logging.warning(f"[{batch_job_id}] Batch zakończył się błędem.")
                    state.completed = True
            
            # NIE uruchamiaj batch_event_generator - to uruchamia przetwarzanie bezpośrednio!
            # Zamiast tego, endpoint będzie monitorował status z bazy danych
            # Dramatiq worker przetworzy batch i zaktualizuje status w bazie
            logging.warning(f"[{batch_job_id}] ⚠️  Endpoint streamingu NIE uruchamia przetwarzania - czeka na Dramatiq worker.")
            
            # Stwórz task, który będzie monitorował status z bazy danych
            async def monitor_batch_status_from_db():
                """Monitoruje status batcha z bazy danych i streamuje aktualizacje."""
                last_status = None
                poll_interval = 2.0  # Sprawdzaj co 2 sekundy
                
                while not state.completed:
                    try:
                        batch_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)
                        if batch_record:
                            current_status = batch_record.get('status', 'unknown')
                            
                            if current_status != last_status:
                                logging.info(f"[{batch_job_id}] Status zmieniony: {last_status} -> {current_status}")
                                status_message = {
                                    "message": f"Batch status: {current_status}",
                                    "status_type": "info" if current_status != "error" else "error",
                                    "batch_status": current_status
                                }
                                _enqueue_batch_message(state, f"event: message\ndata: {json.dumps(status_message)}\n\n")
                                last_status = current_status
                                
                                if current_status in ["completed", "error"]:
                                    state.completed = True
                                    await finalize_batch_status_for_db()
                                    _enqueue_batch_message(state, None)  # Zakończ stream
                                    break
                        
                        await asyncio.sleep(poll_interval)
                    except Exception as e:
                        logging.error(f"[{batch_job_id}] Błąd podczas monitorowania statusu: {e}", exc_info=True)
                        await asyncio.sleep(poll_interval)
            
            # Uruchom monitoring statusu z bazy danych
            state.task = asyncio.create_task(monitor_batch_status_from_db())

        async def event_stream():
            try:
                last_heartbeat = time.time()
                heartbeat_interval = 30.0  # Wysyłaj heartbeat co 30 sekund
                
                while True:
                    try:
                        # Czekaj na wiadomość z timeoutem (żeby móc wysłać heartbeat)
                        message = await asyncio.wait_for(client_queue.get(), timeout=heartbeat_interval)
                        if message is None:
                            break
                        yield message
                        last_heartbeat = time.time()
                    except asyncio.TimeoutError:
                        # Timeout - wyślij heartbeat, żeby utrzymać połączenie
                        current_time = time.time()
                        if current_time - last_heartbeat >= heartbeat_interval:
                            heartbeat_message = f": heartbeat\n\n"  # SSE comment (zaczyna się od :)
                            yield heartbeat_message
                            last_heartbeat = current_time
            except asyncio.CancelledError:
                logging.info(f"[{batch_job_id}] Event stream cancelled")
                raise
            except Exception as e:
                logging.error(f"[{batch_job_id}] Error in event_stream: {e}", exc_info=True)
                raise
            finally:
                state.listeners.discard(client_queue)
                logging.info(f"[{batch_job_id}] Event stream closed, removed client_queue from listeners")

        logging.info(f"Successfully prepared StreamingResponse for batch_job_id: {batch_job_id}") # NOWY LOG
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Unhandled exception in process_batch_status_stream for batch {batch_job_id}: {e}", exc_info=True)
        # Zaktualizuj status batcha na 'error' w bazie danych w przypadku nieobsłużonego wyjątku
        await run_in_threadpool(agent_db.update_batch_job_status, batch_job_id, "error")
        logging.error(f"[{batch_job_id}] Batch status updated to 'error' in DB due to unhandled exception.") # NOWY LOG
        if 'client_queue' in locals() and 'state' in locals():
            state.listeners.discard(client_queue)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error during batch stream setup: {str(e)}"}
        )

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Loading models and enrolled speakers...")
    try:
        # USUNIĘTO: agent_db.init_db() - Inicjalizacja bazy danych SQLite będzie zarządzana przez Database Agent.
        logging.info("Attempting to initialize SQLite database...")
        agent_db.init_db() # PRZYWRÓCONO!
        logging.info("SQLite database initialized successfully during startup.")

        logging.info("Attempting to initialize Qdrant client and embedding model...")
        await initialize_qdrant_resources()
        logging.info("Qdrant client and embedding model initialized successfully during startup.")

        logging.info("Attempting to load diarization and enrolled speakers models...")
        from utils import get_diarization_models, load_enrolled_speakers
        await get_diarization_models()
        logging.info("Diarization and embedding models loaded.")

        await load_enrolled_speakers() # Load shared namespace speaker profiles (if any)
        logging.info("Shared enrolled speaker profiles initialized during startup.")
        logging.info("Enrolled speakers profiles initialized during startup.")

        # Inicjalizacja systemu kolejkowania wiadomości (PostgreSQL)
        async def process_message_callback(query: str, collection_name: Optional[str], username: str) -> str:
            """Callback do przetwarzania wiadomości z kolejki."""
            try:
                # Wyszukaj dokumenty w Qdrant (jak w zakomentowanym kodzie)
                relevant_documents = await search_all_collections(
                    query,
                    limit_per_collection=config.QDRANT_SEARCH_LIMIT_PER_COLLECTION,
                    total_limit=config.QDRANT_SEARCH_TOTAL_LIMIT,
                    target_collection=collection_name
                )
                
                # Przygotuj kontekst z dokumentów
                context_documents = [doc.content for doc in relevant_documents]
                context_str = "\n".join(context_documents) if context_documents else ""
                
                # Wygeneruj odpowiedź używając generate_chat_response
                response = await generate_chat_response(query, context_str)
                return response
            except Exception as e:
                logging.error(f"Błąd podczas przetwarzania wiadomości: {e}", exc_info=True)
                raise
        
        # Uruchom system kolejkowania
        await start_boss()
        
        # Zarejestruj handler do przetwarzania wiadomości
        register_message_handler(process_message_callback)
        
        logging.info("✅ System kolejkowania wiadomości zainicjalizowany i worker uruchomiony.")

        # Po restarcie: dodaj batchy ze statusem "processing" lub "uploaded" do kolejki Dramatiq
        try:
            logging.info("Sprawdzam batchy do ponownego dodania do kolejki Dramatiq...")
            from db.agent_db import pg_db
            pending_batches = await run_in_threadpool(
                lambda: pg_db.execute(
                    "SELECT batch_job_id FROM batch_jobs WHERE status IN ('processing', 'uploaded') ORDER BY created_at DESC",
                    fetch="all"
                ) or []
            )
            
            if pending_batches:
                logging.info(f"Znaleziono {len(pending_batches)} batchów do ponownego dodania do kolejki Dramatiq")
                for batch_record in pending_batches:
                    batch_job_id = batch_record.get("batch_job_id")
                    if batch_job_id:
                        try:
                            # Pobierz informacje o batchu z bazy i przywróć do job_temp_storage
                            batch_db_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)
                            if batch_db_record:
                                file_job_ids_from_db = json.loads(batch_db_record.get('file_job_ids_json', '[]'))
                                params_from_db = batch_db_record.get('params', {})
                                
                                # Przywróć do job_temp_storage
                                job_temp_storage[batch_job_id] = {
                                    'type': 'batch',
                                    'file_job_ids': file_job_ids_from_db,
                                    'status': batch_db_record.get('status', 'uploaded'),
                                    'params': params_from_db
                                }
                                
                                # Dodaj do kolejki Dramatiq
                                job_id = await enqueue_batch_processing(batch_job_id)
                                logging.info(f"[{batch_job_id}] ✅ Batch ponownie dodany do kolejki Dramatiq po restarcie: {job_id}")
                            else:
                                logging.warning(f"[{batch_job_id}] Nie można pobrać informacji o batchu z bazy danych")
                        except Exception as e:
                            logging.error(f"[{batch_job_id}] Błąd podczas ponownego dodawania batcha do kolejki: {e}", exc_info=True)
            else:
                logging.info("Brak batchów do ponownego dodania do kolejki Dramatiq")
        except Exception as e:
            logging.warning(f"Błąd podczas sprawdzania batchów do ponownego dodania: {e}", exc_info=True)

        logging.info("All startup tasks completed.")

    except Exception as e:
        logging.error(f"FATAL ERROR during startup: {e}", exc_info=True)
        # Re-raise the exception to make sure the app doesn't start in a broken state
        raise

class PromptSet(BaseModel):
    name: str
    prompts: Dict[str, str]


class RegenerateMinutesPayload(BaseModel):
    custom_prompt: Optional[str] = None
    output_name: Optional[str] = None
    provider_tokens: Optional[Dict[str, str]] = None

@app.get("/prompts/list")
async def list_prompts_endpoint(current_user: Dict[str, str] = Depends(get_current_user)):
    try:
        username = current_user["username"]
        prompts = await load_prompt_config(username=username, include_defaults=True)
        user_only_prompts = await load_prompt_config(username=username, include_defaults=False)
        default_names = await get_default_prompt_names()
        readonly_names = [name for name in default_names if name not in user_only_prompts]
        editable_names = list(user_only_prompts.keys())
        return JSONResponse(
            content={
                "prompts": prompts,
                "readonly": readonly_names,
                "editable": editable_names,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prompts: {e}")

@app.post("/prompts/save")
async def save_prompts_endpoint(
    prompt_set: PromptSet,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    try:
        username = current_user["username"]
        success = await save_prompt_config(prompt_set.name, prompt_set.prompts, username=username)
        if not success:
            raise HTTPException(status_code=400, detail="Cannot save empty prompt set.")
        return {"message": "Prompt set saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prompts: {e}")

@app.delete("/prompts/delete/{prompt_name}")
async def delete_prompts_endpoint(
    prompt_name: str,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    try:
        username = current_user["username"]
        deleted = await delete_prompt_config(prompt_name, username=username)
        if not deleted:
            raise HTTPException(status_code=400, detail="Nie można usunąć wskazanego zestawu promptów.")
        return {"message": "Prompt set deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prompts: {e}")

# --- NOWE ENDPOINTY DLA IDENYTFIKACJI MÓWCÓW ---

@app.post("/enroll_speaker_direct/")
async def enroll_speaker_direct_endpoint(
    current_user: Dict[str, str] = Depends(get_current_user),
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

    username = current_user["username"]

    try:
        audio_path = await save_uploaded_file_temp(audio_file, enrollment_job_id, temp_dir_obj, temp_dir_path)
        # Zmieniono wywołanie funkcji, aby użyć await bezpośrednio, bo enroll_speaker_from_audio jest już async
        enrollment_success = await enroll_speaker_from_audio(audio_path, speaker_name, username=username)

        if enrollment_success:
            logging.info(f"Speaker '{speaker_name}' enrolled successfully from {audio_path}")
            # Przeładowanie mówców po zapisaniu, aby lista na froncie była aktualna
            await load_enrolled_speakers(username=username)
            return JSONResponse(status_code=200, content={"message": f"Speaker '{speaker_name}' enrolled successfully."})
        else:
            logging.error(f"Failed to enroll speaker '{speaker_name}' from {audio_path}")
            raise HTTPException(status_code=500, detail=f"Failed to enroll speaker '{speaker_name}'. Check logs for details.")

    except Exception as e:
        logging.error(f"Error during direct speaker enrollment for '{speaker_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker enrollment: {str(e)}")
    finally:
        if temp_dir_obj:
            await cleanup_temp_dir(enrollment_job_id) # Asynchroniczne sprzątanie
            # temp_dir_obj.cleanup() # Ta linia jest zbędna, cleanup_temp_dir obsługuje to
            logging.info(f"Temporary directory for enrollment job {enrollment_job_id} cleaned up.")

@app.delete("/delete_speaker/{speaker_name}")
async def delete_speaker_endpoint(
    speaker_name: str,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    if not speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name is required.")

    username = current_user["username"]

    try:
        success = await delete_speaker(speaker_name, username=username)
        if success:
            await load_enrolled_speakers(username=username)
            return JSONResponse(status_code=200, content={"message": f"Speaker '{speaker_name}' deleted successfully."})
        else:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found or could not be deleted.")
    except Exception as e:
        logging.error(f"Error deleting speaker '{speaker_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker deletion: {str(e)}")


@app.get("/user/transcripts/")
async def list_user_transcripts_endpoint(
    current_user: Dict[str, str] = Depends(get_current_user),
):
    username = current_user["username"]
    records = await run_in_threadpool(agent_db.get_processed_files_for_user, username)
    return JSONResponse(content=records)


@app.delete("/user/transcripts/{file_id}")
async def delete_user_transcript_endpoint(
    file_id: int,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    username = current_user["username"]
    file_record = await run_in_threadpool(agent_db.get_file_record_by_id, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="Transkrypcja nie została znaleziona lub nie należy do użytkownika.")

    owner_username = file_record.get("owner_username")
    if owner_username is not None and owner_username != username:
        raise HTTPException(status_code=404, detail="Transkrypcja nie została znaleziona lub nie należy do użytkownika.")

    meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_id)
    qdrant_collection_name = None
    if meeting_data:
        qdrant_collection_name = meeting_data.get("qdrant_collection_name")

    if qdrant_collection_name:
        try:
            await initialize_qdrant_resources()
            collection_removed = await delete_collection_if_exists(qdrant_collection_name)
            if not collection_removed:
                logging.info(
                    f"Qdrant collection '{qdrant_collection_name}' was not present when attempting to delete for file_id={file_id}."
                )
        except Exception as exc:
            logging.error(
                f"Error deleting Qdrant collection '{qdrant_collection_name}' for file_id={file_id}: {exc}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Nie udało się usunąć kolekcji Qdrant powiązanej z transkrypcją.",
            )

    deleted = await run_in_threadpool(agent_db.delete_file_by_id_for_user, file_id, username)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transkrypcja nie została znaleziona lub nie należy do użytkownika.")
    return {"message": "Transkrypcja została usunięta."}


@app.post("/user/transcripts/{file_id}/regenerate_minutes")
async def regenerate_minutes_for_user_endpoint(
    file_id: int,
    payload: RegenerateMinutesPayload = Body(default=RegenerateMinutesPayload()),
    current_user: Dict[str, str] = Depends(get_current_user),
):
    username = current_user["username"]
    file_record = await run_in_threadpool(agent_db.get_file_record_by_id, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku lub brak dostępu.")
    owner_username = file_record.get("owner_username")
    if owner_username is None:
        await run_in_threadpool(agent_db.claim_file_for_user, file_id, username)
    elif owner_username != username:
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku lub brak dostępu.")

    meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_id)
    if not meeting_data:
        raise HTTPException(status_code=400, detail="Brak skonfigurowanych danych spotkania dla tego pliku.")

    collection_name = meeting_data.get("qdrant_collection_name")
    if not collection_name:
        raise HTTPException(status_code=400, detail="Brak kolekcji Qdrant dla tego pliku.")

    meeting_date = meeting_data.get("meeting_date")
    existing_minutes_id = meeting_data.get("minutes_id")
    transcript_id = meeting_data.get("transcript_id")

    if transcript_id is None:
        transcript_record = await run_in_threadpool(agent_db.get_transcript_by_file_id, file_id)
        if transcript_record:
            transcript_id = transcript_record.get("id")
        else:
            raise HTTPException(status_code=400, detail="Brak transkrypcji powiązanej z tym plikiem.")

    original_filepath = file_record.get("filepath")
    original_path_obj: Optional[Path] = None
    if original_filepath:
        try:
            candidate = Path(original_filepath)
            if candidate.exists():
                original_path_obj = candidate
        except Exception:
            original_path_obj = None

    existing_minutes_metadata: Dict[str, str] = {}
    if existing_minutes_id:
        minutes_record = await run_in_threadpool(agent_db.get_meeting_minutes_by_id, existing_minutes_id)
        if minutes_record:
            transcription_model_value = minutes_record.get("transcription_model")
            if transcription_model_value:
                existing_minutes_metadata["transcription_model_display"] = transcription_model_value
            diarization_model_value = minutes_record.get("diarization_model")
            if diarization_model_value:
                existing_minutes_metadata["diarization_model"] = diarization_model_value

    try:
        provider_tokens = _sanitize_provider_token_mapping(payload.provider_tokens) if payload.provider_tokens else {}
    except ValueError as parse_error:
        raise HTTPException(status_code=400, detail=f"Invalid provider_tokens payload: {parse_error}")

    minutes_response, minutes_filename, minutes_db_id = await generate_and_save_minutes(
        collection_name=collection_name,
        original_source_path=original_path_obj,
        custom_prompt=payload.custom_prompt,
        output_name=payload.output_name,
        progress_callback=None,
        file_id=file_id,
        target_date_override=meeting_date,
        reuse_minutes_id=existing_minutes_id,
        gemini_api_key=provider_tokens.get('gemini'),
        metadata=existing_minutes_metadata,
    )

    if minutes_response is None:
        detail_message = minutes_filename if isinstance(minutes_filename, str) else "Nie udało się wygenerować podsumowania."
        raise HTTPException(status_code=500, detail=detail_message)

    final_minutes_id = minutes_db_id or existing_minutes_id
    if final_minutes_id:
        await run_in_threadpool(agent_db.update_meeting_data_minutes_id, file_id, final_minutes_id)

    return JSONResponse(
        content={
            "message": "Podsumowanie zostało wygenerowane ponownie.",
            "minutes_id": final_minutes_id,
            "minutes_path": minutes_filename,
        }
    )

@app.get("/list_processed_files/", dependencies=[Depends(get_current_user)])
async def list_processed_files_endpoint():
    processed_files_info = {}
    output_base_dir = Path(config.OUTPUT_DIR_API_TRANSCRIPTS)
    if not await asyncio.to_thread(output_base_dir.exists):
        return JSONResponse(content={})

    for job_id_dir in await asyncio.to_thread(output_base_dir.iterdir):
        if await asyncio.to_thread(job_id_dir.is_dir):
            transcription_files = await asyncio.to_thread(lambda p: list(p.glob("*_diarized_transcription.txt")), job_id_dir)
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

@app.get("/get_file_speakers/{file_job_id}", dependencies=[Depends(get_current_user)])
async def get_file_speakers_endpoint(file_job_id: str):
    transcription_file_path = Path(config.OUTPUT_DIR_API_TRANSCRIPTS) / file_job_id / f"*_diarized_transcription.txt"
    found_files = await asyncio.to_thread(lambda p: list(p.glob(f"{file_job_id}/*_diarized_transcription.txt")), Path(config.OUTPUT_DIR_API_TRANSCRIPTS))

    if not found_files:
        raise HTTPException(status_code=404, detail=f"Transcription file for job ID {file_job_id} not found.")

    file_path = found_files[0]

    unique_speakers = set()
    try:
        async with asyncio.to_thread(open, file_path, "r", encoding="utf-8") as f:
            for line in await asyncio.to_thread(f.readlines):
                if ":" in line:
                    speaker_label = line.split(":", 1)[0].strip()
                    if speaker_label.startswith("SPEAKER_"):
                        unique_speakers.add(speaker_label)
        return JSONResponse(content=sorted(list(unique_speakers)))
    except Exception as e:
        logging.error(f"Error reading transcription file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read speakers from file: {e}")

@app.get("/get_enrolled_speakers/")
async def get_enrolled_speakers_endpoint(
    current_user: Dict[str, str] = Depends(get_current_user),
):
    speakers = await list_enrolled_speakers(username=current_user["username"])
    return JSONResponse(content=speakers)

@app.exception_handler(404)
async def custom_404_handler(request, exc):
    logging.warning(f"404 Not Found: {request.url}")
    if not request.url.path.startswith("/static/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/processed_files/{file_job_id}/details", response_model=Dict[str, Any], dependencies=[Depends(get_current_user)])
async def get_processed_file_details_endpoint(file_job_id: str):
    try:
        # Pobierz rekord pliku
        file_record = await run_in_threadpool(agent_db.get_file_record_by_id, int(file_job_id))
        if not file_record:
            raise HTTPException(status_code=404, detail=f"File with job ID {file_job_id} not found.")

        processed_at = file_record.get('processed_at')
        details = {
            "file_id": file_record['id'],
            "filename": file_record['filename'],
            "filepath": file_record['filepath'],
            "filehash": file_record['filehash'],
            "status": file_record['status'],
            "api_response": file_record['api_response'],
            "processed_at": processed_at.isoformat() if processed_at and hasattr(processed_at, "isoformat") else processed_at,
        }

        # Pobierz dane spotkania (jeśli istnieją)
        meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_record['id'])
        if meeting_data:
            processed_meeting_data = dict(meeting_data)
            created_at_meeting = processed_meeting_data.get("created_at")
            if created_at_meeting and hasattr(created_at_meeting, "isoformat"):
                processed_meeting_data["created_at"] = created_at_meeting.isoformat()
            details["meeting_data"] = processed_meeting_data
            # Pobierz transkrypcję (jeśli istnieje)
            if meeting_data.get('transcript_id'):
                transcript_record = await run_in_threadpool(agent_db.get_transcript_by_id, meeting_data['transcript_id'])
                if transcript_record:
                    transcript_created_at = transcript_record.get('created_at')
                    details["transcript"] = {
                        "id": transcript_record['id'],
                        "content_path": transcript_record['transcript_path'],
                        "created_at": transcript_created_at.isoformat() if transcript_created_at and hasattr(transcript_created_at, "isoformat") else transcript_created_at,
                    }
            
            # Pobierz protokół (jeśli istnieje)
            if meeting_data.get('minutes_id'):
                minutes_record = await run_in_threadpool(agent_db.get_meeting_minutes_by_id, meeting_data['minutes_id'])
                if minutes_record:
                    minutes_generated_at = minutes_record.get('generated_at')
                    minutes_created_at = minutes_record.get('created_at')
                    details["minutes"] = {
                        "id": minutes_record['id'],
                        "content_path": minutes_record['minutes_path'],
                        "generated_at": minutes_generated_at,
                        "created_at": minutes_created_at.isoformat() if minutes_created_at and hasattr(minutes_created_at, "isoformat") else minutes_created_at,
                    }

        return JSONResponse(content=details)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file_job_id. Must be an integer.")
    except Exception as e:
        logging.error(f"Error retrieving details for file job {file_job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file details: {str(e)}")

@app.get("/processed_files/{file_job_id}/transcript", response_class=HTMLResponse)
async def get_processed_file_transcript_endpoint(
    file_job_id: str,
    request: Request,
    token: Optional[str] = Query(None, alias="token"),
):
    try:
        await _resolve_request_user(request, token)
        # Pobierz rekord pliku
        file_record = await run_in_threadpool(agent_db.get_file_record_by_id, int(file_job_id))
        if not file_record:
            raise HTTPException(status_code=404, detail=f"File with job ID {file_job_id} not found.")

        # Pobierz dane spotkania, aby uzyskać transcript_id
        meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_record['id'])
        if not meeting_data or not meeting_data.get('transcript_id'):
            raise HTTPException(status_code=404, detail=f"No transcript found for file job ID {file_job_id}.")
        
        transcript_record = await run_in_threadpool(agent_db.get_transcript_by_id, meeting_data['transcript_id'])
        if not transcript_record or not transcript_record.get('transcript_path'):
            raise HTTPException(status_code=404, detail=f"Transcript content not found for file job ID {file_job_id}.")

        transcript_path = Path(transcript_record['transcript_path'])
        if not await asyncio.to_thread(transcript_path.exists):
            raise HTTPException(status_code=404, detail=f"Transcript file not found on disk: {transcript_path}")

        content = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
        return HTMLResponse(content=f"<pre>{content}</pre>")

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file_job_id. Must be an integer.")
    except Exception as e:
        logging.error(f"Error retrieving transcript for file job {file_job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transcript: {str(e)}")

@app.get("/processed_files/{file_job_id}/minutes", response_class=HTMLResponse)
async def get_processed_file_minutes_endpoint(
    file_job_id: str,
    request: Request,
    format: str = "text",
    token: Optional[str] = Query(None, alias="token"),
    minutes_id: Optional[int] = Query(None, alias="minutes_id"),
):
    try:
        current_user = await _resolve_request_user(request, token)
        username = current_user["username"]

        file_record = await run_in_threadpool(agent_db.get_file_record_by_id, int(file_job_id))
        if not file_record:
            raise HTTPException(status_code=404, detail=f"File with job ID {file_job_id} not found.")

        owner_username = file_record.get("owner_username")
        if owner_username is None:
            await run_in_threadpool(agent_db.claim_file_for_user, file_record["id"], username)
        elif owner_username != username:
            raise HTTPException(status_code=404, detail="File not found or not accessible.")

        selected_minutes_id: Optional[int] = minutes_id
        if selected_minutes_id is None:
            meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_record["id"])
            if not meeting_data or not meeting_data.get("minutes_id"):
                raise HTTPException(status_code=404, detail=f"No minutes found for file job ID {file_job_id}.")
            selected_minutes_id = meeting_data["minutes_id"]

        minutes_record = await run_in_threadpool(agent_db.get_meeting_minutes_by_id, selected_minutes_id)
        if (
            not minutes_record
            or not minutes_record.get("minutes_path")
            or minutes_record.get("file_id") != file_record["id"]
        ):
            raise HTTPException(status_code=404, detail=f"Minutes content not found for file job ID {file_job_id}.")

        minutes_path = Path(minutes_record["minutes_path"])
        if not await asyncio.to_thread(minutes_path.exists):
            raise HTTPException(status_code=404, detail=f"Minutes file not found on disk: {minutes_path}")

        content = await asyncio.to_thread(minutes_path.read_text, encoding="utf-8")

        if format == "html":
            from summarizer import generate_html_from_text
            html_content = await generate_html_from_text(content)
            return HTMLResponse(content=html_content)
        return HTMLResponse(content=f"<pre>{content}</pre>")

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file_job_id. Must be an integer.")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving minutes for file job {file_job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve minutes: {str(e)}")

# ==================================================
#         CHAT ENDPOINTS Z KOLEJKĄ
# ==================================================

@app.post("/chat/query", response_model=ChatResponse)
async def chat_query_endpoint(chat_query: ChatQuery, request: Request):
    """
    Endpoint do wysyłania zapytań do Agenta Czatowego z kolejką wiadomości.
    """
    logging.info(f"Received chat query: '{chat_query.query}' at /chat/query endpoint.")
    
    try:
        # Pobierz użytkownika z tokena (jeśli dostępny) lub użyj username z requestu
        username = "anonymous"
        try:
            # Spróbuj pobrać token z nagłówka Authorization
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                current_user = await get_current_user(token)
                username = current_user.get("username", "anonymous")
        except:
            # Jeśli nie ma autoryzacji, użyj username z requestu lub anonymous
            username = chat_query.username or "anonymous"
        
        logging.info(
            "Chat request for user: %s (collection=%s)",
            username,
            chat_query.collection_name or "<none>",
        )
        
        # Dodaj wiadomość do kolejki
        try:
            job_id = await enqueue_message(
                username=username,
                query=chat_query.query,
                collection_name=chat_query.collection_name,
                priority=0,
            )
            
            if not job_id:
                raise ValueError("enqueue_message zwróciło None - job_id nie został wygenerowany")
            
            logging.info(f"Wiadomość dodana do kolejki: {job_id}")
            
            # Zwróć odpowiedź informującą, że wiadomość została dodana do kolejki
            return ChatResponse(
                response=f"Twoja wiadomość została przyjęta i jest przetwarzana. ID zadania: {job_id}. "
                        f"Możesz sprawdzić status używając endpointu /chat/query/status/{job_id}"
            )
        except RuntimeError as e:
            # Błąd związany z konfiguracją Dramatiq
            error_msg = str(e)
            logging.error(f"Błąd konfiguracji Dramatiq: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail=f"System kolejkowania nie jest dostępny: {error_msg}. Upewnij się, że Dramatiq worker jest uruchomiony."
            )
        except Exception as e:
            # Inne błędy
            error_msg = str(e)
            logging.error(f"Błąd podczas dodawania wiadomości do kolejki: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Wystąpił błąd podczas dodawania wiadomości do kolejki: {error_msg}"
            )
        
    except Exception as e:
        logging.error(f"Error in chat_query_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Wystąpił błąd podczas dodawania wiadomości do kolejki: {e}"
        )


@app.get("/chat/query/status/{job_id}")
async def get_message_status(job_id: str, request: Request):
    """
    Sprawdza status wiadomości w kolejce.
    
    Zwraca:
        - status: 'pending', 'processing', 'completed', 'failed', 'retrying'
        - response: Odpowiedź (jeśli status = 'completed')
        - error_message: Komunikat błędu (jeśli status = 'failed')
    """
    try:
        job_status = await get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": job_status.get("status"),
            "response": job_status.get("response"),
            "error_message": job_status.get("error_message"),
            "retry_count": job_status.get("retry_count", 0),
            "max_retries": job_status.get("max_retries", 3),
            "created_at": job_status.get("created_at"),
            "processed_at": job_status.get("processed_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting job status: {e}")


@app.get("/chat/query/response/{job_id}")
async def get_message_response(job_id: str, request: Request):
    """
    Pobiera odpowiedź na wiadomość (jeśli jest gotowa).
    
    Zwraca ChatResponse jeśli zadanie jest zakończone, w przeciwnym razie błąd.
    """
    try:
        job_status = await get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        status = job_status.get("status")
        
        if status == "completed":
            return ChatResponse(response=job_status.get("response", ""))
        elif status == "failed":
            error_msg = job_status.get("error_message", "Unknown error")
            return ChatResponse(
                response=f"Przepraszam, wystąpił błąd podczas przetwarzania Twojego zapytania: {error_msg}"
            )
        elif status in ("pending", "processing", "retrying"):
            return ChatResponse(
                response=f"Wiadomość jest nadal przetwarzana. Status: {status}. "
                        f"Próba {job_status.get('retry_count', 0) + 1}/{job_status.get('max_retries', 3)}"
            )
        else:
            return ChatResponse(response=f"Nieznany status zadania: {status}")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting job response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting job response: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Zatrzymuje system kolejkowania przy zamknięciu aplikacji."""
    try:
        await stop_boss()
        logging.info("✅ System kolejkowania zatrzymany podczas shutdown.")
    except Exception as e:
        logging.error(f"Błąd podczas zatrzymywania systemu kolejkowania: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}


@app.get("/queue/stats")
async def get_queue_statistics(request: Request):
    """
    Zwraca statystyki kolejki Dramatiq.
    
    Zwraca:
        - pending: liczba zadań oczekujących
        - completed: liczba zadań zakończonych
        - failed: liczba zadań zakończonych błędem
        - total: całkowita liczba zadań
        - queue_size: rozmiar kolejki
    """
    try:
        stats = await get_queue_stats()
        return stats
    except Exception as e:
        logging.error(f"Error getting queue stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting queue stats: {e}")


@app.get("/queue/jobs")
async def get_queue_jobs(request: Request, limit: int = Query(50, ge=1, le=200)):
    """
    Zwraca listę aktywnych zadań w kolejce.
    
    Args:
        limit: Maksymalna liczba zadań do zwrócenia (1-200)
    
    Returns:
        Lista zadań z ich statusami
    """
    try:
        jobs = await list_active_jobs(limit=limit)
        return {
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
        }
    except Exception as e:
        logging.error(f"Error getting queue jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting queue jobs: {e}")

@app.get("/processed_batches/{batch_job_id}/details", response_model=Dict[str, Any], dependencies=[Depends(get_current_user)])
async def get_processed_batch_details_endpoint(batch_job_id: str):
    logging.info(f"Received request for batch details for batch_job_id: {batch_job_id}")
    
    batch_info = job_temp_storage.get(batch_job_id)

    if not batch_info:
        logging.info(f"[{batch_job_id}] Batch info not found in job_temp_storage. Attempting to retrieve from DB.")
        db_batch_record = await run_in_threadpool(agent_db.get_batch_job, batch_job_id)

        if not db_batch_record:
            logging.error(f"[{batch_job_id}] Batch job with ID {batch_job_id} not found in DB. Returning 404.")
            raise HTTPException(status_code=404, detail=f"Batch job with ID {batch_job_id} not found.")

        # Zrekonstruuj batch_info z bazy danych
        file_job_ids_from_db = json.loads(db_batch_record['file_job_ids_json'])
        params_from_db = json.loads(db_batch_record['params_json'])
        
        batch_info = {
            'type': 'batch',
            'file_job_ids': file_job_ids_from_db,
            'status': db_batch_record['status'],
            'params': params_from_db
        }
        job_temp_storage[batch_job_id] = batch_info # Przywróć do job_temp_storage
        logging.info(f"[{batch_job_id}] Batch info reconstructed from DB and added to job_temp_storage.")

    if batch_info.get('type') != 'batch': # Dodatkowa weryfikacja typu, chociaż get_batch_job już to filtruje
        logging.error(f"[{batch_job_id}] Retrieved record is not of type 'batch'.")
        raise HTTPException(status_code=404, detail=f"Batch job with ID {batch_job_id} not found or is not a batch type.")
    
    file_job_ids = batch_info.get('file_job_ids', [])
    processed_files_details = []

    for file_job_id in file_job_ids:
        file_details = {"file_job_id": file_job_id}
        try:
            file_info = job_temp_storage.get(file_job_id)
            if not file_info or 'db_file_id' not in file_info:
                # Jeśli file_info nie ma w job_temp_storage (np. po restarcie), próbuj pobrać z DB
                logging.warning(f"[{batch_job_id}] File info for {file_job_id} not in job_temp_storage. Attempting to retrieve from DB.")
                db_file_record = await run_in_threadpool(agent_db.get_file_record_by_filehash, file_job_id) # file_job_id jest hashem
                if db_file_record:
                    file_id = db_file_record['id']
                    file_info = {
                        'type': 'file',
                        'batch_job_id': batch_job_id,
                        'status': db_file_record['status'],
                        'db_file_id': file_id,
                        'file_path': db_file_record['filepath'],
                        'params': batch_info.get('params', {})
                    }
                    if db_file_record.get('filename'):
                        file_info['filename'] = db_file_record['filename']
                    job_temp_storage[file_job_id] = file_info # Przywróć do job_temp_storage
                    logging.info(f"[{batch_job_id}] File info for {file_job_id} reconstructed from DB and added to job_temp_storage.")
                else:
                    logging.warning(f"[{batch_job_id}] File record for {file_job_id} not found in database. Skipping.")
                    file_details["status"] = "not_found"
                    file_details["error_message"] = "File record not found in database."
                    processed_files_details.append(file_details)
                    continue

            file_id = file_info.get('db_file_id') 
            if file_id is None:
                raise ValueError(f"db_file_id not found in job_temp_storage for file_job_id: {file_job_id}. This should not happen after reconstruction.")
            
            file_record = await run_in_threadpool(agent_db.get_file_record_by_id, int(file_id))
            
            if file_record:
                file_details["filename"] = file_record['filename']
                file_details["status"] = file_record['status']
                file_details["file_id"] = file_record['id']
                file_details["error_message"] = file_record['api_response'] # Dodaj error_message z api_response

                meeting_data = await run_in_threadpool(agent_db.get_meeting_data_by_file_id, file_record['id'])
                if meeting_data:
                    if meeting_data.get('transcript_id'):
                        transcript_record = await run_in_threadpool(agent_db.get_transcript_by_id, meeting_data['transcript_id'])
                        if transcript_record and transcript_record.get('transcript_path'):
                            file_details["transcription_url"] = f"/processed_files/{file_record['id']}/transcript"
                            file_details["transcription_download_url"] = f"/{transcript_record['transcript_path']}"

                    if meeting_data.get('minutes_id'):
                        minutes_record = await run_in_threadpool(agent_db.get_meeting_minutes_by_id, meeting_data['minutes_id'])
                        if minutes_record and minutes_record.get('minutes_path'):
                            file_details["minutes_html_url"] = f"/processed_files/{file_record['id']}/minutes?format=html"
                            file_details["minutes_download_url"] = f"/{minutes_record['minutes_path']}"

            else:
                file_details["status"] = "not_found"
                file_details["error_message"] = "File record not found in database."

        except Exception as e:
            logging.error(f"Error fetching details for file_job_id {file_job_id}: {e}", exc_info=True)
            file_details["status"] = "error"
            file_details["error_message"] = str(e)
        
        processed_files_details.append(file_details)

    overall_batch_status = "completed"
    if batch_info.get('status') == "error": # Priorytet dla statusu error z bazy danych
        overall_batch_status = "error"
    elif batch_info.get('status') == "processing" or batch_info.get('status') == "uploaded":
        overall_batch_status = "processing"
    else: # Jeśli status z bazy danych nie jest błędem ani w toku, sprawdź pliki
        for file_detail in processed_files_details:
            if file_detail["status"] == "error":
                overall_batch_status = "error"
                break
            elif file_detail["status"] == "processing" or file_detail["status"] == "uploaded" or file_detail["status"] == "transcribed" or file_detail["status"] == "ingesting" or file_detail["status"] == "indexed" or file_detail["status"] == "summarizing" or file_detail["status"] == "finalizing": # Dodano nowe statusy pośrednie
                overall_batch_status = "processing"
                break

    return {
        "batch_job_id": batch_job_id,
        "total_files": len(file_job_ids),
        "files": processed_files_details,
        "batch_status": overall_batch_status
    }
