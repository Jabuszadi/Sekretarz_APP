import logging
import os
import sys
from pathlib import Path
# Dodaj katalog nadrzędny do ścieżki systemowej, aby umożliwić import 'config'
sys.path.append(str(Path(__file__).parent.parent))

import config
import pathlib
import uuid
import shutil
from typing import Dict, Any, List, Optional, Tuple

from fastmcp import FastMCP
from fastmcp.client import Client as MCPClient # Import Client do uzycia wewnetrznego

import agent_db # Zmieniono import na cały moduł
from minutes_service import generate_and_save_minutes
from models import TranscriptionSegment
from processor import process_audio
from ingestion_service import ingest_transcription
import asyncio # Importujemy asyncio
import qdrant_handler # Zmieniono z "from qdrant_handler import initialize_qdrant_resources"
import utils # DODANO: Importujemy utils, bo tam jest load_enrolled_speakers


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicjalizacja FastMCP
mcp_audio_processing = FastMCP()

# Słownik do przechowywania statusu zadań przetwarzania
job_temp_storage: Dict[str, Dict[str, Any]] = {}

def cleanup_temp_dir(job_id: str):
    temp_dir = os.path.join(config.UPLOAD_DIR, job_id)
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

@mcp_audio_processing.tool("process_audio_and_summarize")
async def process_audio_and_summarize_tool(
    file_path: str,
    file_job_id: str,
    output_directory: str,
    whisper_model_size: str = config.WHISPER_MODEL_SIZE,
    output_name: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    chunk_duration: int = 240,
    chunk_overlap: int = 0,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    input_file_path = Path(file_path)

    try:
        # NOWOŚĆ: Leniwa inicjalizacja Qdrant i modelu osadzania
        if qdrant_handler.qdrant_client is None or qdrant_handler.embedding_model is None:
            logging.info(f"  [JOB {file_job_id}] Qdrant client or embedding model not initialized. Initializing now...")
            await qdrant_handler.initialize_qdrant_resources()
            logging.info(f"  [JOB {file_job_id}] Qdrant resources initialized successfully within tool execution.")

        logging.info(f"Starting audio processing for {input_file_path.name} (job_id: {file_job_id})...")
        # POPRAWKA: Poprawne odebranie trzech wartości zwracanych przez process_audio
        transcription_segments, transcription_filename, transcript_id = await process_audio(
            audio_path=input_file_path,
            file_job_id=file_job_id,
            whisper_model_size=whisper_model_size,
        )
        if not transcription_segments:
            raise Exception("Audio processing (transcription/diarization) failed.")
        logging.info(f"  [JOB {file_job_id}] Audio processing completed. Transcription saved to: {transcription_filename}")
        logging.info(f"  [JOB {file_job_id}] Transcription database ID: {transcript_id}") # Logujemy dla informacji

        formatted_segments = [
            TranscriptionSegment(
                start=seg[0].start,
                end=seg[0].end,
                speaker=seg[1],
                text=seg[2]
            )
            for seg in transcription_segments
        ]

        logging.info(f"  [JOB {file_job_id}] Starting transcription ingestion into Qdrant...")
        collection_name = await ingest_transcription(
            formatted_segments,
            input_file_path,
            file_job_id,
            transcript_id,
            chunk_duration, # Upewnij się, że te zmienne są dostępne w tym kontekście
            chunk_overlap   # i pochodzą z argumentów funkcji narzędzia
        )
        if not collection_name:
            raise Exception("Ingestion to Qdrant failed. No collection created.")
        logging.info(f"  [JOB {file_job_id}] Transcription ingested into Qdrant collection: {collection_name}")

        logging.info(f"  [JOB {file_job_id}] Generating meeting minutes summary with LLM (Gemini)....")
        # POPRAWKA: Poprawne odebranie trzech wartości zwracanych przez generate_and_save_minutes
        minutes_content_response, minutes_filename, minutes_db_id = await generate_and_save_minutes(
            collection_name,
            input_file_path,
            custom_prompt=custom_prompt,
            output_name=output_name
        )
        if minutes_content_response is None:
            raise Exception("Minutes generation failed.")
        logging.info(f"  [JOB {file_job_id}] Minutes generated and saved to: {minutes_filename}")
        logging.info(f"  [JOB {file_job_id}] Minutes database ID: {minutes_db_id}") # Logujemy ID bazy danych

        # DODANO: Zapisanie segmentów do bazy danych
        for seg in transcription_segments:
            # seg[0] = segment (zawiera start, end), seg[1] = speaker, seg[2] = text
            agent_db.add_transcript_segment(
                transcript_id,
                seg[0].start,
                seg[0].end,
                seg[1],
                seg[2]
            )


        return {
            "status": "success",
            "message": "File processed and summarized successfully.",
            "transcription": [s.model_dump() for s in formatted_segments] if formatted_segments else [],
            "unique_collection_name": collection_name,
            "minutes_filename": minutes_filename,
            "transcription_filename": transcription_filename
        }
    except Exception as e:
        logging.error(f"❌ [JOB {file_job_id}] Processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "file_job_id": file_job_id
        }
    finally:
        logging.info(f"  [JOB {file_job_id}] Cleaning up temporary directory.")
        cleanup_temp_dir(file_job_id)

# REWIZJA: Usunięto wywołanie asyncio.run(initialize_qdrant_resources())
# To wywołanie powodowało błąd "Event loop is closed"
if __name__ == "__main__":
    logging.info("Starting up Audio Processing FastMCP Server...")
    try:
        # Poniższa linia została usunięta, ponieważ powodowała konflikt pętli zdarzeń:
        # asyncio.run(initialize_qdrant_resources()) 

        # TERAZ: Inicjalizacja zasobów Qdrant nie będzie miała miejsca automatycznie
        # Musimy znaleźć inne rozwiązanie, aby wywołać initialize_qdrant_resources()
        # w kontekście pętli zdarzeń FastMCP lub w inny sposób.
        logging.info("Qdrant resources initialization skipped for now to avoid 'Event loop is closed' error.")

        # DODANO: Ładowanie profili mówców przy starcie agenta audio
        utils.load_enrolled_speakers() # Zmieniono na utils.load_enrolled_speakers()
        logging.info("✅ Enrolled speaker profiles loaded for Audio Agent.")

        mcp_audio_processing.run(transport="http", port=8003, host="0.0.0.0")
    except Exception as e:
        logging.error(f"Failed to start Audio Processing Agent: {e}")
