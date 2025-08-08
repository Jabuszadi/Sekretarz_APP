# ingest.py
import logging
from pathlib import Path
from typing import List
import re
from datetime import datetime
import pandas as pd
import agent_db # Dodaj ten import

# Import configuration variables and validation
from config import (
    TRANSCRIPTS_DIR, BASE_CHUNK_DIR,
    CHUNK_DURATION, CHUNK_OVERLAP,
    validate_config # Import the validation function
)

# Import processing functions from other modules
from srt_parser import parse_srt_files
from chunker import chunk_by_duration, chunk_by_lines
from data_handler import save_chunks_by_variant, load_chunks_as_documents
from qdrant_uploader import upload_documents_to_qdrant
from models import TranscriptionSegment # Assuming TranscriptionSegment is defined in models.py
from langchain.schema import Document # For Langchain Document objects
from utils import get_relevant_date_for_file # Dodaj import tej funkcji

# --- Main Ingestion Orchestration ---
def run_ingestion_pipeline():
    """Runs the full data ingestion pipeline: parse -> chunk -> save -> load -> upload."""
    logging.info("🚀 Starting Data Ingestion Pipeline...")

    # 1. Parse SRT Files
    logging.info("--- Step 1: Parsing SRT files ---")
    parsed_data_df = parse_srt_files(TRANSCRIPTS_DIR)
    if parsed_data_df.empty:
        logging.error("❌ Pipeline halted: Failed to parse SRT files or no data found.")
        return # Stop if parsing fails

    # 2. Chunking and Saving Loop
    logging.info("--- Step 2: Chunking and Saving Variants ---")
    all_variant_folders = []

    # Duration-based chunking
    duration_chunks = chunk_by_duration(parsed_data_df, duration=CHUNK_DURATION, overlap=CHUNK_OVERLAP)
    save_chunks_by_variant(duration_chunks, f"duration_{CHUNK_DURATION}s_overlap_{CHUNK_OVERLAP}s", BASE_CHUNK_DIR)
    all_variant_folders.append(Path(BASE_CHUNK_DIR) / f"duration_{CHUNK_DURATION}s_overlap_{CHUNK_OVERLAP}s")

    # Line-based chunking
    # If needed, CHUNK_LINE_SETTINGS needs to be re-introduced in config.py
    # for lines, overlap in CHUNK_LINE_SETTINGS:
    #      variant_name = f"lines_{lines}_overlap_{overlap}"
    #      logging.info(f"\n  Processing variant: {variant_name}")
    #      line_chunks = chunk_by_lines(parsed_data_df, lines_per_chunk=lines, overlap=overlap)
    #      save_chunks_by_variant(line_chunks, variant_name, BASE_CHUNK_DIR)
    #      all_variant_folders.append(Path(BASE_CHUNK_DIR) / variant_name)

    logging.info("✅ All chunking variants generated and saved.")

    # 3. Loading Chunks and Uploading to Qdrant
    logging.info("--- Step 3: Loading Chunks and Uploading to Qdrant ---")
    successful_uploads = 0
    failed_uploads = 0

    if not all_variant_folders:
         logging.warning("⚠️ No chunk variants were generated. Skipping Qdrant upload.")
    else:
        for variant_folder_path in all_variant_folders:
            if variant_folder_path.is_dir(): # Double check if folder exists
                collection_name = variant_folder_path.name # Use folder name as collection name
                logging.info(f"\n  Processing collection: {collection_name}")

                # Load documents from the variant folder
                documents_to_upload = load_chunks_as_documents(str(variant_folder_path))

                if documents_to_upload:
                    # Upload the loaded documents
                    success = upload_documents_to_qdrant(collection_name, documents_to_upload, "")
                    if success:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                else:
                    logging.warning(f"  No documents loaded from {variant_folder_path}. Skipping upload for collection '{collection_name}'.")
                    # Consider this a failure or just an empty variant? Let's count as failed upload for now.
                    failed_uploads +=1
            else:
                logging.warning(f"  Expected chunk folder '{variant_folder_path}' not found. Skipping.")
                failed_uploads += 1

    logging.info("--- Pipeline Summary ---")
    logging.info(f"  Qdrant Uploads Successful: {successful_uploads}")
    logging.info(f"  Qdrant Uploads Failed/Skipped: {failed_uploads}")
    logging.info("🏁 Ingestion Pipeline Finished.")

# --- New/Modified Ingestion Function for API (single file) ---
async def ingest_transcription(
    transcription_segments: List[TranscriptionSegment],
    original_source_path: Path,
    file_job_id: str,
    transcript_id: int, # Dodano transcript_id
    chunk_duration: int,
    chunk_overlap: int
) -> str: # Returns collection name
    """
    Ingests transcription segments for a single file, chunks them based on provided
    duration and overlap, and uploads them to Qdrant.
    """
    logging.info(f"Ingesting transcription for file job {file_job_id} into Qdrant...")

    # For API flow, we might want a unique collection name based on the file_job_id
    # or a combination of filename and chunking parameters.
    # Let's use a combination of filename, duration, overlap and a timestamp for uniqueness.
    # This ensures collections are unique per file and its chunking parameters.

    # Extract base name without extension
    base_name = original_source_path.stem
    # Sanitize base name for collection name
    sanitized_base_name = "".join([c if c.isalnum() else "_" for c in base_name])
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique collection name that reflects chunking parameters
    collection_name = f"{current_datetime_str}_{sanitized_base_name}"
    # Ensure collection name is not too long or has invalid characters for Qdrant
    collection_name = collection_name[:63] # Qdrant collection name max length (approx)
    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '', collection_name) # Alphanumeric, hyphens, underscores

    logging.info(f"Generated Qdrant collection name: {collection_name}")

    # Convert TranscriptionSegment list to a DataFrame for chunker functions
    # Assuming TranscriptionSegment is a tuple (SegmentTimeInfo, SpeakerID, Text)
    data_for_df = []
    # Pobierz datę dla pliku raz na początku
    file_date_str = get_relevant_date_for_file(original_source_path)

    for segment in transcription_segments:
        # Assuming segment[0] is an object with .start and .end attributes (SegmentTimeInfo)
        # segment[1] is SpeakerID (str), segment[2] is Text (str)
        data_for_df.append({
            'from_time': segment.start,
            'to_time': segment.end,
            'content': segment.text,
            'speaker': segment.speaker,
            'file_path': str(original_source_path),
            'date': file_date_str # Dodano kolumnę 'date'
        })
    transcription_df = pd.DataFrame(data_for_df)
    logging.info(f"DEBUG: Columns in transcription_df before chunking: {transcription_df.columns.tolist()}")

    # Chunk by duration using the provided parameters
    documents_to_upload = chunk_by_duration(transcription_df, duration=chunk_duration, overlap=chunk_overlap)
    
    if not documents_to_upload:
        logging.warning(f"No documents were generated after chunking for file job {file_job_id}.")
        # If no documents are generated, return an empty string or raise an error
        # Returning an empty string will indicate to api_app.py that no collection was created.
        return ""

    # NEW: Save chunks to transcript_chunks table
    chunk_count = 0
    for doc in documents_to_upload:
        chunk_id_val = chunk_count + 1 # Simple ID for now, could be improved
        from_time_val = doc.metadata.get('start_time', 0.0)
        to_time_val = doc.metadata.get('end_time', 0.0)
        speaker_val = doc.metadata.get('speaker', 'UNKNOWN')
        word_count_val = len(doc.page_content.split()) # Simple word count
        chunk_type_val = f"duration_{chunk_duration}s_overlap_{chunk_overlap}s"
        file_path_val = doc.metadata.get('file_path', str(original_source_path))
        date_val = doc.metadata.get('date', '')

        agent_db.add_transcript_chunk(
            transcript_id,
            chunk_id_val,
            from_time_val,
            to_time_val,
            speaker_val,
            word_count_val,
            chunk_type_val,
            file_path_val,
            date_val
        )
        chunk_count += 1
    logging.info(f"Added {chunk_count} chunks to transcript_chunks for transcript_id={transcript_id}")

    # Upload the loaded documents
    success = await upload_documents_to_qdrant(collection_name, documents_to_upload, file_job_id)

    if success:
        logging.info(f"Successfully uploaded {len(documents_to_upload)} chunks to Qdrant collection '{collection_name}'.")
        return collection_name
    else:
        logging.error(f"Failed to upload chunks to Qdrant collection '{collection_name}' for file job {file_job_id}.")
        return "" # Indicate failure with empty string or raise error

# --- Execution ---
if __name__ == "__main__":
    # Validate configuration first
    try:
        validate_config()
    except ValueError as e:
        # Error already logged by validate_config
        exit(1) # Exit if essential config is missing
    except Exception as e:
        logging.error(f"❌ Unexpected error during initial validation: {e}")
        exit(1)

    # Check required directories for THIS script
    if not Path(TRANSCRIPTS_DIR).is_dir():
         logging.error(f"❌ Transcript directory '{TRANSCRIPTS_DIR}' not found. Cannot start ingestion.")
         exit(1)
    # BASE_CHUNK_DIR will be created if it doesn't exist by save_chunks_by_variant

    # Run the main pipeline function
    run_ingestion_pipeline()