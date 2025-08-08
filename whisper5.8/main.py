# main.py
import logging
import os # Added os import
import re # Added re import
from pathlib import Path
from datetime import datetime

# Import configuration constants and validation function
from config import BASE_CHUNK_DIR, TRANSCRIPTS_DIR, validate_config # Added TRANSCRIPTS_DIR

# Import functions from specific modules
from qdrant_handler import search_documents
from summarizer import generate_minutes_of_meeting
from utils import save_minutes_to_file # MODIFIED

# --- Date Extraction Function ---
def extract_unique_dates_from_transcripts(transcripts_directory: str) -> list[str]:
    """
    Scans a directory for filenames matching the pattern YYYY-MM-DD*.
    Extracts unique dates in YYYY-MM-DD format.
    """
    unique_dates = set()
    transcript_path = Path(transcripts_directory)

    if not transcript_path.is_dir():
        logging.error(f"❌ Transcript directory '{transcripts_directory}' not found. Cannot extract dates.")
        return []

    # Regex to find filenames starting with YYYY-MM-DD
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}).*")

    logging.info(f"🔍 Scanning for transcript dates in: {transcript_path.resolve()}")
    found_files = 0
    for entry in transcript_path.iterdir():
        if entry.is_file():
            match = date_pattern.match(entry.name)
            if match:
                date_str = match.group(1)
                # Optional: Validate if the extracted string is a valid date
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    unique_dates.add(date_str)
                    found_files += 1
                except ValueError:
                    logging.warning(f"⚠️ Found pattern matching YYYY-MM-DD in '{entry.name}', but '{date_str}' is not a valid date. Skipping.")
            # else: # Optional: Log files that don't match
            #     logging.debug(f"  File '{entry.name}' does not match date pattern.")

    if not unique_dates:
        logging.warning(f"⚠️ No valid dates found in filenames within '{transcripts_directory}'.")
        return []

    sorted_dates = sorted(list(unique_dates))
    logging.info(f"✅ Found {len(sorted_dates)} unique date(s) from {found_files} matching files: {', '.join(sorted_dates)}")
    return sorted_dates


# --- Main Orchestration Function (per date) ---
def get_minutes_for_single_date(base_directory: str, target_date_iso: str):
    """
    Orchestrates the process for a SINGLE date (YYYY-MM-DD).
    Handles searching, generation, and saving for all collections for that date.
    """
    base_path = Path(base_directory)
    # Base directory existence check moved to main execution block

    # --- Date Conversion ---
    try:
        dt_object = datetime.strptime(target_date_iso, '%Y-%m-%d')
        target_date_qdrant_format = dt_object.strftime('%d-%m-%Y')
        logging.info(f"Processing date: {target_date_iso} (Qdrant search format: {target_date_qdrant_format})")
    except ValueError:
        # This shouldn't happen if extract_unique_dates_from_transcripts validates correctly
        logging.error(f"❌ Internal Error: Invalid date format '{target_date_iso}' received. Skipping this date.")
        return False

    logging.info(f"🚀 Starting meeting minutes generation for date: {target_date_iso}")
    logging.info(f"📂 Searching collections in: {base_path.resolve()}")

    collections_processed = 0
    collections_found = 0
    processing_successful = True

    for item in base_path.iterdir():
        if item.is_dir():
            collections_found += 1
            collection_name = item.name
            logging.info(f"\n🔍 Processing collection: {collection_name}")

            # 1. Search using the DD-MM-YYYY format
            docs = search_documents("meeting summary", collection_name, target_date_qdrant_format)

            if not docs:
                logging.info(f"  • No relevant documents found in Qdrant for date {target_date_qdrant_format} in collection '{collection_name}'. Skipping.")
                continue

            logging.info(f"  • Found {len(docs)} relevant document chunks.")
            context = "\n\n---\n\n".join(docs)

            # 2. Generate minutes (using summarizer)
            logging.info(f"  • Generating minutes using Gemini...")
            minutes = generate_minutes_of_meeting(context)
            if any("Error generating response." in v for v in minutes.values()) or \
               any("Error: Could not initialize LLM." in v for v in minutes.values()):
                logging.error(f"  • Failed to generate minutes for {collection_name} on {target_date_iso}. Skipping save.")
                processing_successful = False
                continue

            # 3. Save using the YYYY-MM-DD format for folder/filename
            save_minutes_to_file(minutes, collection_name, target_date_iso)
            collections_processed += 1

    if collections_found == 0:
         logging.warning(f"\n🏁 Finished date {target_date_iso}. No collection subdirectories found in {base_directory}.")
    elif collections_processed == 0:
        logging.info(f"\n🏁 Finished date {target_date_iso}. No collections with data matching Qdrant date {target_date_qdrant_format} were found or processed successfully in {base_directory}.")
    else:
        logging.info(f"\n🏁 Finished processing date {target_date_iso}. Generated minutes for {collections_processed} collection(s).")

    # Return True only if all collections for this date were processed without generation errors
    return processing_successful and (collections_processed > 0 or collections_found == 0)


# --- Execution ---
if __name__ == "__main__":
    # Validate configuration first
    try:
        validate_config() # Explicitly call validation
    except ValueError as e:
        logging.error(f"❌ Configuration Error: {e}")
        exit(1)
    except Exception as e:
        # Catch potential errors during client/model initialization if they are raised
        logging.error(f"❌ Initialization Error (Qdrant/Embeddings/Gemini likely): {e}")
        exit(1)

    # --- Get Target Dates Automatically ---
    target_dates_to_process = extract_unique_dates_from_transcripts(TRANSCRIPTS_DIR)

    # Exit if no dates were found or the transcript dir doesn't exist
    if not target_dates_to_process:
        logging.error(f"❌ No valid dates extracted from '{TRANSCRIPTS_DIR}'. Cannot proceed.")
        exit(1)

    # --- Ensure Base Chunk Directory Exists ---
    if not Path(BASE_CHUNK_DIR).is_dir():
        logging.error(f"❌ Base directory for chunks '{BASE_CHUNK_DIR}' does not exist. Cannot proceed.")
        exit(1)

    # --- Process Each Date ---
    total_dates_processed_successfully = 0
    total_dates_failed_or_empty = 0

    logging.info(f"Starting processing for {len(target_dates_to_process)} extracted date(s)...")

    for target_date in target_dates_to_process:
        # Pass BASE_CHUNK_DIR here now
        success = get_minutes_for_single_date(BASE_CHUNK_DIR, target_date)
        if success:
            total_dates_processed_successfully += 1
        else:
            total_dates_failed_or_empty += 1
        logging.info("-" * 50) # Separator between dates

    # --- Final Summary ---
    logging.info("=" * 50)
    logging.info("Overall Summary:")
    logging.info(f"  Dates processed with successful minutes generation: {total_dates_processed_successfully}")
    logging.info(f"  Dates with failures, no matching Qdrant data, or no collections found: {total_dates_failed_or_empty}")
    logging.info("=" * 50)

    import os
    import subprocess
    import logging

    # Konfiguracja logowania, aby widzieć DEBUG
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_ram_usage_gb() -> float:
        if os.name == 'nt':
            try:
                pid = os.getpid()
                command = f"wmic process where ProcessID={pid} get WorkingSetSize /value"
                result = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                
                output = result.stdout
                for line in output.splitlines():
                    if "WorkingSetSize=" in line:
                        working_set_size_bytes = int(line.split("=")[1])
                        ram_gb = working_set_size_bytes / (1024**3)
                        logging.info(f"DEBUG: Current RAM usage (WorkingSetSize) inside test: {ram_gb:.2f} GB")
                        return ram_gb
            except Exception as e:
                logging.error(f"Error getting RAM usage on Windows during test: {e}", exc_info=True)
                return 0.0
        else:
            logging.info("RAM usage tracking is currently supported only on Windows (test).")
            return 0.0

    if __name__ == "__main__":
        print("Running RAM usage test...")
        ram_usage = get_current_ram_usage_gb()
        if ram_usage > 0:
            print(f"RAM usage reported: {ram_usage:.2f} GB")
        else:
            print("RAM usage not reported (likely not on Windows or error occurred). Check logs above for details.")
