# data_handler.py

import json
import logging
from pathlib import Path
import asyncio # Import asyncio
from typing import List, Dict, Optional
from langchain_core.documents import Document # Zmieniono import
from datetime import datetime
import sqlite3
import pandas as pd

# Import helper from chunker
from chunker import datetime_to_str


async def save_chunks_by_variant(chunks_by_file: Dict[str, List[Dict]], variant_name: str, output_base_dir: str):
    """Saves generated chunks to JSON files, organized by variant."""
    output_path = Path(output_base_dir) / variant_name
    await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)
    logging.info(f"💾 Saving chunks for variant '{variant_name}' to {output_path}...")

    files_saved = 0
    total_chunks_saved = 0
    if not chunks_by_file:
        logging.warning(f"⚠️ No chunks provided for variant '{variant_name}'. Nothing to save.")
        return

    for original_file_path, chunks in chunks_by_file.items():
        if not chunks: # Skip if a file somehow resulted in zero chunks
            continue

        # Create filename based on the original SRT file stem
        file_stem = Path(original_file_path).stem
        output_file = output_path / f"{file_stem}.json"

        # Convert datetime objects to strings before saving
        serializable_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy() # Avoid modifying the original dict
            chunk_copy['date'] = datetime_to_str(chunk_copy['date'])
            # Convert float times directly to string or keep as float? Keeping as float for JSON.
            # If they need to be string:
            # chunk_copy['from_time'] = str(chunk_copy['from_time'])
            # chunk_copy['to_time'] = str(chunk_copy['to_time'])
            serializable_chunks.append(chunk_copy)


        try:
            f = None # Inicjalizacja uchwytu pliku
            try:
                f = await asyncio.to_thread(open, output_file, "w", encoding="utf-8")
                await asyncio.to_thread(json.dump, serializable_chunks, f, ensure_ascii=False, indent=2)
            finally:
                if f:
                    await asyncio.to_thread(f.close)
            files_saved += 1
            total_chunks_saved += len(chunks)
        except IOError as e:
            logging.error(f"❌ Failed to save chunks for {original_file_path} to {output_file}: {e}")
        except TypeError as e:
            logging.error(f"❌ Failed to serialize chunks for {original_file_path} to JSON: {e}. Check data types.")


    logging.info(f"✅ Saved {total_chunks_saved} chunks across {files_saved} files for variant '{variant_name}'.")


async def load_chunks_as_documents(chunk_variant_folder: str) -> List[Document]:
    """Loads all JSON chunk files from a specific variant folder into LangChain Documents."""
    folder = Path(chunk_variant_folder)
    if not await asyncio.to_thread(folder.is_dir): # Asynchroniczne sprawdzenie katalogu
        logging.error(f"❌ Chunk variant folder not found: {chunk_variant_folder}")
        return []

    # --- FIX: Get variant name from the folder path ---
    variant_name_from_folder = folder.name

    json_files = await asyncio.to_thread(lambda p: list(p.glob("*.json")), folder) # Asynchroniczne listowanie plików
    if not json_files:
        logging.warning(f"⚠️ No JSON chunk files found in: {chunk_variant_folder}")
        return []

    logging.info(f"📄 Loading chunks from {len(json_files)} files in {chunk_variant_folder} (variant: {variant_name_from_folder})...")
    all_documents = []
    total_chunks_loaded = 0

    for file_path in json_files:
        try:
            f = None # Inicjalizacja uchwytu pliku
            try:
                f = await asyncio.to_thread(open, file_path, encoding="utf-8")
                chunks_in_file = await asyncio.to_thread(json.load, f)
            finally:
                if f:
                    await asyncio.to_thread(f.close)

            for chunk_data in chunks_in_file:
                # Ensure essential fields exist, provide defaults if necessary
                page_content = chunk_data.get("content", "")
                metadata = {
                    "chunk_id": chunk_data.get("chunk_id"),
                    "file_path": chunk_data.get("file_path"), # Original filename from chunk data
                    "date": chunk_data.get("date"), # Date should be string 'DD-MM-YYYY' now
                    "speaker": chunk_data.get("speaker", "UNKNOWN"),
                    "from_time": chunk_data.get("from_time"), # Keep as number/string as loaded
                    "to_time": chunk_data.get("to_time"),
                    "from_time_srt": chunk_data.get("from_time_srt"),
                    "to_time_srt": chunk_data.get("to_time_srt"),
                    # --- FIX: Use variant_name_from_folder as fallback ---
                    "chunk_type": chunk_data.get("chunk_type", variant_name_from_folder),
                    "prechunk_id": chunk_data.get("prechunk_id"),
                    "postchunk_id": chunk_data.get("postchunk_id"),
                    "word_count": chunk_data.get("word_count", 0)
                }
                 # Filter out None values from metadata for cleaner storage
                metadata = {k: v for k, v in metadata.items() if v is not None}

                all_documents.append(Document(page_content=page_content, metadata=metadata))
                total_chunks_loaded += 1

        except json.JSONDecodeError as e:
            logging.error(f"❌ Error decoding JSON from file {file_path}: {e}")
        except Exception as e:
            logging.error(f"❌ Unexpected error processing file {file_path}: {e}")

    logging.info(f"✅ Loaded {total_chunks_loaded} chunks into {len(all_documents)} LangChain Documents from variant '{variant_name_from_folder}'.")
    return all_documents