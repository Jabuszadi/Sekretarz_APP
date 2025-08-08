# chunker.py
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import timedelta, datetime
from langchain.schema import Document

# === Helper Functions ===

def seconds_to_srt(seconds: float) -> str:
    """Converts seconds (float) to SRT time format HH:MM:SS,ms."""
    if seconds < 0: seconds = 0 # Handle potential negative start times
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def datetime_to_str(dt) -> str:
    """Converts datetime object to string format YYYY-MM-DD."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d") # Zmieniono format na YYYY-MM-DD
    return str(dt) # Return as is if not datetime

# === Chunking Logic ===

def chunk_by_duration(df: pd.DataFrame, duration: float, overlap: float = 0) -> List[Document]:
    """
    Chunks DataFrame by time duration with overlap and returns a list of Langchain Document objects.
    """
    all_documents = [] # Lista do zbierania obiektów Document
    if df.empty:
        logging.warning("⚠️ Input DataFrame for duration chunking is empty.")
        return all_documents

    grouped = df.groupby("file_path")
    logging.info(f"⏳ Chunking {len(grouped)} files by duration={duration}s, overlap={overlap}s...")

    for file_path, group in grouped:
        group = group.sort_values("from_time").reset_index(drop=True)
        max_time = group["to_time"].max()
        start_time = group["from_time"].min() # Start from the first segment's time
        chunks = [] # Tymczasowa lista do przechowywania słowników chunków
        chunk_id = 1
        current_time = start_time

        while current_time < max_time:
            end_time = current_time + duration
            # Select lines that overlap with the [current_time, end_time) window
            window_df = group[(group["from_time"] < end_time) & (group["to_time"] > current_time)]

            if not window_df.empty:
                chunk_content = " ".join(window_df["content"])
                chunk_word_count = len(chunk_content.split())
                chunk_speakers = ", ".join(window_df["speaker"].unique())
                actual_start = window_df["from_time"].min()
                actual_end = window_df["to_time"].max()

                # Utwórz słownik chunka
                chunk_dict = {
                    "chunk_id": chunk_id,
                    "from_time": actual_start,
                    "to_time": actual_end,
                    "from_time_srt": seconds_to_srt(actual_start),
                    "to_time_srt": seconds_to_srt(actual_end),
                    "word_count": chunk_word_count,
                    "content": chunk_content,
                    "speaker": chunk_speakers,
                    "file_path": Path(file_path).name, # Store only filename
                    "date": datetime_to_str(window_df["date"].iloc[0]), # Ensure date is string for metadata
                    "chunk_type": f"duration_{int(duration)}s_overlap_{int(overlap)}s"
                }
                logging.debug(f"Chunk dict date after datetime_to_str: {chunk_dict['date']}") # NOWY LOG
                chunks.append(chunk_dict)
                chunk_id += 1

            # Move the window forward
            current_time += duration - overlap
            # Ensure we don't get stuck if duration <= overlap
            if duration <= overlap and not window_df.empty:
                 # Move past the current window if overlap is too large
                 current_time = window_df['to_time'].max() if current_time <= window_df['to_time'].max() else current_time + 1e-3

        # Dodaj pre/post chunk IDs i konwertuj na obiekty Langchain Document
        for i, chunk in enumerate(chunks):
            chunk["prechunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunk["postchunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None

            # Utwórz obiekt Langchain Document
            # `page_content` to główna treść chunka
            # `metadata` zawiera wszystkie inne istotne informacje
            document = Document(
                page_content=chunk["content"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "from_time": chunk["from_time"],
                    "to_time": chunk["to_time"],
                    "from_time_srt": chunk["from_time_srt"],
                    "to_time_srt": chunk["to_time_srt"],
                    "word_count": chunk["word_count"],
                    "speaker": chunk["speaker"],
                    "file_path": chunk["file_path"],
                    "date": chunk["date"], # Powinno być już stringiem
                    "chunk_type": chunk["chunk_type"],
                    "prechunk_id": chunk["prechunk_id"],
                    "postchunk_id": chunk["postchunk_id"]
                }
            )
            all_documents.append(document)

        logging.debug(f"  Generated {len(chunks)} duration chunks (and {len(all_documents)} documents) for {Path(file_path).name}")

    logging.info(f"✅ Finished duration chunking, returning {len(all_documents)} documents.")
    return all_documents # Zwróć listę obiektów Document

def chunk_by_lines(df: pd.DataFrame, lines_per_chunk: int, overlap: int = 0) -> Dict[str, List[Dict]]:
    """Chunks DataFrame by number of lines with overlap."""
    chunks_by_file = {}
    if df.empty:
        logging.warning("⚠️ Input DataFrame for line chunking is empty.")
        return chunks_by_file

    grouped = df.groupby("file_path")
    logging.info(f"⏳ Chunking {len(grouped)} files by lines={lines_per_chunk}, overlap={overlap} lines...")

    for file_path, group in grouped:
        group = group.sort_values("from_time").reset_index(drop=True) # Ensure order and 0-based index
        num_lines = len(group)
        chunks = []
        i = 0
        chunk_id = 1

        while i < num_lines:
            start_index = i
            end_index = i + lines_per_chunk
            chunk_df = group.iloc[start_index:end_index]

            if not chunk_df.empty:
                content = " ".join(chunk_df["content"])
                word_count = len(content.split())
                chunks.append({
                    "chunk_id": chunk_id,
                    "from_time": chunk_df["from_time"].min(),
                    "to_time": chunk_df["to_time"].max(),
                    "from_time_srt": seconds_to_srt(chunk_df["from_time"].min()),
                    "to_time_srt": seconds_to_srt(chunk_df["to_time"].max()),
                    "word_count": word_count,
                    "content": content,
                    "speaker": ", ".join(chunk_df["speaker"].unique()),
                    "file_path": Path(file_path).name, # Store only filename
                    "date": chunk_df["date"].iloc[0], # Assumes date is consistent
                    "chunk_type": f"lines_{lines_per_chunk}_overlap_{overlap}"
                })
                chunk_id += 1

            # Move to the next chunk start index
            step = lines_per_chunk - overlap
            i += step if step > 0 else 1 # Ensure progress even with bad overlap settings

        # Add pre/post chunk IDs
        for i, chunk in enumerate(chunks):
            chunk["prechunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunk["postchunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None

        chunks_by_file[file_path] = chunks
        logging.debug(f"  Generated {len(chunks)} line chunks for {Path(file_path).name}")

    logging.info(f"✅ Finished line chunking.")
    return chunks_by_file