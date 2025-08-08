# agent_db.py
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

DB_PATH = "agent_files.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT,
                filehash TEXT,
                status TEXT,
                api_response TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                transcript_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meeting_minutes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_text TEXT,
                minutes_path TEXT,
                generated_at TEXT,
                llm_model TEXT,
                embeddings_model TEXT,
                chunking_method TEXT,
                transcription_model TEXT,
                diarization_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meeting_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                transcript_id INTEGER,
                minutes_id INTEGER,
                meeting_date TEXT,
                qdrant_collection_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES processed_files(id),
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id),
                FOREIGN KEY (minutes_id) REFERENCES meeting_minutes(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER,
                start REAL,
                end REAL,
                speaker TEXT,
                text TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER,
                chunk_id INTEGER,
                from_time REAL,
                to_time REAL,
                speaker TEXT,
                word_count INTEGER,
                chunk_type TEXT,
                file_path TEXT,
                date TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
            )
        """)

def add_file(filename, filepath, filehash, status, api_response):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO processed_files (filename, filepath, filehash, status, api_response)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, filepath, filehash, status, api_response))
        return cursor.lastrowid # Return the ID of the newly inserted row

def add_transcript(content, transcript_path):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO transcripts (content, transcript_path)
            VALUES (?, ?)
        """, (content, transcript_path))
        return cursor.lastrowid

def add_meeting_minutes(summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO meeting_minutes (summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model))
        return cursor.lastrowid

def add_meeting_data(file_id, transcript_id, minutes_id, meeting_date, qdrant_collection_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO meeting_data (file_id, transcript_id, minutes_id, meeting_date, qdrant_collection_name)
            VALUES (?, ?, ?, ?, ?)
        """, (file_id, transcript_id, minutes_id, meeting_date, qdrant_collection_name))
        return cursor.lastrowid

def file_already_processed(filehash):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT 1 FROM processed_files WHERE filehash = ?", (filehash,))
        return cur.fetchone() is not None

def compute_file_hash(file_path):
    import hashlib
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_files_with_error():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT filename, filepath FROM processed_files WHERE status = 'error'")
        return cur.fetchall()

def update_file_status(filehash, status, api_response=""):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE processed_files SET status = ?, api_response = ? WHERE filehash = ?",
            (status, api_response, filehash)
        )

def get_file_status(filehash: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT status FROM processed_files WHERE filehash = ?", (filehash,))
        result = cursor.fetchone()
        return result[0] if result else None

# NOWA FUNKCJA: Pobieranie statusu pliku po ID
def get_file_status_by_id(file_id: int) -> Optional[str]:
    """Retrieves the status of a file by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT status FROM processed_files WHERE id = ?", (file_id,))
        result = cursor.fetchone()
        return result[0] if result else None

# NOWA FUNKCJA: Pobieranie pełnego rekordu pliku (w tym api_response)
def get_file_record_by_hash(filehash: str) -> Optional[Dict[str, Any]]:
    """Retrieves a full file record by its hash."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row # Umożliwia dostęp do kolumn po nazwie
        cursor = conn.execute("SELECT * FROM processed_files WHERE filehash = ?", (filehash,))
        record = cursor.fetchone()
        return dict(record) if record else None

def get_file_record_by_id(file_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM processed_files WHERE id = ?", (file_id,))
        record = cursor.fetchone()
        return dict(record) if record else None

def get_file_record_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Retrieves a full file record by its filename."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row # Umożliwia dostęp do kolumn po nazwie
        cursor = conn.execute("SELECT * FROM processed_files WHERE filename = ? ORDER BY processed_at DESC LIMIT 1", (filename,))
        record = cursor.fetchone()
        return dict(record) if record else None

# NOWA FUNKCJA: Pobieranie pełnego rekordu pliku po filepath
def get_file_record_by_filepath(filepath: str) -> Optional[Dict[str, Any]]:
    """Retrieves a full file record by its file path."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM processed_files WHERE filepath = ?", (filepath,))
        record = cursor.fetchone()
        return dict(record) if record else None

# NOWA FUNKCJA: Wykonywanie zapytań SQL (tylko SELECT)
def execute_read_query(query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    """
    Wykonuje zapytanie SQL typu SELECT na bazie danych i zwraca wyniki.
    Zapewnia bezpieczeństwo poprzez użycie parametryzowanych zapytań.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # Umożliwia dostęp do kolumn po nazwie
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_all_processed_files() -> List[Dict[str, Any]]:
    """Retrieves a summary of all processed files (id, filename, status, processed_at)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Zmieniono SELECT, aby pobierał tylko filename i filehash dla plików ze statusem 'success'
        cursor = conn.execute("SELECT filename, filehash FROM processed_files WHERE status = 'success' ORDER BY processed_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def get_transcript_by_id(transcript_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves the full content and metadata of a transcript by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        record = cursor.fetchone()
        return dict(record) if record else None

def get_meeting_minutes_by_id(minutes_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves the full content and metadata of meeting minutes by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM meeting_minutes WHERE id = ?", (minutes_id,))
        record = cursor.fetchone()
        return dict(record) if record else None

def get_meeting_data_by_file_id(file_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves meeting data linking transcript, minutes, and Qdrant collection by file ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM meeting_data WHERE file_id = ?", (file_id,))
        record = cursor.fetchone()
        return dict(record) if record else None

def get_all_meeting_minutes_summary() -> List[Dict[str, Any]]:
    """Retrieves a summary of all meeting minutes (id, minutes_path, summary_text, generated_at, created_at)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT id, minutes_path, summary_text, generated_at, created_at FROM meeting_minutes ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def add_transcript_segment(transcript_id, start, end, speaker, text):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO transcript_segments (transcript_id, start, end, speaker, text)
            VALUES (?, ?, ?, ?, ?)
        """, (transcript_id, start, end, speaker, text))
        return cursor.lastrowid

def add_transcript_chunk(transcript_id, chunk_id, from_time, to_time, speaker, word_count, chunk_type, file_path, date):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO transcript_chunks (transcript_id, chunk_id, from_time, to_time, speaker, word_count, chunk_type, file_path, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (transcript_id, chunk_id, from_time, to_time, speaker, word_count, chunk_type, file_path, date))
        return cursor.lastrowid

def get_segments_by_transcript(transcript_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT * FROM transcript_segments WHERE transcript_id = ? ORDER BY start
        """, (transcript_id,))
        return [dict(row) for row in cursor.fetchall()]

def search_segments(speaker=None, text_query=None, time_from=None, time_to=None):
    query = "SELECT * FROM transcript_segments WHERE 1=1"
    params = []
    if speaker:
        query += " AND speaker = ?"
        params.append(speaker)
    if text_query:
        query += " AND text LIKE ?"
        params.append(f"%{text_query}%")
    if time_from is not None:
        query += " AND end >= ?"
        params.append(time_from)
    if time_to is not None:
        query += " AND start <= ?"
        params.append(time_to)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def get_transcripts_with_minutes() -> List[Dict[str, Any]]:
    """Zwraca uproszczone informacje o transkryptach z podsumowaniami."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            SELECT 
                t.id as transcript_id,
                mm.summary_text,
                mm.generated_at as minutes_generated_at
            FROM transcripts t
            LEFT JOIN meeting_data md ON t.id = md.transcript_id
            LEFT JOIN meeting_minutes mm ON md.minutes_id = mm.id
            ORDER BY t.created_at DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            result = {
                "transcript_id": row[0],
                "summary_text": row[1] if row[1] else "Brak podsumowania",
                "minutes_generated_at": row[2] if row[2] else "Brak daty"
            }
            results.append(result)
        
        return results

def get_transcripts_with_minutes_limit(limit: int = 5, order: str = "DESC") -> List[Dict[str, Any]]:
    """Zwraca uproszczone informacje o transkryptach z podsumowaniami z limitem i opcjonalnym sortowaniem."""
    if order.upper() not in ["ASC", "DESC"]:
        order = "DESC"  # domyślnie najnowsze
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(f"""
            SELECT 
                t.id as transcript_id,
                mm.summary_text,
                mm.generated_at as minutes_generated_at
            FROM transcripts t
            LEFT JOIN meeting_data md ON t.id = md.transcript_id
            LEFT JOIN meeting_minutes mm ON md.minutes_id = mm.id
            ORDER BY t.created_at {order}
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            result = {
                "transcript_id": row[0],
                "summary_text": row[1] if row[1] else "Brak podsumowania",
                "minutes_generated_at": row[2] if row[2] else "Brak daty"
            }
            results.append(result)
        
        return results

def update_meeting_data_minutes_id(file_id: int, minutes_id: int) -> bool:
    """Updates the minutes_id for a given file_id in the meeting_data table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            UPDATE meeting_data 
            SET minutes_id = ? 
            WHERE file_id = ?
        """, (minutes_id, file_id))
        return cursor.rowcount > 0

def add_meeting_minutes_with_update(summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model, meeting_date_str=None, existing_minutes_id=None):
    """Enhanced version of add_meeting_minutes that supports updating existing records."""
    with sqlite3.connect(DB_PATH) as conn:
        if existing_minutes_id:
            # Update existing record
            cursor = conn.execute("""
                UPDATE meeting_minutes 
                SET summary_text = ?, minutes_path = ?, generated_at = ?, 
                    llm_model = ?, embeddings_model = ?, chunking_method = ?, 
                    transcription_model = ?, diarization_model = ?
                WHERE id = ?
            """, (summary_text, minutes_path, generated_at, llm_model, embeddings_model, 
                  chunking_method, transcription_model, diarization_model, existing_minutes_id))
            return existing_minutes_id
        else:
            # Insert new record
            cursor = conn.execute("""
                INSERT INTO meeting_minutes (summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (summary_text, minutes_path, generated_at, llm_model, embeddings_model, chunking_method, transcription_model, diarization_model))
            return cursor.lastrowid



def search_transcripts_semantic(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Wyszukuje transkrypty używając prostego wyszukiwania tekstowego, przeszukując zarówno treść transkryptów, jak i mówców w segmentach."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row # Aby wyniki były dostępne jako słowniki
            # Pobierz wszystkie transkrypty
            cursor_transcripts = conn.execute("""
                SELECT id, content, transcript_path, created_at
                FROM transcripts
                ORDER BY created_at DESC
            """)
            transcripts = cursor_transcripts.fetchall()
            
            # Pobierz wszystkie segmenty transkrypcji
            cursor_segments = conn.execute("""
                SELECT transcript_id, speaker, text, start, end
                FROM transcript_segments
            """)
            transcript_segments = cursor_segments.fetchall()

        if not transcripts:
            return []

        search_results = []
        query_lower = query.lower()

        # Iteruj przez segmenty transkrypcji, aby znaleźć dopasowania
        for segment in transcript_segments:
            segment_text_lower = segment['text'].lower()
            speaker_lower = segment['speaker'].lower() if segment['speaker'] else ''

            # Sprawdź, czy zapytanie znajduje się w tekście segmentu lub w nazwie mówcy
            if query_lower in segment_text_lower or query_lower in speaker_lower:
                # Znajdź pełną treść transkryptu dla tego segmentu
                # Zmieniono 't' na 't_content'
                full_transcript_row = next((row for row in transcripts if row['id'] == segment['transcript_id']), None)
                
                if full_transcript_row:
                    transcript_id = full_transcript_row['id']
                    # Sprawdź, czy już dodaliśmy ten transkrypt do wyników, aby uniknąć duplikatów
                    if not any(res['transcript_id'] == transcript_id for res in search_results):
                        search_results.append({
                            "transcript_id": transcript_id,
                            "content_snippet": segment['text'], # Możemy zwrócić segment jako snippet
                            "transcript_path": full_transcript_row['transcript_path'],
                            "created_at": full_transcript_row['created_at'],
                            "speaker": segment['speaker'],
                            "start_time": segment['start'],
                            "end_time": segment['end'],
                            "similarity_score": 1.0, # Dla wyszukiwania tekstowego, uznajmy za idealne dopasowanie
                            "occurrences": segment_text_lower.count(query_lower) + speaker_lower.count(query_lower)
                        })
                # Jeśli osiągnęliśmy limit, przerwij
                if len(search_results) >= limit:
                    break
        
        # Jeśli nie znaleziono dopasowań w segmentach, ale chcemy też przeszukać całe transkrypty
        if not search_results and "semantic" in query: # Przykład, możesz to dostosować
            for t in transcripts:
                if query_lower in t['content'].lower():
                    search_results.append({
                        "transcript_id": t['id'],
                        "content_snippet": t['content'][:500] + "...", # Zwracamy początek transkryptu
                        "transcript_path": t['transcript_path'],
                        "created_at": t['created_at'],
                        "similarity_score": 0.9, # Mniej precyzyjne niż segment
                        "occurrences": t['content'].lower().count(query_lower)
                    })
                if len(search_results) >= limit:
                    break


        return search_results

    except Exception as e:
        logging.error(f"Error in semantic search: {e}", exc_info=True)
        return []
