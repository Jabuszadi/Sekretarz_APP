import sys
import os

# Dodanie katalogu głównego projektu do ścieżki Pythona
# Pozwala to na importowanie modułów z katalogu głównego, takich jak agent_db
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import agent_db 
from summarizer import generate_chat_response
import config 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Konfiguracja logowania
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # USUNIĘTO
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Inicjalizacja FastMCP dla Agenta Bazy Danych
mcp_database = FastMCP()

# --- Modele Pydantic dla narzędzi bazodanowych (pozostałe, które są zwracane) ---

class ProcessedFileSummary(BaseModel):
    id: Optional[int] = None
    filename: str
    status: Optional[str] = None
    processed_at: Optional[str] = None
    filepath: Optional[str] = None
    filehash: Optional[str] = None
    api_response: Optional[str] = None
    error_message: Optional[str] = None

class TranscriptRecord(BaseModel):
    id: int
    content: str
    transcript_path: str
    created_at: str

class MeetingMinutesRecord(BaseModel):
    id: int
    summary_text: str
    minutes_path: str
    generated_at: str
    llm_model: Optional[str] = None
    embeddings_model: Optional[str] = None
    chunking_method: Optional[str] = None
    transcription_model: Optional[str] = None
    diarization_model: Optional[str] = None
    created_at: str

class MeetingMinutesSummary(BaseModel):
    id: int
    minutes_path: str
    summary_text: str
    generated_at: str
    created_at: str

class MeetingDataRecord(BaseModel):
    id: int
    file_id: int
    transcript_id: int
    minutes_id: int
    meeting_date: str
    qdrant_collection_name: str
    created_at: str

# --- Helpers ---

def _get_file_record_for_user(file_id: Optional[int], username: str) -> Optional[Dict[str, Any]]:
    if file_id is None:
        return None
    record = agent_db.get_file_record_by_id(file_id)
    if not record:
        return None
    owner = record.get("owner_username")
    if owner is None:
        agent_db.claim_file_for_user(file_id, username)
        record = agent_db.get_file_record_by_id(file_id)
        owner = record.get("owner_username")
    if owner != username:
        return None
    return record


def _to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def _get_transcript_for_user(transcript_id: int, username: str) -> Optional[Dict[str, Any]]:
    transcript = agent_db.get_transcript_by_id(transcript_id)
    if not transcript:
        return None
    file_id = transcript.get("file_id")
    if _get_file_record_for_user(file_id, username) is None:
        return None
    return transcript


def _get_minutes_for_user(minutes_id: int, username: str) -> Optional[Dict[str, Any]]:
    minutes = agent_db.get_meeting_minutes_by_id(minutes_id)
    if not minutes:
        return None
    file_id_rows = agent_db.execute_read_query(
        "SELECT file_id FROM meeting_data WHERE minutes_id = %s LIMIT 1",
        (minutes_id,),
    )
    file_id = file_id_rows[0]["file_id"] if file_id_rows else None
    if _get_file_record_for_user(file_id, username) is None:
        return None
    return minutes


def _get_meeting_data_for_user(file_id: int, username: str) -> Optional[Dict[str, Any]]:
    if _get_file_record_for_user(file_id, username) is None:
        return None
    return agent_db.get_meeting_data_by_file_id(file_id)


def _filter_minutes_summary_for_user(
    summaries: List[Dict[str, Any]],
    username: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in summaries:
        minutes_id = item.get("id") or item.get("minutes_id") or item.get("transcript_id")
        if minutes_id is None:
            continue
        file_id_rows = agent_db.execute_read_query(
            "SELECT file_id FROM meeting_data WHERE minutes_id = %s LIMIT 1",
            (minutes_id,),
        )
        file_id = file_id_rows[0]["file_id"] if file_id_rows else None
        if _get_file_record_for_user(file_id, username):
            filtered.append(item)
    return filtered

# --- Narzędzia FastMCP dla interakcji z bazą danych ---

@mcp_database.tool("get_all_processed_files", description="Retrieves a summary of processed files for the given user including their ID, filename, status, and processing timestamp.")
async def get_all_processed_files_tool(username: str) -> List[ProcessedFileSummary]:
    logging.info("Attempting to retrieve all processed files summary.")
    try:
        processed_files_summary = agent_db.get_processed_files_for_user(username)
        logging.info(f"Retrieved {len(processed_files_summary)} processed files for user %s.", username)
        summaries: List[ProcessedFileSummary] = []
        seen_ids: Set[int] = set()
        for entry in processed_files_summary:
            file_id = entry.get("file_id")
            if file_id is None or file_id in seen_ids:
                continue
            if _get_file_record_for_user(file_id, username) is None:
                continue
            seen_ids.add(file_id)
            summaries.append(
                ProcessedFileSummary(
                    id=file_id,
                    filename=entry.get("filename", "unknown"),
                    status=entry.get("status"),
                    processed_at=_to_iso(entry.get("processed_at")),
                    filepath=entry.get("filepath"),
                    filehash=entry.get("filehash"),
                    api_response=entry.get("api_response"),
                    error_message=entry.get("api_response") if entry.get("status") == "error" else entry.get("error_message"),
                )
            )
        return summaries
    except Exception as e:
        logging.error(f"Error retrieving all processed files: {e}", exc_info=True)
        return []

@mcp_database.tool("get_transcript_by_id", description="Retrieves the full content and metadata of a transcript by its database ID.")
async def get_transcript_by_id_tool(id: int, username: str) -> Optional[TranscriptRecord]:
    logging.info(f"Attempting to retrieve transcript with ID: {id}")
    try:
        transcript = _get_transcript_for_user(id, username)
        if transcript:
            logging.info(f"Retrieved transcript for ID: {id}")
            return TranscriptRecord(
                id=transcript["id"],
                content=transcript.get("content") or "",
                transcript_path=transcript.get("transcript_path"),
                created_at=_to_iso(transcript.get("created_at")) or "",
            )
        else:
            logging.warning(f"No transcript found for ID: {id}")
        return None
    except Exception as e:
        logging.error(f"Error retrieving transcript by ID {id}: {e}", exc_info=True)
        return None

@mcp_database.tool("get_meeting_minutes_by_id", description="Retrieves the full content and metadata of meeting minutes by its database ID.")
async def get_meeting_minutes_by_id_tool(id: int, username: str) -> Optional[MeetingMinutesRecord]:
    logging.info(f"Attempting to retrieve meeting minutes with ID: {id}")
    try:
        minutes = _get_minutes_for_user(id, username)
        if minutes:
            logging.info(f"Retrieved meeting minutes for ID: {id}")
            return MeetingMinutesRecord(
                id=minutes["id"],
                summary_text=minutes.get("summary_text") or "",
                minutes_path=minutes.get("minutes_path") or "",
                generated_at=_to_iso(minutes.get("generated_at")) or "",
                llm_model=minutes.get("llm_model"),
                embeddings_model=minutes.get("embeddings_model"),
                chunking_method=minutes.get("chunking_method"),
                transcription_model=minutes.get("transcription_model"),
                diarization_model=minutes.get("diarization_model"),
                created_at=_to_iso(minutes.get("created_at")) or "",
            )
        else:
            logging.warning(f"No meeting minutes found for ID: {id}")
        return None
    except Exception as e:
        logging.error(f"Error retrieving meeting minutes by ID {id}: {e}", exc_info=True)
        return None

@mcp_database.tool("get_meeting_data_by_file_id", description="Retrieves meeting data linking transcript, minutes, and Qdrant collection by file ID.")
async def get_meeting_data_by_file_id_tool(file_id: int, username: str) -> Optional[MeetingDataRecord]:
    logging.info(f"Attempting to retrieve meeting data for file ID: {file_id}")
    try:
        meeting_data = _get_meeting_data_for_user(file_id, username)
        if meeting_data:
            logging.info(f"Retrieved meeting data for file ID: {file_id}")
            return MeetingDataRecord(
                id=meeting_data["id"],
                file_id=meeting_data["file_id"],
                transcript_id=meeting_data.get("transcript_id"),
                minutes_id=meeting_data.get("minutes_id"),
                meeting_date=_to_iso(meeting_data.get("meeting_date")) or "",
                qdrant_collection_name=meeting_data.get("qdrant_collection_name") or "",
                created_at=_to_iso(meeting_data.get("created_at")) or "",
            )
        else:
            logging.warning(f"No meeting data found for file ID: {file_id}")
        return None
    except Exception as e:
        logging.error(f"Error retrieving meeting data by file ID {file_id}: {e}", exc_info=True)
        return None

@mcp_database.tool("get_all_meeting_minutes_summary", description="Retrieves a summary of stored meeting minutes (ID, path, summary text, and creation timestamp) for the given user.")
async def get_all_meeting_minutes_summary_tool(username: str) -> List[MeetingMinutesSummary]: 
    logging.info("Attempting to retrieve all meeting minutes summary.")
    try:
        minutes_summaries = agent_db.get_all_meeting_minutes_summary()
        filtered = _filter_minutes_summary_for_user(minutes_summaries, username)
        logging.info("Retrieved %s meeting minutes summary for user %s.", len(filtered), username)
        if not filtered:
            return []
        return [
            MeetingMinutesSummary(
                id=item["id"],
                minutes_path=item.get("minutes_path") or "",
                summary_text=item.get("summary_text") or "",
                generated_at=_to_iso(item.get("generated_at")) or "",
                created_at=_to_iso(item.get("created_at")) or "",
            )
            for item in filtered
        ]
    except Exception as e:
        logging.error(f"Error retrieving all meeting minutes summary: {e}", exc_info=True)
        return []

@mcp_database.tool("execute_sql_query", description="Wykonuje zapytanie SQL typu SELECT na bazie danych i zwraca wyniki. Użyj tego, aby uzyskać dane z baz danych, które nie są dostępne poprzez inne narzędzia. Pamiętaj, aby podawać pełne, poprawne zapytania SQL, np. SELECT * FROM processed_files LIMIT 5.")
async def execute_sql_query_tool(query: str, _username: str) -> List[Dict[str, Any]]:
    logging.info(f"Attempting to execute SQL query: {query}")
    try:
        if not query.strip().upper().startswith("SELECT"):
            return [{"error": "Narzędzie execute_sql_query obsługuje tylko zapytania SELECT."}]

        results = agent_db.execute_read_query(query)
        
        if not results:
            return [] 
        return results 
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}", exc_info=True)
        return [{"error": f"Przepraszam, wystąpił błąd podczas wykonywania zapytania SQL: {e}"}]

@mcp_database.tool("get_file_record_by_filename", description="Retrieves a full file record by its filename.")
async def get_file_record_by_filename_tool(filename: str, username: str) -> Optional[ProcessedFileSummary]:
    logging.info(f"Attempting to retrieve file record by filename: {filename}")
    try:
        record = agent_db.get_file_record_by_filename(filename)
        if record:
            user_record = _get_file_record_for_user(record.get("id"), username)
            if user_record:
                return ProcessedFileSummary(**user_record)
        return None
    except Exception as e:
        logging.error(f"Error retrieving file record by filename {filename}: {e}", exc_info=True)
        return None

@mcp_database.tool("get_file_record_by_id", description="Retrieves a full file record by its ID.")
async def get_file_record_by_id_tool(id: int, username: str) -> Optional[ProcessedFileSummary]:
    logging.info(f"Attempting to retrieve file record by ID: {id}")
    try:
        record = _get_file_record_for_user(id, username)
        if record:
            return ProcessedFileSummary(**record)
        return None
    except Exception as e:
        logging.error(f"Error retrieving file record by ID {id}: {e}", exc_info=True)
        return None

@mcp_database.tool("get_file_record_by_filepath", description="Retrieves a full file record by its file path.")
async def get_file_record_by_filepath_tool(filepath: str, username: str) -> Optional[ProcessedFileSummary]:
    logging.info(f"Attempting to retrieve file record by filepath: {filepath}")
    try:
        record = agent_db.get_file_record_by_filepath(filepath)
        if record:
            user_record = _get_file_record_for_user(record.get("id"), username)
            if user_record:
                return ProcessedFileSummary(**user_record)
        return None
    except Exception as e:
        logging.error(f"Error retrieving file record by filepath {filepath}: {e}", exc_info=True)
        return None

@mcp_database.tool("add_meeting_minutes", description="Adds new meeting minutes to the database or updates existing ones. Returns the ID of the new/updated record.")
async def add_meeting_minutes_tool(
    summary_text: str,
    minutes_path: str,
    generated_at: str,
    llm_model: Optional[str] = None,
    embeddings_model: Optional[str] = None,
    chunking_method: Optional[str] = None,
    transcription_model: Optional[str] = None,
    diarization_model: Optional[str] = None,
    meeting_date_str: Optional[str] = None,
    existing_minutes_id: Optional[int] = None,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info("Attempting to add or update meeting minutes.")
    try:
        if existing_minutes_id is not None and username:
            if _get_minutes_for_user(existing_minutes_id, username) is None:
                return {"error": "Nie masz dostępu do tych minut spotkania."}
        minutes_id = agent_db.add_meeting_minutes_with_update(
            summary_text=summary_text,
            minutes_path=minutes_path,
            generated_at=generated_at,
            llm_model=llm_model,
            embeddings_model=embeddings_model,
            chunking_method=chunking_method,
            transcription_model=transcription_model,
            diarization_model=diarization_model,
            meeting_date_str=meeting_date_str,
            existing_minutes_id=existing_minutes_id
        )
        return {"minutes_id": minutes_id}
    except Exception as e:
        logging.error(f"Error adding/updating meeting minutes: {e}", exc_info=True)
        return {"error": f"Error adding/updating meeting minutes: {e}"}

@mcp_database.tool("update_meeting_data_minutes_id", description="Updates the minutes_id for a given file_id in the meeting_data table.")
async def update_meeting_data_minutes_id_tool(file_id: int, minutes_id: int, username: str) -> Dict[str, Any]:
    logging.info(f"Attempting to update meeting_data for file_id {file_id} with minutes_id {minutes_id}.")
    try:
        if _get_file_record_for_user(file_id, username) is None:
            return {"status": "error", "message": "Nie masz dostępu do podanego pliku."}
        minutes_record = _get_minutes_for_user(minutes_id, username)
        if minutes_record is None:
            return {"status": "error", "message": "Nie masz dostępu do podanych minut spotkania."}
        agent_db.update_meeting_data_minutes_id(file_id, minutes_id)
        logging.info("Meeting data updated successfully.")
        return {"status": "success", "message": "Meeting data updated successfully."}
    except Exception as e:
        logging.error(f"Error updating meeting data for file_id {file_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Error updating meeting data: {e}"}

@mcp_database.tool("search_transcripts_sql", description="Wyszukuje segmenty transkrypcji w bazie danych SQLite na podstawie mówcy, zapytania tekstowego, oraz zakresu czasowego. Zwraca listę pasujących segmentów.")
async def search_transcripts_sql_tool(
    speaker: Optional[str] = None,
    text_query: Optional[str] = None,
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
    collection_name: Optional[str] = None, # NOWE: Dodano parametr nazwy kolekcji
    username: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logging.info(f"Searching SQL transcripts with speaker='{speaker}', text_query='{text_query}', time_from={time_from}, time_to={time_to}, collection='{collection_name}'")
    try:
        segments = agent_db.search_segments(
            speaker=speaker,
            text_query=text_query,
            time_from=time_from,
            time_to=time_to,
            collection_name=collection_name, # NOWE: Przekazanie nazwy kolekcji
            username=username,
        )
        if not segments:
            return [{"message": "Nie znaleziono segmentów pasujących do kryteriów."}]
        return segments
    except Exception as e:
        logging.error(f"Error searching SQL transcripts: {e}", exc_info=True)
        return [{"error": f"Wystąpił błąd podczas wyszukiwania w transkryptach SQL: {e}"}]

@mcp_database.tool("chat_with_transcripts", description="Umożliwia swobodną rozmowę z przetworzonymi transkryptami. Domyślnie używa wyszukiwania semantycznego. Możesz wymusić wyszukiwanie w bazie SQL, podając use_sql_search=True i opcjonalne kryteria: sql_speaker, sql_text_query, sql_time_from, sql_time_to.")
async def chat_with_transcripts_tool(
    query: str,
    collection_name: Optional[str] = None, # NOWE: Dodano parametr nazwy kolekcji
    use_sql_search: Optional[bool] = False,
    sql_speaker: Optional[str] = None,
    sql_text_query: Optional[str] = None,
    sql_time_from: Optional[float] = None,
    sql_time_to: Optional[float] = None,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info(f"Chatting with transcripts query: '{query}' in collection: {collection_name}")
    context_documents_content = []
    
    try:
        if use_sql_search:
            logging.info("Using SQL search for context.")
            sql_segments_result = agent_db.search_segments(
                speaker=sql_speaker,
                text_query=sql_text_query,
                time_from=sql_time_from,
                time_to=sql_time_to,
                collection_name=collection_name, # NOWE: Przekazanie nazwy kolekcji
                username=username,
            )
            
            if sql_segments_result and isinstance(sql_segments_result, list) and len(sql_segments_result) > 0 and "error" in sql_segments_result[0]:
                return sql_segments_result[0]
            
            if sql_segments_result and isinstance(sql_segments_result, list) and len(sql_segments_result) > 0 and "message" in sql_segments_result[0] and sql_segments_result[0]["message"] == "Nie znaleziono segmentów pasujących do kryteriów.":
                context_documents_content = []
            elif sql_segments_result:
                context_documents_content = [seg["text"] for seg in sql_segments_result]
        else:
            logging.info(f"Using semantic search for context in collection: {collection_name}")
            semantic_results = agent_db.search_transcripts_semantic(
                query,
                collection_name=collection_name,
                username=username,
            ) # NOWE: Przekazanie nazwy kolekcji
            
            if semantic_results and isinstance(semantic_results, list) and len(semantic_results) > 0 and "error" in semantic_results[0]:
                return semantic_results[0]
            
            context_documents_content = [res["content_snippet"] for res in semantic_results]

        if not context_documents_content:
            return {"response": "Nie znalazłem żadnych informacji w bazie danych, które odpowiadałyby na Twoje pytanie."}

        context_str = "\n".join(context_documents_content)
        response = await generate_chat_response(query, context_str)
        return {"response": response}

    except Exception as e:
        logging.error(f"Error in chat_with_transcripts_tool: {e}", exc_info=True)
        return {"error": f"Wystąpił błąd podczas rozmowy z transkryptami: {e}"}

@mcp_database.tool("get_transcripts_with_minutes", description="Retrieves transcripts with their associated meeting minutes summaries for the given user, showing transcript ID, summary text, and generation date.")
async def get_transcripts_with_minutes_tool(username: str) -> List[Dict[str, Any]]:
    try:
        logging.info("DEBUG: Calling agent_db.get_transcripts_with_minutes().")
        raw_result = agent_db.get_transcripts_with_minutes()
        result: List[Dict[str, Any]] = []
        for item in raw_result:
            transcript_id = item.get("transcript_id")
            transcript = _get_transcript_for_user(transcript_id, username)
            if not transcript:
                continue
            result.append(item)
        logging.info(f"DEBUG: get_transcripts_with_minutes returned {len(result) if result else 0} records.")
        return result
    except Exception as e:
        logging.error(f"FATAL ERROR in get_transcripts_with_minutes_tool: {e}", exc_info=True)
        return [{"error": f"Database error: {str(e)}"}]

@mcp_database.tool("get_top_transcripts", description="Pobiera top X najnowszych transkryptów z podsumowaniami spotkań.")
async def get_top_transcripts_tool(limit: int = 5, username: str = "") -> List[Dict[str, Any]]:
    try:
        logging.info(f"DEBUG: Calling agent_db.get_transcripts_with_minutes_limit(limit={limit}, order='DESC').")
        raw_result = agent_db.get_transcripts_with_minutes_limit(limit=limit, order="DESC")
        result: List[Dict[str, Any]] = []
        for item in raw_result:
            transcript_id = item.get("transcript_id")
            if _get_transcript_for_user(transcript_id, username):
                result.append(item)
        logging.info(f"DEBUG: get_top_transcripts returned {len(result) if result else 0} records.")
        return result
    except Exception as e:
        logging.error(f"FATAL ERROR in get_top_transcripts_tool: {e}", exc_info=True)
        return [{"error": f"Database error: {str(e)}"}]

@mcp_database.tool("get_bottom_transcripts", description="Pobiera bottom X najstarszych transkryptów z podsumowaniami spotkań.")
async def get_bottom_transcripts_tool(limit: int = 5, username: str = "") -> List[Dict[str, Any]]:
    try:
        logging.info(f"DEBUG: Calling agent_db.get_transcripts_with_minutes_limit(limit={limit}, order='ASC').")
        raw_result = agent_db.get_transcripts_with_minutes_limit(limit=limit, order="ASC")
        result: List[Dict[str, Any]] = []
        for item in raw_result:
            transcript_id = item.get("transcript_id")
            if _get_transcript_for_user(transcript_id, username):
                result.append(item)
        logging.info(f"DEBUG: get_bottom_transcripts returned {len(result) if result else 0} records.")
        return result
    except Exception as e:
        logging.error(f"FATAL ERROR in get_bottom_transcripts_tool: {e}", exc_info=True)
        return [{"error": f"Database error: {str(e)}"}]

@mcp_database.tool("search_transcripts_semantic", description="Wyszukuje transkrypty używając cosine similarity z embeddings. Zwraca najbardziej podobne fragmenty.")
async def search_transcripts_semantic_tool(query: str, limit: int = 5, collection_name: Optional[str] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        logging.info(f"Performing semantic search for: '{query}' in collection: {collection_name}")
        result = agent_db.search_transcripts_semantic(query, limit, collection_name, username=username)
        logging.info(f"Semantic search returned {len(result)} results.")
        return result
    except Exception as e:
        logging.error(f"Error in semantic search tool: {e}", exc_info=True)
        return [{"error": f"Semantic search error: {str(e)}"}]

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting up Database FastMCP Server...")
    try:
        logging.info("Initializing database...")
        agent_db.init_db()
        logging.info("Database initialized successfully.")
        logging.info("Running FastMCP database agent...")
        mcp_database.run(transport="http", host="0.0.0.0", port=8002)
    except Exception as e:
        logging.error(f"Failed to start Database Agent: {e}", exc_info=True)
