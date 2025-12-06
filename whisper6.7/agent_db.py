"""
Warstwa dostępu do bazy danych oparta o Postgresa (Supabase).
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config

if not config.USE_SUPABASE:
    raise RuntimeError(
        "agent_db został skonfigurowany wyłącznie dla backendu Postgres. "
        "Ustaw USE_SUPABASE=1 i skonfiguruj zmienne SUPABASE_DB_URL / SUPABASE_URL."
    )

from db import postgres as pg_db

RowDict = Dict[str, Any]


def init_db() -> None:
    """
    Tworzy brakujące tabele zgodnie z docelowym schematem Postgresa.
    """
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS processed_files (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            filepath TEXT,
            filehash TEXT UNIQUE,
            status TEXT,
            api_response TEXT,
            processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            owner_username TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id SERIAL PRIMARY KEY,
            file_id INTEGER REFERENCES processed_files(id) ON DELETE CASCADE,
            content TEXT,
            transcript_path TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS meeting_minutes (
            id SERIAL PRIMARY KEY,
            file_id INTEGER REFERENCES processed_files(id) ON DELETE CASCADE,
            summary_text TEXT,
            minutes_path TEXT,
            generated_at TEXT,
            llm_model TEXT,
            embeddings_model TEXT,
            chunking_method TEXT,
            transcription_model TEXT,
            diarization_model TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS transcript_segments (
            id SERIAL PRIMARY KEY,
            transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
            start DOUBLE PRECISION,
            "end" DOUBLE PRECISION,
            speaker TEXT,
            text TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS transcript_chunks (
            id SERIAL PRIMARY KEY,
            transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
            chunk_id INTEGER,
            from_time DOUBLE PRECISION,
            to_time DOUBLE PRECISION,
            speaker TEXT,
            word_count INTEGER,
            chunk_type TEXT,
            file_path TEXT,
            "date" TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS meeting_data (
            id SERIAL PRIMARY KEY,
            file_id INTEGER REFERENCES processed_files(id) ON DELETE CASCADE,
            transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
            minutes_id INTEGER REFERENCES meeting_minutes(id) ON DELETE CASCADE,
            meeting_date TEXT,
            qdrant_collection_name TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS batch_jobs (
            id SERIAL PRIMARY KEY,
            batch_job_id TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            params_json JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS batch_job_files (
            batch_job_id INTEGER REFERENCES batch_jobs(id) ON DELETE CASCADE,
            file_id INTEGER REFERENCES processed_files(id) ON DELETE CASCADE,
            PRIMARY KEY (batch_job_id, file_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        );
        """,
        """
        ALTER TABLE processed_files ADD COLUMN IF NOT EXISTS owner_username TEXT;
        """,
    ]

    for statement in ddl_statements:
        pg_db.execute(statement)


def add_file(
    filename: str,
    filepath: str,
    filehash: str,
    status: str,
    api_response: str,
    owner_username: Optional[str] = None,
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO processed_files (filename, filepath, filehash, status, api_response, owner_username)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (filename, filepath, filehash, status, api_response, owner_username),
        fetch="one",
    )
    return int(row["id"])


def add_transcript(
    content: str,
    transcript_path: str,
    file_id: Optional[int] = None,
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO transcripts (file_id, content, transcript_path)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (file_id, content, transcript_path),
        fetch="one",
    )
    return int(row["id"])


def add_meeting_minutes(
    summary_text: str,
    minutes_path: str,
    generated_at: str,
    llm_model: str,
    embeddings_model: str,
    chunking_method: str,
    transcription_model: str,
    diarization_model: str,
    file_id: Optional[int] = None,
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO meeting_minutes (
            file_id,
            summary_text,
            minutes_path,
            generated_at,
            llm_model,
            embeddings_model,
            chunking_method,
            transcription_model,
            diarization_model
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            file_id,
            summary_text,
            minutes_path,
            generated_at,
            llm_model,
            embeddings_model,
            chunking_method,
            transcription_model,
            diarization_model,
        ),
        fetch="one",
    )
    return int(row["id"])


def add_meeting_data(
    file_id: int,
    transcript_id: Optional[int],
    minutes_id: Optional[int],
    meeting_date: Optional[str],
    qdrant_collection_name: Optional[str],
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO meeting_data (
            file_id,
            transcript_id,
            minutes_id,
            meeting_date,
            qdrant_collection_name
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (file_id, transcript_id, minutes_id, meeting_date, qdrant_collection_name),
        fetch="one",
    )
    return int(row["id"])


def file_already_processed(filehash: str) -> bool:
    row = pg_db.execute(
        "SELECT 1 FROM processed_files WHERE filehash = %s",
        (filehash,),
        fetch="one",
    )
    return row is not None


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def get_files_with_error() -> List[Tuple[str, str]]:
    rows = pg_db.execute(
        "SELECT filename, filepath FROM processed_files WHERE status = 'error'",
        fetch="all",
    ) or []
    return [(row["filename"], row["filepath"]) for row in rows]


def delete_file_by_path(filepath: str) -> bool:
    """
    Usuwa rekord pliku (oraz powiązane dane dzięki ON DELETE CASCADE) na podstawie pełnej ścieżki.

    Zwraca:
        bool: True, jeżeli rekord został usunięty.
    """
    row = pg_db.execute(
        """
        DELETE FROM processed_files
        WHERE filepath = %s
        RETURNING id
        """,
        (filepath,),
        fetch="one",
    )
    return row is not None


def update_file_status(filehash: str, status: str, api_response: str = "") -> None:
    logging.debug(
        "[DB] Updating file status hash=%s status=%s response=%s",
        filehash,
        status,
        api_response,
    )
    pg_db.execute(
        """
        UPDATE processed_files
        SET status = %s,
            api_response = %s,
            processed_at = CURRENT_TIMESTAMP
        WHERE filehash = %s
        """,
        (status, api_response, filehash),
    )


def claim_file_for_user(file_id: int, username: str) -> None:
    """
    Przypisuje istniejący rekord pliku do użytkownika, jeśli nie miał właściciela.
    """
    pg_db.execute(
        """
        UPDATE processed_files
        SET owner_username = %s
        WHERE id = %s AND owner_username IS NULL
        """,
        (username, file_id),
    )


def update_file_status_by_id(file_id: int, status: str, api_response: str = "") -> None:
    logging.debug(
        "[DB] Updating file status id=%s status=%s response=%s",
        file_id,
        status,
        api_response,
    )
    pg_db.execute(
        """
        UPDATE processed_files
        SET status = %s,
            api_response = %s,
            processed_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (status, api_response, file_id),
    )


def get_file_status(filehash: str) -> Optional[str]:
    row = pg_db.execute(
        "SELECT status FROM processed_files WHERE filehash = %s",
        (filehash,),
        fetch="one",
    )
    return row["status"] if row else None


def get_file_status_by_id(file_id: int) -> Optional[str]:
    row = pg_db.execute(
        "SELECT status FROM processed_files WHERE id = %s",
        (file_id,),
        fetch="one",
    )
    return row["status"] if row else None


def update_file_hash(file_id: int, filehash: str) -> None:
    logging.debug("[DB] Updating file hash for id=%s to %s", file_id, filehash)
    pg_db.execute(
        """
        UPDATE processed_files
        SET filehash = %s
        WHERE id = %s
        """,
        (filehash, file_id),
    )


def _record_to_dict(row: Optional[RowDict]) -> Optional[RowDict]:
    return dict(row) if row else None


def get_file_record_by_hash(filehash: str) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM processed_files WHERE filehash = %s",
            (filehash,),
            fetch="one",
        )
    )


def get_file_record_by_filehash(filehash: str) -> Optional[RowDict]:
    return get_file_record_by_hash(filehash)


def get_file_record_by_id(file_id: int) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM processed_files WHERE id = %s",
            (file_id,),
            fetch="one",
        )
    )


def get_file_record_by_filename(filename: str) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            """
            SELECT *
            FROM processed_files
            WHERE filename = %s
            ORDER BY processed_at DESC
            LIMIT 1
            """,
            (filename,),
            fetch="one",
        )
    )


def get_file_record_by_filepath(filepath: str) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM processed_files WHERE filepath = %s",
            (filepath,),
            fetch="one",
        )
    )


def execute_read_query(query: str, params: Sequence[Any] = ()) -> List[RowDict]:
    rows = pg_db.execute(query, params, fetch="all") or []
    return [dict(row) for row in rows]


def get_all_processed_files() -> List[RowDict]:
    rows = pg_db.execute(
        """
        SELECT id, filename, filepath, filehash, status, processed_at
        FROM processed_files
        WHERE status = 'completed'
        ORDER BY processed_at DESC
        """,
        fetch="all",
    ) or []

    result: List[RowDict] = []
    for row in rows:
        row_dict = dict(row)
        row_dict["filename"] = Path(row_dict["filepath"]).name if row_dict["filepath"] else row_dict["filename"]
        row_dict.pop("filepath", None)
        result.append(row_dict)
    return result


def get_processed_files_for_user(username: str) -> List[RowDict]:
    """
    Zwraca listę transkryptów dla użytkownika wraz ze wszystkimi wersjami minut.
    Każda wersja podsumowania spotkania (meeting_minutes) otrzymuje osobny rekord.
    """
    rows = pg_db.execute(
        """
        SELECT
            pf.id AS file_id,
            pf.filename,
            pf.filehash,
            pf.status,
            pf.api_response,
            pf.processed_at,
            pf.filepath,
            pf.owner_username,
            t.id AS transcript_id,
            t.transcript_path,
            t.created_at AS transcript_created_at
        FROM processed_files pf
        LEFT JOIN (
            SELECT DISTINCT ON (file_id)
                id,
                file_id,
                transcript_path,
                created_at
            FROM transcripts
            ORDER BY file_id, created_at DESC
        ) AS t ON t.file_id = pf.id
        WHERE pf.owner_username = %s OR pf.owner_username IS NULL
        ORDER BY pf.processed_at DESC NULLS LAST, pf.id DESC
        """,
        (username,),
        fetch="all",
    ) or []

    base_records: Dict[int, RowDict] = {}
    file_ids: List[int] = []

    for row in rows:
        row_dict = dict(row)
        file_id = row_dict["file_id"]

        processed_at = row_dict.get("processed_at")
        if processed_at is not None:
            if hasattr(processed_at, "isoformat"):
                row_dict["processed_at"] = processed_at.isoformat()
            else:
                try:
                    row_dict["processed_at"] = datetime.fromisoformat(str(processed_at)).isoformat()
                except ValueError:
                    row_dict["processed_at"] = str(processed_at)

        transcript_created_at = row_dict.get("transcript_created_at")
        if transcript_created_at is not None:
            if hasattr(transcript_created_at, "isoformat"):
                row_dict["transcript_created_at"] = transcript_created_at.isoformat()
            else:
                try:
                    row_dict["transcript_created_at"] = datetime.fromisoformat(str(transcript_created_at)).isoformat()
                except ValueError:
                    row_dict["transcript_created_at"] = str(transcript_created_at)

        filepath = row_dict.pop("filepath", None)
        if filepath:
            row_dict["filename"] = Path(filepath).name

        owner_username = row_dict.get("owner_username")
        if owner_username is None:
            claim_file_for_user(file_id, username)
            row_dict["owner_username"] = username

        row_dict["has_transcript"] = row_dict.get("transcript_id") is not None
        row_dict["has_minutes"] = False

        base_records[file_id] = row_dict

    file_ids = list(base_records.keys())

    minutes_map: Dict[int, List[RowDict]] = defaultdict(list)
    if file_ids:
        minutes_rows = pg_db.execute(
            """
            SELECT
                id,
                file_id,
                summary_text,
                minutes_path,
                generated_at,
                created_at,
                llm_model,
                embeddings_model,
                chunking_method,
                transcription_model,
                diarization_model
            FROM meeting_minutes
            WHERE file_id = ANY(%s)
            ORDER BY created_at ASC
            """,
            (file_ids,),
            fetch="all",
        ) or []

        for row in minutes_rows:
            minutes_dict = dict(row)
            created_at = minutes_dict.get("created_at")
            if created_at is not None and hasattr(created_at, "isoformat"):
                minutes_dict["created_at"] = created_at.isoformat()
            generated_at = minutes_dict.get("generated_at")
            if generated_at is not None and not hasattr(generated_at, "isoformat"):
                try:
                    minutes_dict["generated_at"] = datetime.fromisoformat(str(generated_at)).isoformat()
                except ValueError:
                    minutes_dict["generated_at"] = str(generated_at)
            summary_text = minutes_dict.pop("summary_text", "") or ""
            minutes_dict["summary_preview"] = " ".join(summary_text.split())[:280]
            minutes_dict["summary_length"] = len(summary_text)
            minutes_map[minutes_dict["file_id"]].append(minutes_dict)

    results: List[RowDict] = []

    for file_id, base_data in base_records.items():
        minutes_list = minutes_map.get(file_id, [])
        if minutes_list:
            minutes_sorted = sorted(
                minutes_list,
                key=lambda item: item.get("created_at") or item.get("generated_at") or "",
                reverse=True,
            )
            total_versions = len(minutes_sorted)
            for index, minutes_entry in enumerate(minutes_sorted, start=1):
                entry = dict(base_data)
                entry["has_minutes"] = True
                entry["minutes_id"] = minutes_entry["id"]
                entry["minutes_path"] = minutes_entry.get("minutes_path")
                entry["minutes_generated_at"] = minutes_entry.get("generated_at")
                entry["minutes_created_at"] = minutes_entry.get("created_at")
                entry["minutes_version"] = total_versions - index + 1
                entry["is_latest_minutes"] = index == 1
                entry["minutes_summary_preview"] = minutes_entry.get("summary_preview")
                entry["minutes_summary_length"] = minutes_entry.get("summary_length", 0)
                results.append(entry)
        else:
            entry = dict(base_data)
            entry["has_minutes"] = False
            entry["minutes_id"] = None
            entry["minutes_path"] = None
            entry["minutes_generated_at"] = None
            entry["minutes_created_at"] = None
            entry["minutes_version"] = None
            entry["is_latest_minutes"] = True
            entry["minutes_summary_preview"] = None
            entry["minutes_summary_length"] = 0
            results.append(entry)

    def sort_key(item: RowDict) -> str:
        return (
            item.get("minutes_created_at")
            or item.get("minutes_generated_at")
            or item.get("processed_at")
            or ""
        )

    results.sort(key=sort_key, reverse=True)
    return results


def get_transcript_by_id(transcript_id: int) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM transcripts WHERE id = %s",
            (transcript_id,),
            fetch="one",
        )
    )


def get_transcript_by_file_id(file_id: int) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            """
            SELECT *
            FROM transcripts
            WHERE file_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (file_id,),
            fetch="one",
        )
    )


def get_meeting_minutes_by_id(minutes_id: int) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM meeting_minutes WHERE id = %s",
            (minutes_id,),
            fetch="one",
        )
    )


def get_meeting_data_by_file_id(file_id: int) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM meeting_data WHERE file_id = %s",
            (file_id,),
            fetch="one",
        )
    )


def delete_file_by_id_for_user(file_id: int, username: str) -> bool:
    row = pg_db.execute(
        """
        DELETE FROM processed_files
        WHERE id = %s AND (owner_username = %s OR owner_username IS NULL)
        RETURNING id
        """,
        (file_id, username),
        fetch="one",
    )
    return row is not None


def get_all_meeting_minutes_summary() -> List[RowDict]:
    rows = pg_db.execute(
        """
        SELECT id, minutes_path, summary_text, generated_at, created_at
        FROM meeting_minutes
        ORDER BY created_at DESC
        """,
        fetch="all",
    ) or []
    return [dict(row) for row in rows]


def add_transcript_segment(
    transcript_id: int,
    start: float,
    end: float,
    speaker: str,
    text: str,
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO transcript_segments (transcript_id, start, "end", speaker, text)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (transcript_id, start, end, speaker, text),
        fetch="one",
    )
    return int(row["id"])


def add_transcript_chunk(
    transcript_id: int,
    chunk_id: int,
    from_time: float,
    to_time: float,
    speaker: str,
    word_count: int,
    chunk_type: str,
    file_path: str,
    date: str,
) -> int:
    row = pg_db.execute(
        """
        INSERT INTO transcript_chunks (
            transcript_id,
            chunk_id,
            from_time,
            to_time,
            speaker,
            word_count,
            chunk_type,
            file_path,
            "date"
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            transcript_id,
            chunk_id,
            from_time,
            to_time,
            speaker,
            word_count,
            chunk_type,
            file_path,
            date,
        ),
        fetch="one",
    )
    return int(row["id"])


def get_segments_by_transcript(transcript_id: int) -> List[RowDict]:
    rows = pg_db.execute(
        """
        SELECT *
        FROM transcript_segments
        WHERE transcript_id = %s
        ORDER BY start
        """,
        (transcript_id,),
        fetch="all",
    ) or []
    return [dict(row) for row in rows]


def search_segments(
    speaker: Optional[str] = None,
    text_query: Optional[str] = None,
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
    collection_name: Optional[str] = None,
    username: Optional[str] = None,
) -> List[RowDict]:
    clauses: List[str] = []
    params: List[Any] = []

    if speaker:
        clauses.append("ts.speaker = %s")
        params.append(speaker)
    if text_query:
        clauses.append("ts.text ILIKE %s")
        params.append(f"%{text_query}%")
    if time_from is not None:
        clauses.append('ts."end" >= %s')
        params.append(time_from)
    if time_to is not None:
        clauses.append("ts.start <= %s")
        params.append(time_to)
    if collection_name:
        clauses.append("md.qdrant_collection_name = %s")
        params.append(collection_name)
    if username:
        clauses.append("(pf.owner_username = %s OR pf.owner_username IS NULL)")
        params.append(username)

    where_clause = " AND ".join(clauses)
    if where_clause:
        where_clause = " AND " + where_clause

    query = f"""
        SELECT
            ts.*,
            md.qdrant_collection_name,
            pf.id AS file_id,
            pf.owner_username
        FROM transcript_segments ts
        JOIN transcripts t ON ts.transcript_id = t.id
        JOIN meeting_data md ON t.id = md.transcript_id
        JOIN processed_files pf ON t.file_id = pf.id
        WHERE 1=1
        {where_clause}
        ORDER BY ts.start
    """

    rows = pg_db.execute(query, tuple(params), fetch="all") or []
    results: List[RowDict] = []
    for row in rows:
        row_dict = dict(row)
        owner_username = row_dict.pop("owner_username", None)
        file_id = row_dict.pop("file_id", None)
        if username:
            if file_id is None:
                continue
            if owner_username is None:
                claim_file_for_user(int(file_id), username)
                owner_username = username
            if owner_username != username:
                continue
        results.append(row_dict)
    return results


def search_transcripts_semantic(
    query: str,
    limit: int = 5,
    collection_name: Optional[str] = None,
    username: Optional[str] = None,
) -> List[RowDict]:
    try:
        transcript_where = []
        params: List[Any] = []

        if collection_name:
            transcript_where.append("md.qdrant_collection_name = %s")
            params.append(collection_name)
        if username:
            transcript_where.append("(pf.owner_username = %s OR pf.owner_username IS NULL)")
            params.append(username)

        transcript_where_clause = ""
        if transcript_where:
            transcript_where_clause = " AND " + " AND ".join(transcript_where)

        transcripts = pg_db.execute(
            f"""
            SELECT
                t.id,
                t.content,
                t.transcript_path,
                t.created_at,
                md.qdrant_collection_name,
                pf.id AS file_id,
                pf.owner_username
            FROM transcripts t
            LEFT JOIN meeting_data md ON t.id = md.transcript_id
            JOIN processed_files pf ON t.file_id = pf.id
            WHERE 1=1
            {transcript_where_clause}
            ORDER BY t.created_at DESC
            """,
            tuple(params),
            fetch="all",
        ) or []

        segments = pg_db.execute(
            f"""
            SELECT
                ts.transcript_id,
                ts.speaker,
                ts.text,
                ts.start,
                ts."end",
                md.qdrant_collection_name,
                pf.id AS file_id,
                pf.owner_username
            FROM transcript_segments ts
            INNER JOIN transcripts t ON ts.transcript_id = t.id
            LEFT JOIN meeting_data md ON t.id = md.transcript_id
            JOIN processed_files pf ON t.file_id = pf.id
            WHERE 1=1
            {transcript_where_clause}
            """,
            tuple(params),
            fetch="all",
        ) or []

        if not transcripts and not segments:
            return []

        transcripts_map: Dict[int, RowDict] = {}
        for row in transcripts:
            row_dict = dict(row)
            owner_username = row_dict.pop("owner_username", None)
            file_id = row_dict.pop("file_id", None)
            if username:
                if file_id is None:
                    continue
                if owner_username is None:
                    claim_file_for_user(int(file_id), username)
                    owner_username = username
                if owner_username != username:
                    continue
            transcripts_map[row_dict["id"]] = row_dict
        search_results: List[RowDict] = []
        query_lower = query.lower()
        added_transcript_ids = set()

        for segment in segments:
            segment_dict = dict(segment)
            owner_username = segment_dict.pop("owner_username", None)
            file_id = segment_dict.pop("file_id", None)
            if username:
                if file_id is None:
                    continue
                if owner_username is None:
                    claim_file_for_user(int(file_id), username)
                    owner_username = username
                if owner_username != username:
                    continue

            segment_text = (segment_dict["text"] or "").lower()
            speaker_lower = (segment_dict["speaker"] or "").lower()
            if query_lower in segment_text or query_lower in speaker_lower:
                transcript_id = segment_dict["transcript_id"]
                if transcript_id in added_transcript_ids:
                    continue

                full_transcript = transcripts_map.get(transcript_id)
                if full_transcript:
                    search_results.append(
                        {
                            "transcript_id": transcript_id,
                            "content_snippet": segment_dict["text"],
                            "transcript_path": full_transcript["transcript_path"],
                            "created_at": full_transcript["created_at"],
                            "qdrant_collection_name": full_transcript.get("qdrant_collection_name"),
                            "speaker": segment_dict["speaker"],
                            "start_time": segment_dict["start"],
                            "end_time": segment_dict["end"],
                            "similarity_score": 1.0,
                            "occurrences": segment_text.count(query_lower) + speaker_lower.count(query_lower),
                        }
                    )
                    added_transcript_ids.add(transcript_id)

                if len(search_results) >= limit:
                    break

        if not search_results:
            for transcript_id, transcript in transcripts_map.items():
                content_lower = (transcript.get("content") or "").lower()
                if query_lower in content_lower and transcript_id not in added_transcript_ids:
                    search_results.append(
                        {
                            "transcript_id": transcript_id,
                            "content_snippet": (transcript.get("content") or "")[:500] + "...",
                            "transcript_path": transcript.get("transcript_path"),
                            "created_at": transcript.get("created_at"),
                            "qdrant_collection_name": transcript.get("qdrant_collection_name"),
                            "similarity_score": 0.9,
                            "occurrences": content_lower.count(query_lower),
                        }
                    )
                    added_transcript_ids.add(transcript_id)
                if len(search_results) >= limit:
                    break

        return search_results
    except Exception as exc:
        logging.error("Error in semantic search: %s", exc, exc_info=True)
        return []


def get_transcripts_with_minutes() -> List[RowDict]:
    rows = pg_db.execute(
        """
        SELECT
            t.id AS transcript_id,
            mm.summary_text,
            mm.generated_at AS minutes_generated_at
        FROM transcripts t
        LEFT JOIN meeting_data md ON t.id = md.transcript_id
        LEFT JOIN meeting_minutes mm ON md.minutes_id = mm.id
        ORDER BY t.created_at DESC
        """,
        fetch="all",
    ) or []

    results: List[RowDict] = []
    for row in rows:
        results.append(
            {
                "transcript_id": row["transcript_id"],
                "summary_text": row["summary_text"] or "Brak podsumowania",
                "minutes_generated_at": row["minutes_generated_at"] or "Brak daty",
            }
        )
    return results


def get_transcripts_with_minutes_limit(
    limit: int = 5,
    order: str = "DESC",
) -> List[RowDict]:
    order_upper = order.upper()
    if order_upper not in ("ASC", "DESC"):
        order_upper = "DESC"

    rows = pg_db.execute(
        f"""
        SELECT
            t.id AS transcript_id,
            mm.summary_text,
            mm.generated_at AS minutes_generated_at
        FROM transcripts t
        LEFT JOIN meeting_data md ON t.id = md.transcript_id
        LEFT JOIN meeting_minutes mm ON md.minutes_id = mm.id
        ORDER BY t.created_at {order_upper}
        LIMIT %s
        """,
        (limit,),
        fetch="all",
    ) or []

    results: List[RowDict] = []
    for row in rows:
        results.append(
            {
                "transcript_id": row["transcript_id"],
                "summary_text": row["summary_text"] or "Brak podsumowania",
                "minutes_generated_at": row["minutes_generated_at"] or "Brak daty",
            }
        )
    return results


def update_meeting_data_minutes_id(file_id: int, minutes_id: int) -> bool:
    pg_db.execute(
        """
        UPDATE meeting_data
        SET minutes_id = %s
        WHERE file_id = %s
        """,
        (minutes_id, file_id),
    )
    return True


def add_meeting_minutes_with_update(
    summary_text: str,
    minutes_path: str,
    generated_at: str,
    llm_model: str,
    embeddings_model: str,
    chunking_method: str,
    transcription_model: str,
    diarization_model: str,
    meeting_date_str: Optional[str] = None,
    existing_minutes_id: Optional[int] = None,
    file_id: Optional[int] = None,
) -> int:
    if existing_minutes_id:
        pg_db.execute(
            """
            UPDATE meeting_minutes
            SET summary_text = %s,
                minutes_path = %s,
                generated_at = %s,
                llm_model = %s,
                embeddings_model = %s,
                chunking_method = %s,
                transcription_model = %s,
                diarization_model = %s
            WHERE id = %s
            """,
            (
                summary_text,
                minutes_path,
                generated_at,
                llm_model,
                embeddings_model,
                chunking_method,
                transcription_model,
                diarization_model,
                existing_minutes_id,
            ),
        )
        return existing_minutes_id
    return add_meeting_minutes(
        summary_text,
        minutes_path,
        generated_at,
        llm_model,
        embeddings_model,
        chunking_method,
        transcription_model,
        diarization_model,
        file_id=file_id,
    )


def create_user(username: str, hashed_password: str) -> Optional[int]:
    try:
        row = pg_db.execute(
            """
            INSERT INTO users (username, hashed_password)
            VALUES (%s, %s)
            RETURNING id
            """,
            (username, hashed_password),
            fetch="one",
        )
        return int(row["id"])
    except Exception as exc:  # pragma: no cover - unikalne ograniczenie
        if getattr(exc, "sqlstate", "") == "23505":
            logging.error("User with username %s already exists.", username)
            return None
        raise


def get_user(username: str) -> Optional[RowDict]:
    return _record_to_dict(
        pg_db.execute(
            "SELECT * FROM users WHERE username = %s",
            (username,),
            fetch="one",
        )
    )


def add_batch_job(
    batch_job_id: str,
    status: str,
    params: Optional[Dict[str, Any]] = None,
    file_ids: Optional[Sequence[int]] = None,
) -> int:
    params_json = json.dumps(params or {})
    row = pg_db.execute(
        """
        INSERT INTO batch_jobs (batch_job_id, status, params_json)
        VALUES (%s, %s, %s::jsonb)
        RETURNING id
        """,
        (batch_job_id, status, params_json),
        fetch="one",
    )
    db_id = int(row["id"])
    if file_ids:
        set_batch_job_files(db_id, file_ids)
    return db_id


def set_batch_job_files(batch_job_db_id: int, file_ids: Sequence[int]) -> None:
    pg_db.execute(
        "DELETE FROM batch_job_files WHERE batch_job_id = %s",
        (batch_job_db_id,),
    )
    for file_id in file_ids:
        pg_db.execute(
            """
            INSERT INTO batch_job_files (batch_job_id, file_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (batch_job_db_id, file_id),
        )


def append_batch_job_file(batch_job_db_id: int, file_id: int) -> None:
    pg_db.execute(
        """
        INSERT INTO batch_job_files (batch_job_id, file_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        (batch_job_db_id, file_id),
    )


def get_batch_job(batch_job_id: str) -> Optional[RowDict]:
    record = pg_db.execute(
        "SELECT * FROM batch_jobs WHERE batch_job_id = %s",
        (batch_job_id,),
        fetch="one",
    )
    if not record:
        return None

    record_dict = dict(record)

    params_value = record_dict.get("params_json")
    if isinstance(params_value, str):
        try:
            params_dict = json.loads(params_value)
        except json.JSONDecodeError:
            params_dict = {}
    else:
        params_dict = params_value or {}
    record_dict["params"] = params_dict
    record_dict["params_json"] = json.dumps(params_dict)
    raw_job_ids = params_dict.get("file_job_ids")
    if isinstance(raw_job_ids, list):
        record_dict["file_job_ids_json"] = json.dumps(raw_job_ids)
    else:
        record_dict["file_job_ids_json"] = json.dumps([])

    files_rows = pg_db.execute(
        "SELECT file_id FROM batch_job_files WHERE batch_job_id = %s ORDER BY file_id",
        (record_dict["id"],),
        fetch="all",
    ) or []
    file_ids = [row["file_id"] for row in files_rows]
    record_dict["file_ids"] = file_ids

    file_hashes: List[str] = []
    if file_ids:
        hash_rows = pg_db.execute(
            "SELECT id, filehash FROM processed_files WHERE id = ANY(%s)",
            (file_ids,),
            fetch="all",
        ) or []
        hash_map = {row["id"]: row["filehash"] for row in hash_rows}
        for file_id in file_ids:
            file_hash = hash_map.get(file_id)
            if file_hash:
                file_hashes.append(file_hash)
    record_dict["file_hashes"] = file_hashes
    return record_dict


def update_batch_job_status(batch_job_id: str, status: str) -> bool:
    pg_db.execute(
        """
        UPDATE batch_jobs
        SET status = %s
        WHERE batch_job_id = %s
        """,
        (status, batch_job_id),
    )
    return True


def update_batch_job_file_ids_json(batch_job_id: str, file_ids_json: str) -> bool:
    raw_items = json.loads(file_ids_json) if file_ids_json else []
    batch_record = get_batch_job(batch_job_id)
    if not batch_record:
        return False

    params_update = json.dumps({"file_job_ids": raw_items})
    pg_db.execute(
        """
        UPDATE batch_jobs
        SET params_json = COALESCE(params_json, '{}'::jsonb) || %s::jsonb
        WHERE id = %s
        """,
        (params_update, batch_record["id"]),
    )

    numeric_file_ids: List[int] = []
    for item in raw_items:
        if isinstance(item, int):
            numeric_file_ids.append(item)
        else:
            file_hash = str(item)
            record = get_file_record_by_hash(file_hash)
            if record and record.get("id") is not None:
                numeric_file_ids.append(int(record["id"]))
            else:
                logging.warning(
                    "Could not resolve file hash '%s' to processed_files.id for batch '%s'. Skipping.",
                    file_hash,
                    batch_job_id,
                )

    set_batch_job_files(batch_record["id"], numeric_file_ids)
    return True

