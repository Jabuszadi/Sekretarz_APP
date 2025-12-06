# minimal_mcp_server.py
import config
import asyncio
import json
import logging
import os
import uuid
import mimetypes
from datetime import datetime
from fastmcp import FastMCP, Context, Client
from file_handlers import cleanup_temp_dir, job_temp_storage
from ingest import ingest_transcription
from minutes_service import generate_and_save_minutes
from pathlib import Path
from qdrant_handler import initialize_qdrant_resources, search_all_collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Set
from utils import (
    extract_date_from_filename,
    get_current_ram_usage_gb,
    get_current_vram_usage_gb,
    get_diarization_models,
    get_file_creation_date,
    get_relevant_date_for_file,
    save_minutes_to_file,
    load_enrolled_speakers,
    build_transcription_metadata,
)
from processor import process_audio, convert_mkv_to_wav as convert_to_wav
import shutil
import tempfile
from models import TranscriptionSegment

import agent_db
import asyncio

# Definicja run_in_threadpool jako alias dla asyncio.to_thread
run_in_threadpool = asyncio.to_thread

from summarizer import gemini_model, generate_minutes_of_meeting, generate_chat_response


from agents.database_agent import mcp_database

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, create_model

import httpx
from contextlib import asynccontextmanager
from contextvars import ContextVar
from jose import JWTError, jwt

# === LLAMAINDEX IMPORTS ===
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from workflows.errors import WorkflowRuntimeError
from fastmcp.client import Client, StreamableHttpTransport
# --- ZMIENNE GLOBALNE ---
mcp_router_client = FastMCP()

# --- Modele Pydantic ---
class ChatQuery(BaseModel):
    query: str = Field(..., description="Zapytanie użytkownika do chatbota.")
    collection_name: Optional[str] = Field(None, description="Opcjonalna nazwa kolekcji Qdrant do użycia w zapytaniu.")
    username: Optional[str] = Field(None, description="Opcjonalna nazwa użytkownika (jeśli klient chce ją jawnie przekazać).")

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Nazwa narzędzia do wywołania.")
    parameters: Dict[str, Any] = Field(..., description="Parametry do przekazania narzędziu.")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Odpowiedź chatbota.")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Lista sugerowanych wywołań narzędzi FastMCP.")

class ListCollectionsParams(BaseModel):
    limit: Optional[int] = Field(None, description="Opcjonalny limit liczby kolekcji do wyświetlenia.")

class QueryMeetingDataParams(BaseModel):
    file_id: Optional[str] = Field(None, description="ID pliku, dla którego szukamy danych.")
    query: str = Field(..., description="Zapytanie tekstowe do przeszukania danych spotkania.")

class GetByIdParams(BaseModel):
    id: int = Field(..., description="ID rekordu w bazie danych.")

class GetByFileIdParams(BaseModel):
    file_id: int = Field(..., description="ID pliku w tabeli processed_files.")

class ExecuteSQLQueryParams(BaseModel):
    query: str = Field(..., description="Zapytanie SQL do wykonania (tylko SELECT).")

class GetTranscriptContentParams(BaseModel):
    file_id: int = Field(..., description="ID pliku, dla którego należy pobrać treść transkrypcji.")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # USUNIĘTO

logger = logging.getLogger(__name__)

class ProcessedFileSummary(BaseModel):
    id: int
    filename: str
    status: str
    processed_at: str
    filepath: Optional[str] = None
    filehash: Optional[str] = None
    api_response: Optional[str] = None
    error_message: Optional[str] = None

class TranscriptRecord(BaseModel):
    id: int
    content: str
    transcript_path: str
    created_at: str

class TranscriptSummary(BaseModel):
    id: int
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


current_username: ContextVar[Optional[str]] = ContextVar("current_username", default=None)


def _extract_token_from_request(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token:
            return token
    cookie_token = request.cookies.get("authToken") or request.cookies.get("access_token")
    if cookie_token:
        cookie_token = cookie_token.strip()
        if cookie_token.lower().startswith("bearer "):
            cookie_token = cookie_token.split(" ", 1)[1].strip()
        if cookie_token:
            return cookie_token
    return None


def _decode_username_from_token(raw_token: str) -> str:
    try:
        payload = jwt.decode(raw_token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise ValueError("Token payload missing 'sub'.")
        return username
    except (JWTError, ValueError) as exc:
        logging.warning("Nieudana dekodacja tokenu JWT: %s", exc, exc_info=True)
        raise HTTPException(status_code=401, detail="Nieprawidłowy token uwierzytelniający.")


def _resolve_username(request: Request, explicit_username: Optional[str] = None) -> str:
    if explicit_username:
        return explicit_username
    token = _extract_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Brak tokenu uwierzytelniającego.")
    return _decode_username_from_token(token)

class MeetingDataRecord(BaseModel):
    id: int
    file_id: int
    transcript_id: int
    minutes_id: int
    meeting_date: str
    qdrant_collection_name: str
    created_at: str

class GetByFilenameParams(BaseModel):
    filename: str = Field(..., description="Nazwa pliku do wyszukania.")

class AddMeetingMinutesParams(BaseModel):
    summary_text: str
    minutes_path: str
    generated_at: str
    llm_model: Optional[str] = None
    embeddings_model: Optional[str] = None
    chunking_method: Optional[str] = None
    transcription_model: Optional[str] = None
    diarization_model: Optional[str] = None
    meeting_date_str: Optional[str] = None
    existing_minutes_id: Optional[int] = None

class UpdateMeetingDataMinutesIdParams(BaseModel):
    file_id: int
    minutes_id: int

# Zmodyfikowana inicjalizacja z wykorzystaniem lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_llamaindex_agent
    logging.info("Starting up Minimal FastMCP Server (API & Router)...")
    try:
        await initialize_qdrant_resources()
        logging.info("✅ Qdrant resources initialized.")
        load_enrolled_speakers()
        logging.info("✅ Enrolled speaker profiles loaded.")

        db_agent_url = "http://127.0.0.1:8002/mcp/"
        logging.info(f"Attempting to connect to Database Agent at {db_agent_url}")

        # === ZMIANA: Użyj tymczasowego klienta do discovery, potem utwórz wrappery ===
        db_agent_tools = []
        max_retries = 10
        retry_delay = 1  # seconds
        db_agent_tools_raw = None
        
        for i in range(max_retries):
            try:
                # Użyj tymczasowego klienta z kontekstem async with
                transport = StreamableHttpTransport(db_agent_url)
                async with Client(transport) as temp_client:
                    db_agent_tools_raw = await temp_client.list_tools()
                logging.info(f"✅ Successfully connected to Database Agent at {db_agent_url} after {i+1} attempts.")
                break # Połączenie udane, wyjdź z pętli
            except Exception as e:
                logging.warning(f"Attempt {i+1}/{max_retries}: Error connecting to Database Agent at {db_agent_url}: {e}")
                if i < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"❌ Failed to connect to Database Agent after {max_retries} attempts.")
                    raise # Rethrow exception if all retries fail

        # Połączenie udane, kontynuuj przetwarzanie narzędzi
        try:
            for i, tool_info in enumerate(db_agent_tools_raw):
                logging.info(f"DEBUG: Processing remote tool {i}: Name={tool_info.name}")
                parameters_schema_dict = {}
                if hasattr(tool_info, 'parameters'):
                    if isinstance(tool_info.parameters, dict):
                        parameters_schema_dict = tool_info.parameters
                    elif hasattr(tool_info.parameters, 'model_dump'):
                        parameters_schema_dict = tool_info.parameters.model_dump()

                tool_args_model = _json_schema_to_pydantic_model(tool_info.name, parameters_schema_dict)
                param_names: Set[str] = set()
                if isinstance(parameters_schema_dict, dict):
                    properties = parameters_schema_dict.get("properties", {})
                    if isinstance(properties, dict):
                        param_names = set(properties.keys())

                # === ZMIANA: Wrapper tworzy nowe połączenie dla każdego wywołania ===
                def make_remote_tool_wrapper(tool_name_closure, expected_params: Set[str]):
                    async def remote_tool_wrapper(**kwargs):
                        username_value = current_username.get()
                        transport = StreamableHttpTransport(db_agent_url)
                        async with Client(transport) as client:
                            if (
                                username_value
                                and ("username" in expected_params or "username" not in kwargs)
                            ):
                                try:
                                    return await client.call_tool(
                                        tool_name_closure,
                                        {**kwargs, "username": username_value},
                                    )
                                except Exception as exc:
                                    logging.warning(
                                        "Auto-injected username rejected by remote tool %s: %s. Retrying without.",
                                        tool_name_closure,
                                        exc,
                                    )
                            return await client.call_tool(tool_name_closure, kwargs)
                    return remote_tool_wrapper

                db_agent_tools.append(FunctionTool.from_defaults(
                    fn=make_remote_tool_wrapper(tool_info.name, param_names),
                    name=tool_info.name,
                    description=tool_info.description or f"Remote tool: {tool_info.name}",
                    fn_schema=tool_args_model
                ))
                logging.info(f"✅ Successfully added remote LlamaIndex tool: {tool_info.name}")
            logging.info(f"Loaded {len(db_agent_tools)} tools from Database Agent.")
            logging.info("✅ Successfully connected to Database Agent for tool discovery.")
        except Exception as e:
            logging.error(f"❌ Error connecting to or listing tools from Database Agent: {e}", exc_info=True)
            raise

        # === LLAMAINDEX LLM & AGENT SETUP (bez zmian) ===
        router_llm = GoogleGenAI(
            model=config.GEMINI_MODEL_NAME,
            api_key=config.GOOGLE_API_KEY,
            temperature=0.1,
        )
        logging.info(f"✅ Successfully initialized Router LLM (LlamaIndex Gemini): {config.GEMINI_MODEL_NAME}")

        # Gather local router tools as FunctionTool for llamaindex (bez zmian)
        local_router_tools = []
        local_router_tools_raw = await mcp_router_client._list_tools()
        for tool_info in local_router_tools_raw:
            try:
                parameters_schema_dict = {}
                if hasattr(tool_info, 'parameters'):
                    if isinstance(tool_info.parameters, dict):
                        parameters_schema_dict = tool_info.parameters
                    elif hasattr(tool_info.parameters, 'model_dump'):
                        parameters_schema_dict = tool_info.parameters.model_dump()

                tool_args_model = _json_schema_to_pydantic_model(tool_info.name, parameters_schema_dict)
                param_names: Set[str] = set()
                if isinstance(parameters_schema_dict, dict):
                    properties = parameters_schema_dict.get("properties", {})
                    if isinstance(properties, dict):
                        param_names = set(properties.keys())

                async def tool_wrapper(*args, _tool_name=tool_info.name, _param_names=param_names, **kwargs):
                    username_value = current_username.get()
                    if username_value and ("username" in _param_names or "username" not in kwargs):
                        kwargs = {**kwargs, "username": username_value}
                    return await mcp_router_client.call_tool(_tool_name, kwargs)

                local_router_tools.append(FunctionTool.from_defaults(
                    fn=tool_wrapper,
                    name=tool_info.name,
                    description=tool_info.description or f"Local tool: {tool_info.name}",
                    fn_schema=tool_args_model
                ))
                logging.info(f"✅ Successfully added local LlamaIndex tool: {tool_info.name}")
            except Exception as e:
                logging.error(f"❌ Error adding local tool {tool_info.name}: {e}")

        all_tools = local_router_tools + db_agent_tools
        logging.info(f"Total tools available to router LlamaIndex agent: {len(all_tools)}")


        router_llamaindex_agent = ReActAgent(
            tools=all_tools,
            llm=router_llm,
            verbose=True,
            system_prompt=(
                "Jesteś inteligentnym asystentem AI. Twoim zadaniem jest odpowiadanie na pytania użytkownika, używając dostępnych narzędzi.\n"
                "**ŚCIŚLE PRZESTRZEGAJ PONIŻSZYCH ZASAD WYBORU NARZĘDZI:**\n"
                "\n"
                "1.  **DO BEZPOŚREDNIEGO WYSZUKIWANIA TRANSKRYPTÓW (po słowach kluczowych, temacie, fragmentach treści, mówcy):**\n"
                "    -   **Zawsze użyj narzędzia `search_transcripts_semantic` lub `search_transcripts_sql`.**\n"
                "    -   `search_transcripts_semantic` jest domyślne dla ogólnych zapytań tematycznych (np. 'Który transkrypt dotyczy [temat]?', 'Znajdź fragmenty o [temat]').\n"
                "    -   `search_transcripts_sql` użyj, jeśli użytkownik prosi o wyszukiwanie po konkretnym mówcy (`speaker`) lub dokładnym tekście (`text_query`) w segmentach, ewentualnie z zakresem czasowym (`time_from`, `time_to`).\n"
                "    -   Przykład dla `search_transcripts_semantic`: `search_transcripts_semantic(query='cross-validation')`.\n"
                "    -   Przykład dla `search_transcripts_sql`: `search_transcripts_sql(speaker='Maciej', text_query='budżet')`.\n"
                "    -   **NIE UŻYWAJ `chat_with_transcripts` DO TEGO CELU!**\n"
                "\n"
                "2.  **DO ANALIZY I PODSUMOWYWANIA TREŚCI TRANSKRYPTÓW (swobodna rozmowa, głębsza analiza):**\n"
                "    -   Użyj narzędzia `chat_with_transcripts`.\n"
                "    -   To narzędzie służy do pytań wymagających zrozumienia kontekstu i generowania podsumowań (np. 'Podsumuj mi spotkanie', 'Co było powiedziane o budżecie?', 'Wyjaśnij mi X z transkryptu').\n"
                "    -   Parametr: `query` (string). Przykład: `chat_with_transcripts(query='Podsumuj spotkanie o cross-validation')`.\n"
                "\n"
                "3.  **DO WYŚWIETLANIA LISTY TRANSKRYPTÓW (wszystkie, top X, bottom X):**\n"
                "    -   `get_transcripts_with_minutes()`: Dla wszystkich transkryptów z podsumowaniami (NIE przyjmuje parametrów).\n"
                "    -   `get_top_transcripts(limit=X)`: Dla top X najnowszych transkryptów.\n"
                "    -   `get_bottom_transcripts(limit=X)`: Dla bottom X najstarszych transkryptów.\n"
                "\n"
                "4.  **DO WYKONANIA ZAPYTANIA SQL (gdy użytkownik prosi o surowe dane z bazy):**\n"
                "    -   Użyj `execute_sql_query(query='TWOJE ZAPYTANIE SQL')` (tylko SELECT).\n"
                "\n"
                "**WAŻNE:** narzędzia bazodanowe działają w kontekście zalogowanego użytkownika. Parametr `username` jest dołączany automatycznie – nie próbuj go pomijać ani nadpisywać inną wartością.\n"
                "\n"
                "Zawsze dostarczaj zwięzłą i bezpośrednią odpowiedź użytkownikowi, bazując na wynikach narzędzi. Jeśli narzędzie zwróci pusty wynik, poinformuj o tym użytkownika."
            )
        )


        logging.info("✅ Router LlamaIndex ReActAgent initialized with all tools.")

        logging.info("🎉 Minimal FastMCP Server startup complete!")
        yield
    finally:
        logging.info("Shutting down Minimal FastMCP Server...\n")
        # === USUNIĘTO: Nie ma już globalnego db_agent_client do zamykania ===
        # cleanup_temp_dir()
        logging.info("Minimal FastMCP Server shutdown complete.\n")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=".")

app.mount("/mcp_api", mcp_router_client)
app.mount("/static", StaticFiles(directory="static"), name="static")

# === NARZĘDZIA FASTMCP (hostowane przez ten serwer - Router) ===

@mcp_router_client.tool()
async def process_audio_and_summarize(
    file_path: str,
    file_job_id: str,
    output_directory: str,
    transcription_provider: str = config.DEFAULT_TRANSCRIPTION_PROVIDER,
    transcription_model: Optional[str] = None,
    output_name: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    chunk_duration: int = 240,
    chunk_overlap: int = 0,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info(f"Calling internal processing logic for {file_path}")
    return {"status": "error", "message": "process_audio_and_summarize tool not fully implemented yet."}

@mcp_router_client.tool("list_processed_files")
async def list_processed_files_tool(username: str) -> Dict[str, Any]:
    logging.info("Attempting to retrieve all processed files summary from DB via Database Agent (remote call).")
    try:
        # === ZMIANA: Użyj nowego połączenia dla każdego wywołania ===
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            raw_tool_response = await client.call_tool("get_all_processed_files", {"username": username})
        return_data = await _extract_tool_output(raw_tool_response)
        return {"text_summary": return_data}
    except Exception as e:
        logging.error(f"Error listing processed files from Database Agent: {e}", exc_info=True)
        return {"error": f"Wystąpił błąd podczas pobierania listy przetworzonych plików: {str(e)}"}

@mcp_router_client.tool("list_qdrant_collections", description="Wyświetla listę dostępnych kolekcji Qdrant dla danego użytkownika, opcjonalnie z limitem liczby kolekcji.")
async def list_qdrant_collections_tool(username: str, limit: Optional[int] = None) -> str:
    logging.info(f"Attempting to list Qdrant collections with limit: {limit}")
    try:
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                logging.warning("Invalid limit value '%s' provided; ignoring.", limit)
                limit = None
        collections_set: Set[str] = set()
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            processed_response = await client.call_tool("get_all_processed_files", {"username": username})
            processed = await _extract_tool_output_dict(processed_response)
            if isinstance(processed, list):
                for entry in processed:
                    file_id = entry.get("id")
                    if file_id is None:
                        continue
                    meeting_data = await client.call_tool("get_meeting_data_by_file_id", {"file_id": file_id, "username": username})
                    meeting_dict = await _extract_tool_output_dict(meeting_data)
                    collection_name = meeting_dict.get("qdrant_collection_name") if isinstance(meeting_dict, dict) else None
                    if collection_name:
                        collections_set.add(collection_name)
        collections = list(collections_set)
        if limit:
            collections = collections[: limit]
        if not collections:
            return "No Qdrant collections found."
        return "Dostępne kolekcje Qdrant:\n" + "\n".join(collections)
    except Exception as e:
        logging.error(f"Error listing Qdrant collections: {e}", exc_info=True)
        return f"Error listing Qdrant collections: {e}"

@mcp_router_client.tool("search_qdrant_data", description="Wyszukuje dane w Qdrant na podstawie zapytania tekstowego dla konkretnego użytkownika. Zwraca fragmenty treści z powiązanych plików.")
async def search_qdrant_data_tool(query: str, username: str) -> List[Dict[str, Any]]:
    logging.info(f"Attempting to search Qdrant data with query: {query} for user {username}")
    try:
        qdrant_results: List[Any] = await search_all_collections(query)
        if not qdrant_results:
            return [{"message": "No results found for your query."}]
        formatted_results = []
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            for result in qdrant_results:
                file_id = "N/A"
                filename = "N/A"
                if hasattr(result, 'file_path'): # Usunięto warunek db_agent_client
                    file_record_response = await client.call_tool(
                        "get_file_record_by_filepath",
                        {"filepath": result.file_path, "username": username},
                    )
                    if hasattr(file_record_response, 'structured_content') and file_record_response.structured_content:
                        file_record = file_record_response.structured_content
                        if file_record and not file_record.get('error'):
                            file_id = file_record.get('id', 'N/A')
                            filename = file_record.get('filename', 'N/A')
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
                formatted_results.append({
                    "file_id": file_id,
                    "filename": filename,
                    "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "full_content": result.content
                })
        return formatted_results
    except Exception as e:
        logging.error(f"Error searching Qdrant data: {e}", exc_info=True)
        return [{"error": f"Error searching Qdrant data: {e}"}]

@mcp_router_client.tool("list_directory_contents", description="Wyświetla zawartość podanego katalogu, w tym podkatalogi i pliki. Opcjonalnie może być rekurencyjne.")
async def list_directory_contents_tool(directory_path: str) -> Dict[str, str]:
    logging.info(f"Attempting to list contents of directory: {directory_path}")
    target_path = Path(directory_path)
    if not target_path.is_dir():
        return {"text_summary": f"Błąd: Katalog '{directory_path}' nie istnieje lub nie jest katalogiem."}
    markdown_output = f"## Zawartość katalogu '{directory_path}':\n\n"
    def _list_directory_recursive(current_path, indent_level=0):
        nonlocal markdown_output
        indent_prefix = "    " * indent_level
        try:
            for item in sorted(os.listdir(current_path)):
                item_path = current_path / item
                if item_path.is_dir():
                    markdown_output += f"{indent_prefix}- **{item}/**\n"
                    _list_directory_recursive(item_path, indent_level + 1)
                else:
                    markdown_output += f"{indent_prefix}- {item}\n"
        except Exception as e:
            logging.error(f"Error listing directory {current_path}: {e}")
    _list_directory_recursive(target_path, 0)
    return {"text_summary": markdown_output}

@mcp_router_client.tool("generate_summary_for_processed_file", description="Generuje podsumowanie spotkania na podstawie wcześniej przetworzonego pliku zidentyfikowanego po jego oryginalnej nazwie dla danego użytkownika.")
async def generate_summary_for_processed_file_tool(filename: str, username: str, custom_prompt: Optional[str] = None) -> str:
    logging.info(f"Attempting to generate summary for file: {filename} (user: {username})")
    try:
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            file_record_response = await client.call_tool("get_file_record_by_filename", {"filename": filename, "username": username})
            file_record = await _extract_tool_output_dict(file_record_response)
            if not file_record or file_record.get('error'):
                return f"Przepraszam, nie znaleziono pliku '{filename}' w bazie danych przetworzonych plików."
            file_id = file_record['id']
            meeting_data_record_response = await client.call_tool("get_meeting_data_by_file_id", {"file_id": file_id, "username": username})
            meeting_data_record = await _extract_tool_output_dict(meeting_data_record_response)
            if not meeting_data_record or meeting_data_record.get('error'):
                return f"Przepraszam, nie znaleziono danych spotkania dla pliku '{filename}' (ID: {file_id}). Być może plik nie został jeszcze w pełni przetworzony lub brakuje powiązanych danych."
            transcript_id = meeting_data_record.get('transcript_id')
            if not transcript_id:
                return f"Brak dostępnej transkrypcji dla pliku '{filename}' (brak transcript_id w danych spotkania)."
            transcript_record_response = await client.call_tool("get_transcript_by_id", {"id": transcript_id, "username": username})
            transcript_record = await _extract_tool_output_dict(transcript_record_response)
            full_transcript_text = transcript_record.get('content') if transcript_record else None
            if not full_transcript_text:
                return f"Nie można pobrać treści transkrypcji dla ID: {transcript_id}."
            logging.info(f"Retrieved full transcript for summary generation (length: {len(full_transcript_text)}).")
            summary_response = await generate_minutes_of_meeting(full_transcript_text, custom_prompt=custom_prompt)
            if summary_response:
                logging.info(f"Summary generated for file {filename}.")
                existing_minutes_id = meeting_data_record.get('minutes_id')
                meeting_date_str = meeting_data_record.get('meeting_date')
                if not meeting_date_str:
                    processed_file_response = await client.call_tool("get_file_record_by_id", {"id": file_id, "username": username})
                    processed_file = await _extract_tool_output_dict(processed_file_response)
                    if processed_file and not processed_file.get('error') and processed_file.get('processed_at'):
                        meeting_date_str = processed_file['processed_at'].split(' ')[0]
                    else:
                        meeting_date_str = datetime.now().strftime("%Y-%m-%d")
                add_minutes_args = {
                    "summary_text": json.dumps(summary_response, ensure_ascii=False),
                    "minutes_path": f"{meeting_data_record.get('qdrant_collection_name', 'unknown')}_summary.txt",
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "llm_model": config.GEMINI_MODEL_NAME,
                    "embeddings_model": config.EMBEDDING_MODEL_NAME,
                    "chunking_method": f"segmenty {config.CHUNK_DURATION}s z nałożeniem {config.CHUNK_OVERLAP}s",
                    "transcription_model": config.WHISPER_MODEL_SIZE,
                    "diarization_model": config.PYANNOTE_PIPELINE,
                    "meeting_date_str": meeting_date_str,
                    "existing_minutes_id": existing_minutes_id,
                    "username": username,
                }
                add_minutes_response = await client.call_tool("add_meeting_minutes", add_minutes_args)
                minutes_id_info = await _extract_tool_output_dict(add_minutes_response)
                minutes_id = minutes_id_info.get('minutes_id') if minutes_id_info else None
                if existing_minutes_id is None and minutes_id:
                    await client.call_tool("update_meeting_data_minutes_id", {"file_id": file_id, "minutes_id": minutes_id, "username": username})
                formatted_summary_parts = []
                for k, v in summary_response.items():
                    formatted_summary_parts.append(f"**{k.replace('_', ' ')}:**\n{v}")
                return f"Podsumowanie dla pliku '{filename}' zostało wygenerowane i zapisane:\n\n" + "\n\n".join(formatted_summary_parts)
            else:
                return f"Nie udało się wygenerować podsumowania dla pliku '{filename}' (odpowiedź LLM była pusta)."
    except Exception as e:
        logging.error(f"Error in MCP tool 'generate_summary_for_processed_file' for file '{filename}': {e}", exc_info=True)
        return f"Przepraszam, wystąpił błąd podczas generowania podsumowania dla pliku '{filename}': {e}"

@mcp_router_client.tool("get_transcript_content_for_file", description="Pobiera pełną treść transkrypcji dla danego ID pliku (tylko transkrypty użytkownika).")
async def get_transcript_content_for_file_tool(file_id: int, username: str) -> str:
    logging.info(f"Attempting to retrieve transcript content for file_id: {file_id} (user: {username})")
    try:
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            meeting_data_record_response = await client.call_tool("get_meeting_data_by_file_id", {"file_id": file_id, "username": username})
            meeting_data_record = await _extract_tool_output_dict(meeting_data_record_response)
            if not meeting_data_record or meeting_data_record.get('error'):
                return f"Przepraszam, nie znaleziono danych spotkania dla pliku o ID: {file_id}. Być może plik nie został jeszcze w pełni przetworzony lub brakuje powiązanych danych."
            transcript_id = meeting_data_record.get('transcript_id')
            if not transcript_id:
                return f"Brak dostępnej transkrypcji dla pliku o ID: {file_id} (brak transcript_id w danych spotkania)."
            transcript_record_response = await client.call_tool("get_transcript_by_id", {"id": transcript_id, "username": username})
            transcript_record = await _extract_tool_output_dict(transcript_record_response)
            full_transcript_text = transcript_record.get('content') if transcript_record else None
            if not full_transcript_text:
                return f"Treść transkrypcji dla ID: {transcript_id} jest pusta."
            return f"Pełna treść transkrypcji dla pliku o ID {file_id}:\n\n{full_transcript_text}"
    except Exception as e:
        logging.error(f"Error retrieving transcript content for file_id {file_id}: {e}", exc_info=True)
        return f"Przepraszam, wystąpił błąd podczas pobierania treści transkrypcji dla pliku o ID {file_id}: {e}"

@mcp_router_client.tool("list_transcripts", description="Wyświetla wszystkie transkrypty z podsumowaniami spotkań dla danego użytkownika. Używa get_transcripts_with_minutes z Database Agent.")
async def list_transcripts_tool(username: str) -> Dict[str, Any]:
    logging.info("Attempting to retrieve transcripts with minutes from Database Agent.")
    try:
        transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
        async with Client(transport) as client:
            raw_tool_response = await client.call_tool("get_transcripts_with_minutes", {"username": username})
        return_data = await _extract_tool_output(raw_tool_response)
        return {"text_summary": return_data}
    except Exception as e:
        logging.error(f"Error listing transcripts from Database Agent: {e}", exc_info=True)
        return {"error": f"Wystąpił błąd podczas pobierania transkryptów: {str(e)}"}

async def _extract_tool_output(tool_result: Any) -> str:
    if hasattr(tool_result, 'structured_content') and tool_result.structured_content is not None:
        # Tutaj sprawdzamy strukturę JSONRPCResponse z 'structuredContent' i 'result'
        if 'result' in tool_result.structured_content and isinstance(tool_result.structured_content['result'], list):
            actual_result_list = tool_result.structured_content['result']

            # Specjalna obsługa dla get_all_processed_files
            if all(isinstance(item, dict) and 'filename' in item and 'filehash' in item for item in actual_result_list):
                output_lines = ["Oto lista wszystkich przetworzonych plików:"]
                if not actual_result_list:
                    output_lines.append("Brak przetworzonych plików do wyświetlenia.")
                else:
                    for item in actual_result_list:
                        display_filename = item.get('filename', 'N/A')
                        output_lines.append(f"* {display_filename}")
                return "\n".join(output_lines)
            
            # Specjalna obsługa dla listy transkryptów z podsumowaniami
            if all(isinstance(item, dict) and 'transcript_id' in item and 'summary_text' in item and 'minutes_generated_at' in item for item in actual_result_list):
                output_lines = ["Oto lista przetworzonych transkryptów:"]
                if not actual_result_list:
                    output_lines.append("Nie znaleziono żadnych przetworzonych transkryptów.")
                else:
                    for item in actual_result_list:
                        transcript_id = item.get('transcript_id', 'N/A')
                        summary_text = item.get('summary_text', 'Brak podsumowania')
                        minutes_generated_at = item.get('minutes_generated_at', 'Brak daty')
                        
                        output_lines.append(f"* ID Transkryptu: {transcript_id}")
                        output_lines.append(f"  Podsumowanie: {summary_text}")
                        output_lines.append(f"  Data generacji podsumowania: {minutes_generated_at}\n")
                return "\n".join(output_lines)
            
            # Ogólna obsługa listy słowników (np. wyniki zapytań SQL)
            if all(isinstance(item, dict) for item in actual_result_list):
                if actual_result_list and len(set(frozenset(d.keys()) for d in actual_result_list)) == 1:
                    headers = list(actual_result_list[0].keys())
                    markdown_table = "| " + " | ".join(headers) + " |\n"
                    markdown_table += "|---" * len(headers) + "|\n"
                    for row in actual_result_list:
                        values = [str(row.get(h, '')) for h in headers]
                        markdown_table += "| " + " | ".join(values) + " |\n"
                    return markdown_table
                else:
                    return json.dumps(actual_result_list, indent=2)
            else:
                if actual_result_list:
                    list_items_markdown = [f"{item}" for item in actual_result_list]
                    return "\n".join(list_items_markdown)
                else:
                    return "Brak zawartości."
        elif isinstance(tool_result.structured_content, dict):
            return json.dumps(tool_result.structured_content, indent=2)
        else:
            return str(tool_result.structured_content)
    elif hasattr(tool_result, 'data') and tool_result.data is not None:
        return await _extract_tool_output(tool_result.data)
    elif hasattr(tool_result, 'content') and tool_result.content:
        return "\n".join([c.text for c in tool_result.content if hasattr(c, 'text')])
    elif hasattr(tool_result, 'is_error') and tool_result.is_error:
        return f"Error from tool: {tool_result.content[0].text if hasattr(tool_result, 'content') and tool_result.content else 'Unknown error'}"
    if isinstance(tool_result, dict):
        if 'text_summary' in tool_result and isinstance(tool_result['text_summary'], str):
            return tool_result['text_summary']
        if 'error' in tool_result and isinstance(tool_result['error'], str):
            return tool_result['error']
        if 'response' in tool_result:
            return tool_result['response']
        if 'result' in tool_result and isinstance(tool_result['result'], (list, dict)):
            return await _extract_tool_output(tool_result['result'])
        else:
            return json.dumps(tool_result, indent=2)
    elif isinstance(tool_result, list):
        # Poprzednia obsługa wyników wyszukiwania Qdrant (jeśli nadal potrzebna)
        if all(isinstance(item, dict) and 'file_id' in item and 'filename' in item for item in tool_result):
            markdown_output = "### Wyniki wyszukiwania w Qdrant:\n\n"
            for item in tool_result:
                markdown_output += f"- **ID pliku:** `{item.get('file_id', 'N/A')}`\n"
                markdown_output += f"  **Nazwa pliku:** `{item.get('filename', 'N/A')}`\n"
                markdown_output += f"  **Fragment treści:** \"{item.get('content_preview', 'Brak podglądu.')}\"\n\n"
            return markdown_output
        elif all(isinstance(item, dict) for item in tool_result):
            if tool_result and len(set(frozenset(d.keys()) for d in tool_result)) == 1:
                headers = list(tool_result[0].keys())
                markdown_table = "| " + " | ".join(headers) + " |\n"
                markdown_table += "|---" * len(headers) + "|\n"
                for row in tool_result:
                    values = [str(row.get(h, '')) for h in headers]
                    markdown_table += "| " + " | ".join(values) + " |\n"
                return markdown_table
            else:
                return json.dumps(tool_result, indent=2)
        else:
            if tool_result:
                list_items_markdown = [f"{item}" for item in tool_result]
                return "\n".join(list_items_markdown)
            else:
                return "Brak zawartości."
    else:
        return str(tool_result)

async def _extract_tool_output_dict(raw_tool_response: Any) -> Optional[Dict[str, Any]]:
    if hasattr(raw_tool_response, 'structured_content') and raw_tool_response.structured_content:
        return raw_tool_response.structured_content.get('result')
    return None

def _json_schema_to_pydantic_model(tool_name: str, parameters_schema: Dict[str, Any]) -> Type[BaseModel]:
    if not parameters_schema:
        # Zwracamy dynamicznie utworzoną, pustą podklasę BaseModel
        return create_model(f"{tool_name.replace('-', '_').replace('.', '_').replace('/', '_')}ArgsEmpty", __base__=BaseModel)
    properties = parameters_schema.get('properties', {})
    required_fields = set(parameters_schema.get('required', []))
    fields_for_pydantic_model = {}
    for prop_name, prop_schema in properties.items():
        py_type: Type = Any
        if prop_schema.get('type') == 'string':
            py_type = str
        elif prop_schema.get('type') == 'number':
            py_type = float
        elif prop_schema.get('type') == 'integer':
            py_type = int
        elif prop_schema.get('type') == 'boolean':
            py_type = bool
        elif prop_schema.get('type') == 'array':
            items_schema = prop_schema.get('items', {})
            item_type: Type = Any
            if items_schema.get('type') == 'string': item_type = str
            elif items_schema.get('type') == 'number': item_type = float
            elif items_schema.get('type') == 'integer': item_type = int
            py_type = List[item_type]
        elif prop_schema.get('type') == 'object':
            py_type = Dict[str, Any]
        elif '$ref' in prop_schema:
            logging.warning(f"Unhandled $ref in schema for {tool_name}.{prop_name}. Defaulting to Any.")
            py_type = Any
        if prop_name in required_fields:
            fields_for_pydantic_model[prop_name] = (py_type, Field(..., description=prop_schema.get('description', '')))
        else:
            fields_for_pydantic_model[prop_name] = (Optional[py_type], Field(None, description=prop_schema.get('description', '')))
    DynamicArgsModel = create_model(
        f"{tool_name.replace('-', '_').replace('.', '_').replace('/', '_')}Args",
        **fields_for_pydantic_model,
        __base__=BaseModel
    )
    return DynamicArgsModel

# === ENDPOINTY FASTAPI ===
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("chat_interface.html")

@app.get("/chat_ui", response_class=HTMLResponse)
async def serve_chat_interface(request: Request):
    return templates.TemplateResponse(request, "chat_interface.html")

@app.post("/chat/query", response_model=ChatResponse)
async def chat_query_endpoint(chat_query: ChatQuery, request: Request):
    global router_llamaindex_agent
    logging.info(f"Received chat query: '{chat_query.query}' at /chat/query endpoint.")
    if router_llamaindex_agent is None:
        raise HTTPException(status_code=503, detail="Agent Router not initialized. Please try again later.")
    context_token = None
    try:
        username = _resolve_username(request, chat_query.username)
        logging.info(
            "Chat request authenticated for user: %s (collection=%s)",
            username,
            chat_query.collection_name or "<none>",
        )
        context_token = current_username.set(username)
        # Wzmocnij zapytanie użytkownika o nazwę kolekcji, jeśli została podana
        full_query = chat_query.query
        if chat_query.collection_name:
            full_query = f"Użyj kolekcji Qdrant: {chat_query.collection_name}. " + full_query

        agent_response = await router_llamaindex_agent.run(user_msg=full_query)
        logging.info(f"DEBUG: agent_response.response dir: {dir(agent_response.response)}")
        logging.info(f"DEBUG: Output from LlamaIndex agent: {agent_response.response.content}")
        return ChatResponse(response=agent_response.response.content)
    except WorkflowRuntimeError as e:
        logging.error(f"Error in chat_query_endpoint: {e}", exc_info=True)
        # Wychwyć bardziej szczegółowe informacje o błędzie, jeśli są dostępne z WorkflowRuntimeError
        error_message = f"Przepraszam, wystąpił błąd podczas przetwarzania Twojego zapytania: {e}"
        if isinstance(e, WorkflowRuntimeError) and "list index out of range" in str(e):
            error_message = "Przepraszam, agent AI napotkał problem podczas generowania odpowiedzi (brak treści). Może to być spowodowane filtrami bezpieczeństwa, problemami z modelem lub niejasnym zapytaniem. Spróbuj zadać pytanie ponownie lub przeformułować je."
        elif isinstance(e, WorkflowRuntimeError):
            error_message = f"Przepraszam, agent AI napotkał wewnętrzny błąd: {e}"
        
        return ChatResponse(response=error_message)
    finally:
        if context_token is not None:
            current_username.reset(context_token)

@app.post("/upload-audio")
async def upload_audio(
    audio_file: UploadFile = File(...),
    user_query: Optional[str] = Form(None)
):
    job_id = str(uuid.uuid4())
    temp_dir = Path(config.UPLOAD_DIR) / job_id
    os.makedirs(temp_dir, exist_ok=True)
    file_location = temp_dir / audio_file.filename
    original_filename = audio_file.filename

    logging.info(f"Received file upload: {original_filename} (job_id: {job_id})")

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logging.info(f"File saved to: {file_location}")

        processed_audio_path = file_location
        mime_type, _ = mimetypes.guess_type(file_location)
        if mime_type and mime_type.startswith('video/'):
            logging.info(f"Detected video file: {original_filename}. Converting to WAV...")
            wav_filename = f"{file_location.stem}.wav"
            converted_wav_path = temp_dir / wav_filename
            if not convert_to_wav(str(file_location), str(converted_wav_path)):
                raise Exception("Video to audio conversion failed.")
            processed_audio_path = converted_wav_path
            logging.info(f"Video converted to audio: {processed_audio_path}")
        elif mime_type and not mime_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail=f"Nieobsługiwany typ pliku: {mime_type}. Proszę przesłać plik audio lub wideo.")

        logging.info(f"Calling process_audio_and_summarize tool for job {job_id} on Audio Agent...")
        final_audio_processing_result = None
        tool_args = {
            "file_path": str(processed_audio_path),
            "file_job_id": job_id,
            "output_directory": str(config.OUTPUT_DIR_API_TRANSCRIPTS),
            "original_filename": original_filename,
            "custom_prompt": user_query
        }
        final_audio_processing_result = await process_audio(
            audio_path=processed_audio_path,
            file_job_id=job_id,
            transcription_model=None,
            transcription_provider=config.DEFAULT_TRANSCRIPTION_PROVIDER,
        )
        if final_audio_processing_result[0]:
            transcription_segments = final_audio_processing_result[0]
            transcription_filename = final_audio_processing_result[1]
            transcript_id = final_audio_processing_result[2]
            collection_name_for_file = await ingest_transcription(
                transcription_segments,
                processed_audio_path,
                job_id,
                transcript_id,
                config.CHUNK_DURATION,
                config.CHUNK_OVERLAP
            )
            provider_metadata = build_transcription_metadata(config.DEFAULT_TRANSCRIPTION_PROVIDER, None)
            provider_id = provider_metadata.get("transcription_provider", "")
            provider_metadata["diarization_model"] = (
                "Modal remote diarization" if provider_id in {"gemini", "assemblyai"} else config.PYANNOTE_PIPELINE
            )
            minutes_content_response, minutes_filename, minutes_db_id = await generate_and_save_minutes(
                collection_name=collection_name_for_file,
                original_source_path=processed_audio_path,
                custom_prompt=user_query,
                output_name=original_filename,
                metadata=provider_metadata,
            )
            file_hash = await run_in_threadpool(agent_db.compute_file_hash, processed_audio_path)
            file_id = await run_in_threadpool(agent_db.add_file, original_filename, str(processed_audio_path), file_hash, "processing", "")
            await run_in_threadpool(agent_db.add_meeting_data, file_id, transcript_id, minutes_db_id, get_relevant_date_for_file(processed_audio_path), collection_name_for_file)
            await run_in_threadpool(agent_db.update_file_status, file_hash, "completed", "Processing successful")
            return JSONResponse(content={
                "status": "success",
                "message": "Audio file processed and summarized successfully.",
                "result": {
                    "transcription_filename": transcription_filename,
                    "minutes_filename": minutes_filename,
                    "collection_name": collection_name_for_file,
                    "minutes_content": minutes_content_response.model_dump() if minutes_content_response else None
                }
            })
        else:
            raise HTTPException(status_code=500, detail="Audio processing failed or produced no segments.")
    except Exception as e:
        logging.error(f"Failed to upload or start processing file for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas przesyłania lub przetwarzania pliku: {e}")

@app.post("/mcp_tool")
async def mcp_tool(request: Request):
    """
    Uniwersalny endpoint do wywoływania dowolnego narzędzia FastMCP na agencie bazodanowym.
    Przykład body:
    {
      "tool_name": "chat_with_transcripts",
      "params": {
        "query": "O czym była rozmowa?",
        "collection_name": "spotkanie1"
      }
    }
    """
    data = await request.json()
    tool_name = data.get("tool_name")
    params = data.get("params", {})

    transport = StreamableHttpTransport("http://127.0.0.1:8002/mcp/")
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, {"params": params})
        # Jeśli odpowiedź ma .dict(), zwróć jako JSON, w innym wypadku zwróć surowo
        if hasattr(result, "dict"):
            return JSONResponse(result.dict())
        return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server with Uvicorn (Minimal MCP Server)...")
    uvicorn.run("minimal_mcp_server:app", host="0.0.0.0", port=8000, reload=True)