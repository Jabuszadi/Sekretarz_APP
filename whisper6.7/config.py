# config.py
import os
import logging
import json
from pathlib import Path
import torch
from dotenv import load_dotenv
from transformers.utils import is_flash_attn_2_available

# --- Load Environment Variables ---
load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")

# --- Configure Logging ---
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # <--- Upewnij się, że poziom to DEBUG - USUNIĘTO

# ==================================================
#             API Keys & Endpoints
# ==================================================
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Keep even if potentially optional for some setups
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Needed for Gemini/Google AI features
if GOOGLE_API_KEY:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_DB_URL = (
    os.getenv("SUPABASE_DB_URL")
    or os.getenv("SUPABASE_DATABASE_URL")
    or os.getenv("DATABASE_URL")
)
SUPABASE_SCHEMA = os.getenv("SUPABASE_SCHEMA", "public")
USE_SUPABASE = _env_flag(
    "USE_SUPABASE",
    bool(SUPABASE_URL and SUPABASE_API_KEY and SUPABASE_DB_URL),
)

# Redis configuration for Dramatiq queue
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# It's strongly recommended to load sensitive tokens like HF_TOKEN from environment variables
# Example: HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN", "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") # Default fallback if not in .env
USE_OPENAI_WHISPER_API = _env_flag("USE_OPENAI_WHISPER_API", True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or None
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or None
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "gpt-4o-mini-transcribe")
OPENAI_WHISPER_RESPONSE_FORMAT = os.getenv("OPENAI_WHISPER_RESPONSE_FORMAT", "verbose_json")
OPENAI_WHISPER_LANGUAGE = os.getenv("OPENAI_WHISPER_LANGUAGE")
GEMINI_TRANSCRIBE_API_KEY = (
    os.getenv("GOOGLE_TRANSCRIBE_API_KEY")
    or os.getenv("GOOGLE_TRANSRIBE_API_KEY")
    or os.getenv("GEMINI_TRANSCRIBE_API_KEY")
    or os.getenv("GOOGLE_TRANSCRIBE_KEY")
)
_openai_whisper_temperature_raw = os.getenv("OPENAI_WHISPER_TEMPERATURE")
try:
    OPENAI_WHISPER_TEMPERATURE = (
        float(_openai_whisper_temperature_raw)
        if _openai_whisper_temperature_raw is not None
        else None
    )
except ValueError:
    logging.warning(
        "Ignoring invalid OPENAI_WHISPER_TEMPERATURE value '%s'. Expected a float.",
        _openai_whisper_temperature_raw,
    )
    OPENAI_WHISPER_TEMPERATURE = None

DEFAULT_TRANSCRIPTION_PROVIDER = os.getenv("DEFAULT_TRANSCRIPTION_PROVIDER", "gpt").strip().lower() or "gpt"
DEFAULT_GPT_TRANSCRIPTION_MODEL = os.getenv("DEFAULT_GPT_TRANSCRIPTION_MODEL", "normal").strip().lower() or "normal"
DEFAULT_GEMINI_TRANSCRIPTION_MODEL = os.getenv("DEFAULT_GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
ASSEMBLYAI_ALLOWED_MODELS = {"best", "nano", "slam-1", "universal"}

_default_assemblyai_model = os.getenv("DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL", "best").strip().lower() or "best"
if _default_assemblyai_model not in ASSEMBLYAI_ALLOWED_MODELS:
    logging.warning(
        "Unsupported DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL '%s'. Falling back to 'best'.",
        _default_assemblyai_model,
    )
    _default_assemblyai_model = "best"
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL = _default_assemblyai_model
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

_DEFAULT_OPENAI_WHISPER_MODEL_MAP = {
    "default": None,
    "mini": os.getenv("GPT_TRANSCRIBE_MODEL_MINI", "gpt-4o-mini-transcribe"),
    "normal": os.getenv("GPT_TRANSCRIBE_MODEL_NORMAL", "gpt-4o-transcribe"),
}
_openai_whisper_model_map_raw = os.getenv("OPENAI_WHISPER_MODEL_MAP")
if _openai_whisper_model_map_raw:
    try:
        _parsed_model_map = json.loads(_openai_whisper_model_map_raw)
        if isinstance(_parsed_model_map, dict):
            _normalized_user_map = {}
            for key, value in _parsed_model_map.items():
                if not isinstance(key, str):
                    logging.warning(
                        "Skipping non-string key in OPENAI_WHISPER_MODEL_MAP: %r", key
                    )
                    continue
                if value is not None and not isinstance(value, str):
                    logging.warning(
                        "Skipping invalid mapping for '%s' in OPENAI_WHISPER_MODEL_MAP. Expected string or null, got %r.",
                        key,
                        value,
                    )
                    continue
                normalized_key = key.strip().lower()
                if not normalized_key:
                    continue
                _normalized_user_map[normalized_key] = value
            OPENAI_WHISPER_MODEL_MAP = {
                **_DEFAULT_OPENAI_WHISPER_MODEL_MAP,
                **_normalized_user_map,
            }
        else:
            logging.warning(
                "OPENAI_WHISPER_MODEL_MAP must be a JSON object. Using default model mapping."
            )
            OPENAI_WHISPER_MODEL_MAP = dict(_DEFAULT_OPENAI_WHISPER_MODEL_MAP)
    except json.JSONDecodeError as map_error:
        logging.warning(
            "Could not parse OPENAI_WHISPER_MODEL_MAP JSON (%s). Using default mapping.",
            map_error,
        )
        OPENAI_WHISPER_MODEL_MAP = dict(_DEFAULT_OPENAI_WHISPER_MODEL_MAP)
else:
    OPENAI_WHISPER_MODEL_MAP = dict(_DEFAULT_OPENAI_WHISPER_MODEL_MAP)

WHISPER_LEGACY_ALIAS_REDIRECTS = {
    "tiny": "mini",
    "base": "mini",
    "small": "mini",
    "medium": "normal",
    "large": "normal",
    "large-v1": "normal",
    "large-v2": "normal",
    "large-v3": "normal",
}

_default_gpt_provider_entry = {
    "provider_id": "gpt",
    "label": os.getenv("GPT_TRANSCRIPTION_LABEL", "OpenAI Whisper"),
    "default_model": DEFAULT_GPT_TRANSCRIPTION_MODEL,
    "models": [
        {
            "id": "mini",
            "label": f"Mini ({OPENAI_WHISPER_MODEL_MAP.get('mini', 'gpt-4o-mini-transcribe')})",
            "metadata": {
                "openai_target": OPENAI_WHISPER_MODEL_MAP.get("mini", "gpt-4o-mini-transcribe"),
            },
        },
        {
            "id": "normal",
            "label": f"Normal ({OPENAI_WHISPER_MODEL_MAP.get('normal', 'gpt-4o-transcribe')})",
            "metadata": {
                "openai_target": OPENAI_WHISPER_MODEL_MAP.get("normal", "gpt-4o-transcribe"),
            },
        },
    ],
}

_gemini_models_override = os.getenv("GEMINI_TRANSCRIPTION_MODELS")
if _gemini_models_override:
    try:
        parsed_models = json.loads(_gemini_models_override)
        if isinstance(parsed_models, list) and parsed_models:
            _gemini_model_entries = []
            for entry in parsed_models:
                if isinstance(entry, str):
                    cleaned = entry.strip()
                    if cleaned:
                        _gemini_model_entries.append({"id": cleaned, "label": cleaned})
                elif isinstance(entry, dict):
                    model_id = entry.get("id")
                    if not model_id:
                        continue
                    label = entry.get("label") or str(model_id)
                    _gemini_model_entries.append({"id": str(model_id), "label": label})
                else:
                    logging.warning("Skipping unsupported entry in GEMINI_TRANSCRIPTION_MODELS: %r", entry)
            if not _gemini_model_entries:
                raise ValueError("Empty list after parsing GEMINI_TRANSCRIPTION_MODELS.")
        else:
            raise ValueError("Parsed GEMINI_TRANSCRIPTION_MODELS is not a non-empty list.")
    except (json.JSONDecodeError, ValueError) as gemini_error:
        logging.warning("Could not parse GEMINI_TRANSCRIPTION_MODELS (%s). Using defaults.", gemini_error)
        _gemini_model_entries = [
            {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
        ]
else:
    _gemini_model_entries = [
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
    ]

_assemblyai_models_override = os.getenv("ASSEMBLYAI_TRANSCRIPTION_MODELS")
if _assemblyai_models_override:
    try:
        parsed_models = json.loads(_assemblyai_models_override)
        if isinstance(parsed_models, list) and parsed_models:
            _assemblyai_model_entries = []
            for entry in parsed_models:
                model_id = None
                label = None
                if isinstance(entry, str):
                    model_id = entry.strip().lower()
                    label = entry.strip()
                elif isinstance(entry, dict):
                    raw_id = entry.get("id")
                    if raw_id:
                        model_id = str(raw_id).strip().lower()
                        label = entry.get("label") or str(raw_id)
                else:
                    logging.warning("Skipping unsupported entry in ASSEMBLYAI_TRANSCRIPTION_MODELS: %r", entry)
                    continue
                if not model_id:
                    continue
                if model_id not in ASSEMBLYAI_ALLOWED_MODELS:
                    logging.warning(
                        "Skipping unsupported AssemblyAI model '%s'. Allowed values: %s",
                        model_id,
                        ", ".join(sorted(ASSEMBLYAI_ALLOWED_MODELS)),
                    )
                    continue
                _assemblyai_model_entries.append({"id": model_id, "label": label or model_id})
            if not _assemblyai_model_entries:
                raise ValueError("Empty list after parsing ASSEMBLYAI_TRANSCRIPTION_MODELS.")
        else:
            raise ValueError("Parsed ASSEMBLYAI_TRANSCRIPTION_MODELS is not a non-empty list.")
    except (json.JSONDecodeError, ValueError) as assemblyai_error:
        logging.warning("Could not parse ASSEMBLYAI_TRANSCRIPTION_MODELS (%s). Using defaults.", assemblyai_error)
        _assemblyai_model_entries = [
            {"id": "best", "label": "Best"},
            {"id": "nano", "label": "Nano"},
            {"id": "slam-1", "label": "SLAM-1"},
            {"id": "universal", "label": "Universal"},
        ]
else:
    _assemblyai_model_entries = [
        {"id": "best", "label": "Best"},
        {"id": "nano", "label": "Nano"},
        {"id": "slam-1", "label": "SLAM-1"},
        {"id": "universal", "label": "Universal"},
    ]

_default_gemini_provider_entry = {
    "provider_id": "gemini",
    "label": os.getenv("GEMINI_TRANSCRIPTION_LABEL", "Gemini"),
    "default_model": DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
    "models": _gemini_model_entries,
}

_default_assemblyai_provider_entry = {
    "provider_id": "assemblyai",
    "label": os.getenv("ASSEMBLYAI_TRANSCRIPTION_LABEL", "AssemblyAI"),
    "default_model": DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL,
    "models": _assemblyai_model_entries,
}

_transcription_provider_catalog_override = os.getenv("TRANSCRIPTION_PROVIDER_CATALOG")
if _transcription_provider_catalog_override:
    try:
        parsed_catalog = json.loads(_transcription_provider_catalog_override)
        if isinstance(parsed_catalog, list):
            TRANSCRIPTION_PROVIDER_CATALOG = parsed_catalog
        elif isinstance(parsed_catalog, dict):
            TRANSCRIPTION_PROVIDER_CATALOG = parsed_catalog.get("providers") or [
                _default_gpt_provider_entry,
                _default_gemini_provider_entry,
                _default_assemblyai_provider_entry,
            ]
        else:
            logging.warning("TRANSCRIPTION_PROVIDER_CATALOG must be a list or dict. Using defaults.")
            TRANSCRIPTION_PROVIDER_CATALOG = [
                _default_gpt_provider_entry,
                _default_gemini_provider_entry,
                _default_assemblyai_provider_entry,
            ]
    except json.JSONDecodeError as catalog_error:
        logging.warning("Could not parse TRANSCRIPTION_PROVIDER_CATALOG JSON (%s). Using defaults.", catalog_error)
        TRANSCRIPTION_PROVIDER_CATALOG = [
            _default_gpt_provider_entry,
            _default_gemini_provider_entry,
            _default_assemblyai_provider_entry,
        ]
else:
    TRANSCRIPTION_PROVIDER_CATALOG = [
        _default_gpt_provider_entry,
        _default_gemini_provider_entry,
        _default_assemblyai_provider_entry,
    ]

GEMINI_TRANSCRIPTION_PROMPT = os.getenv(
    "GEMINI_TRANSCRIPTION_PROMPT",
    "Transcribe the audio accurately in the original language. Return only the transcription text without additional commentary.",
)
GEMINI_TRANSCRIBE_MAX_RETRIES = int(os.getenv("GEMINI_TRANSCRIBE_MAX_RETRIES", 3))
GEMINI_TRANSCRIBE_RETRY_BASE_DELAY = float(os.getenv("GEMINI_TRANSCRIBE_RETRY_BASE_DELAY", 5.0))
GEMINI_SUMMARY_DELAY_SECONDS = float(os.getenv("GEMINI_SUMMARY_DELAY_SECONDS", 3.0))

# Rate limiting dla Gemini API
# RPM = Requests Per Minute, RPD = Requests Per Day
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "10"))  # Domyślnie 10 requestów/minutę
GEMINI_RPD_LIMIT = int(os.getenv("GEMINI_RPD_LIMIT", "84"))  # Domyślnie 84 requesty/dzień

# === JWT (JSON Web Token) Configuration ===
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key") # ZMIEŃ TO W PRODUKCJI!
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)) # Token wygaśnie po 1440 minutach (24h)

# DODANY KOD DIAGNOSTYCZNY: Wydrukuj loaded SECRET_KEY i ALGORITHM
# print(f"[DIAGNOSTIC] Loaded SECRET_KEY (first 5 chars): {SECRET_KEY[:5]}...")
# print(f"[DIAGNOSTIC] Loaded ALGORITHM: {ALGORITHM}")
# KONIEC KODU DIAGNOSTYCZNEGO

# Dodaj tę linię:
QDRANT_SEARCH_RESULTS_LIMIT = int(os.getenv("QDRANT_SEARCH_RESULTS_LIMIT", 5)) # <-- DODAJ TO

# NOWE DODATKI DLA AGENTA CZATOWEGO
QDRANT_SEARCH_LIMIT_PER_COLLECTION = int(os.getenv("QDRANT_SEARCH_LIMIT_PER_COLLECTION", 5))
QDRANT_SEARCH_TOTAL_LIMIT = int(os.getenv("QDRANT_SEARCH_TOTAL_LIMIT", 20))

# ==================================================
#             General & Path Settings
# ==================================================
# Determine device based on CUDA availability
DEVICE = "cuda" # Nowa wartość dla GPU

# Ustawienia pamięci GPU
if torch.cuda.is_available():
    # Ogranicz użycie pamięci GPU
    torch.cuda.set_per_process_memory_fraction(0.9)  # Użyj tylko 70% dostępnej pamięci
    # Włącz optymalizacje pamięci
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Dodaj tę linię, aby wskazać niestandardową lokalizację pamięci podręcznej modeli Whisper.
# Upewnij się, że ta ścieżka jest poprawna i wskazuje na katalog nadrzędny 'models'.
# W Twoim przypadku: C:\Users\G\Documents\GitHub\sekretarz\whisper6.1\
# os.environ["XDG_CACHE_HOME"] = str(Path(__file__).parent.parent / "whisper6.1") # <--- DODAJ TĘ LINIĘ

# Ustaw zmienną środowiskową WHISPER_MODELS_DIR, aby wskazać bezpośrednio folder z modelami.
# To jest bardziej specyficzne dla Whisper i może działać lepiej.
# os.environ["WHISPER_MODELS_DIR"] = str(Path(__file__).parent.parent / "whisper6.1" / "models") # <--- DODAJ TĘ LINIĘ

# --- Modal diarization settings ---
USE_MODAL_DIARIZATION = _env_flag("USE_MODAL_DIARIZATION", False)
MODAL_DIARIZATION_APP = os.getenv("MODAL_DIARIZATION_APP", "sekretarz-diarization")
MODAL_DIARIZATION_FUNCTION = os.getenv("MODAL_DIARIZATION_FUNCTION", "run_diarization")
MODAL_ENV = os.getenv("MODAL_ENV")

# --- Directory Paths ---
# Directory for input transcripts for meeting minutes/ingestion
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "transcripts_alchemist")
# Base directory for chunked data during ingestion
BASE_CHUNK_DIR = os.getenv("BASE_CHUNK_DIR", "chunked_data")
# Output directory for generated meeting minutes
OUTPUT_DIR_MINUTES = os.getenv("OUTPUT_DIR_MINUTES", "meeting_minutes") # Renamed for clarity
# Output directory for API generated transcriptions
OUTPUT_DIR_API_TRANSCRIPTS = os.getenv("OUTPUT_DIR_API_TRANSCRIPTS", "output_api_transcripts")

# Directory for enrolled speaker voice samples
SPEAKER_ENROLLMENT_DIR = os.getenv("SPEAKER_ENROLLMENT_DIR", "speaker_enrollment")

# NEW: Directory for temporary file uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads") # Domyślny katalog na przesłane pliki

# Default input/output directories for MKV/Audio processing (can be overridden)
# It's best practice to set these via env variables or arguments rather than hardcoding paths.
DEFAULT_INPUT_DIR_MKV = os.getenv("DEFAULT_INPUT_DIR_MKV", r"C:\\path\\to\\your\\mkv_files") # Placeholder
DEFAULT_OUTPUT_DIR_MKV = os.getenv("DEFAULT_OUTPUT_DIR_MKV", r"C:\\path\\to\\your\\mkv_output") # Placeholder

DEFAULT_INPUT_DIR_AUDIO = os.getenv("DEFAULT_INPUT_DIR_AUDIO", r"C:\path\to\your\audio_files") # Placeholder
DEFAULT_OUTPUT_DIR_DIARIZATION = os.getenv("DEFAULT_OUTPUT_DIR_DIARIZATION", r"C:\path\to\your\diarization_output") # Placeholder

# Optional: Specify the full path to the ffmpeg executable if it's not in your system's PATH.
FFMPEG_PATH = os.getenv("FFMPEG_PATH", None) # Example: r"C:\ffmpeg\bin\ffmpeg.exe"

# ==================================================
#      Meeting Minutes Summarization Settings
# ==================================================
QUERY_SECTIONS = {
    "Główne Tematy Omówione": "Jakie były główne tematy omawiane podczas spotkania?",
    "Kluczowe Podjęte Decyzje": "Jakie kluczowe decyzje zostały podjęte podczas spotkania?",
    "Lista Zadań do Wykonania": "Wymień zadania wspomniane na spotkaniu, wraz z osobą odpowiedzialną i terminem, jeśli są dostępne.",
    "Ważne Działania Następcze lub Kolejne Kroki": "Jakie ważne działania następcze lub kolejne kroki zostały wspomniane?",
    "Najważniejsze Uwagi lub Komentarze od Uczestników": "Podsumuj istotne uwagi lub komentarze od uczestników spotkania."
}
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash") # ZMIENIONO: Model na gemini-2.5-flash
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "infly/inf-retriever-v1-1.5b") # Google's embedding model

# ==================================================
#         Data Ingestion & Chunking Settings
# ==================================================

# Domyślne wartości dla długości chunka i nakładania
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", 240)) # Domyślna długość chunka w sekundach
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))     # Domyślne nakładanie chunków w sekundach

# CHUNK_DURATIONS i CHUNK_DURATION_OVERLAPS mogą zostać usunięte lub zakomentowane,
# jeśli nie są już używane w innych miejscach kodu.
# CHUNK_DURATIONS = [240, 600, 900] # Example durations in seconds
# CHUNK_DURATION_OVERLAPS = [0, 30, 45, 120] # Example overlaps in seconds
# CHUNK_LINE_SETTINGS = [(65, 5), (125, 2)] # Example (line_count, overlap_lines) settings

# ==================================================
#     Whisper / MKV Processing Settings (HF Pipeline)
# ==================================================
HF_PIPELINE_MODEL = os.getenv("HF_PIPELINE_MODEL", "openai/whisper-large-v3")
HF_TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32 # FP16 recommended for CUDA
HF_ATTENTION_IMPL = "flash_attention_2" if is_flash_attn_2_available() and DEVICE.startswith("cuda") else "sdpa" # Use Flash Attention 2 if available
HF_CHUNK_LENGTH_S = int(os.getenv("HF_CHUNK_LENGTH_S", 30)) # Chunk length for pipeline processing
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", 8)) # Adjust based on GPU memory
HF_TRANSLATION_TASK = os.getenv("HF_TRANSLATION_TASK", "translate") # 'translate' or 'transcribe'

# ==================================================
#     Whisper / Audio Diarization Settings
# ==================================================
# Uses openai-whisper library and pyannote.audio
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3") # Model size for openai-whisper
PYANNOTE_PIPELINE = os.getenv("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1") # Pyannote diarization model
DIARIZATION_TASK = os.getenv("DIARIZATION_TASK", "translate") # Task for diarization mode ('translate' or 'transcribe')

# Pyannote Speaker Embedding and Identification Settings
PYANNOTE_EMBEDDING_MODEL = os.getenv("PYANNOTE_EMBEDDING_MODEL", "pyannote/embedding") # Model for speaker embeddings
PYANNOTE_IDENTIFICATION_THRESHOLD = float(os.getenv("PYANNOTE_IDENTIFICATION_THRESHOLD", 0.51)) # ZMIENIONO: Obniżono próg do 0.70

# ==================================================
#                Prompt Settings
# ==================================================
PROMPT_SETTINGS_FILE = "prompt-setting.json" # <--- DODAJ TĘ LINIĘ

# ==================================================
#                Validation Function
# ==================================================
def validate_config():
    """Validates required configuration variables."""
    logging.info("Checking configuration...")
    errors = []

    # Core requirements (adjust based on which parts of the app you use)
    if not QDRANT_ENDPOINT:
        errors.append("QDRANT_ENDPOINT environment variable not set (needed for vector store).")
    # if not QDRANT_API_KEY: # Uncomment if API key is strictly required
    #     errors.append("QDRANT_API_KEY environment variable not set.")
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY environment variable not set (needed for Gemini summarization/embeddings).")
    if not HF_TOKEN or HF_TOKEN == "hf_...": # Check if default placeholder is still there
         logging.warning("HF_TOKEN environment variable not set or using placeholder (needed for Pyannote diarization model download). Diarization might fail.")
         # Decide if this should be a hard error or just a warning
         # errors.append("HF_TOKEN environment variable not set or is placeholder.")

    # Check essential directories based on assumed usage
    # Scripts using these should ideally create them if they don't exist (makedirs(exist_ok=True))
    # Adding checks here provides an early warning.
    if not Path(TRANSCRIPTS_DIR).is_dir():
         logging.warning(f"Transcripts directory '{TRANSCRIPTS_DIR}' configured but does not exist.")
    if not Path(BASE_CHUNK_DIR).is_dir():
         logging.warning(f"Base chunk directory '{BASE_CHUNK_DIR}' configured but does not exist.")
    if not Path(OUTPUT_DIR_MINUTES).is_dir():
         logging.warning(f"Meeting minutes output directory '{OUTPUT_DIR_MINUTES}' configured but does not exist.")

    # Optional checks for MKV/Audio default paths (less critical as they are often overridden)
    # if DEFAULT_INPUT_DIR_MKV == r"C:\path\to\your\mkv_files":
    #      logging.warning("DEFAULT_INPUT_DIR_MKV is set to the default placeholder value.")
    # Add similar checks for other default paths if desired

    if USE_OPENAI_WHISPER_API and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY environment variable not set (required for Whisper API usage).")

    if USE_SUPABASE:
        if not SUPABASE_URL:
            errors.append("SUPABASE_URL environment variable not set (required when USE_SUPABASE=1).")
        if not SUPABASE_API_KEY:
            errors.append("SUPABASE_API_KEY environment variable not set (required when USE_SUPABASE=1).")
        if not SUPABASE_DB_URL:
            errors.append("SUPABASE_DB_URL (lub DATABASE_URL) environment variable not set for Supabase/Postgres connection.")

    if errors:
        for error in errors:
            logging.error(f"Configuration Error: {error}")
        raise ValueError("Missing or invalid essential configuration. Please check environment variables and logs.")
    else:
        logging.info("Configuration variables seem present (check warnings for potential issues).")

