# config.py
import os
import logging
from pathlib import Path
import torch
from dotenv import load_dotenv
from transformers.utils import is_flash_attn_2_available

# --- Load Environment Variables ---
print("Attempting to load .env file...") # Temporary diagnostic print
load_dotenv()
print(".env file loading attempted.") # Temporary diagnostic print

# --- Configure Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # <--- Upewnij się, że poziom to DEBUG

# ==================================================
#             API Keys & Endpoints
# ==================================================
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Keep even if potentially optional for some setups
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Needed for Gemini/Google AI features

# It's strongly recommended to load sensitive tokens like HF_TOKEN from environment variables
# Example: HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN", "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") # Default fallback if not in .env

# Temporary diagnostic print to check the value of HF_TOKEN after getting it
print(f"DEBUG: HF_TOKEN from os.getenv: '{os.getenv('HF_TOKEN')}'")
print(f"DEBUG: HF_TOKEN assigned in config.py: '{HF_TOKEN}'")
if HF_TOKEN == "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":
    print("DEBUG: HF_TOKEN is still the default placeholder value!")
else:
    print("DEBUG: HF_TOKEN seems to have been loaded or set.")

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
PYANNOTE_IDENTIFICATION_THRESHOLD = float(os.getenv("PYANNOTE_IDENTIFICATION_THRESHOLD", 0.70)) # ZMIENIONO: Obniżono próg do 0.70

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

    if errors:
        for error in errors:
            logging.error(f"Configuration Error: {error}")
        raise ValueError("Missing or invalid essential configuration. Please check environment variables and logs.")
    else:
        logging.info("Configuration variables seem present (check warnings for potential issues).")

