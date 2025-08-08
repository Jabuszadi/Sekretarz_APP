#processor.py
import torch
import os
import glob
import subprocess
from tqdm import tqdm # For progress indication in console
import config # Common configuration
import warnings
import logging
import asyncio
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

# Import TranscriptionSegment z models.py
from models import TranscriptionSegment
# Import diarize_text i get_diarization_models z utils.py
from utils import diarize_text, get_diarization_models
import agent_db # NEW: Import agent_db

# Ignoruj ostrzeżenia o TF32
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")

# Ignoruj ostrzeżenia o std()
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

# Włącz TF32 dla lepszej wydajności (opcjonalne)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- Imports from mkv_processor ---
# Try importing transformers stuff safely
try:
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available
    hf_pipeline = pipeline # Rename to avoid conflict if needed, but seems ok
except ImportError:
    print("Warning: transformers library not found. Needed for MKV processing. Install it: pip install transformers")
    hf_pipeline = None

# --- Imports from diarization_processor ---
# Try importing models safely
try:
    import whisper
except ImportError:
    print("Warning: openai-whisper library not found. Needed for diarization processing. Install it: pip install -U openai-whisper")
    whisper = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline # Rename to avoid conflict
except ImportError:
    print("Warning: pyannote.audio library not found. Needed for diarization. Install it: pip install pyannote.audio")
    PyannotePipeline = None

# ================================================
# Functions from mkv_processor.py
# ================================================

def convert_mkv_to_wav(mkv_path, output_wav_path, ffmpeg_path=None):
    """Converts an MKV file to a 16kHz mono WAV file using ffmpeg."""
    ffmpeg_executable = ffmpeg_path or config.FFMPEG_PATH or 'ffmpeg'
    command = [
        ffmpeg_executable,
        '-i', mkv_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y',
        output_wav_path
    ]
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        result = subprocess.run(command, check=True, capture_output=True, text=True, creationflags=creationflags)
        print(f"Successfully converted '{os.path.basename(mkv_path)}' to '{os.path.basename(output_wav_path)}'")
        return True
    except FileNotFoundError:
        print(f"Error: '{ffmpeg_executable}' command not found. Is ffmpeg installed and in PATH, or is FFMPEG_PATH in config.py set correctly?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error converting '{os.path.basename(mkv_path)}': Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during conversion of '{os.path.basename(mkv_path)}': {e}")
        return False

def find_mkv_files(directory):
    """Finds all MKV files recursively in the specified directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return []
    mkv_files = glob.glob(os.path.join(directory, '**', '*.mkv'), recursive=True)
    print(f"Found {len(mkv_files)} MKV files in '{directory}'.")
    return mkv_files

def process_mkv_files(input_directory, output_directory):
    """
    Finds MKV files, converts them to WAV, processes them (transcribe/translate)
    using the Hugging Face Whisper pipeline, and saves the result to a TXT file.
    """
    if hf_pipeline is None:
        print("Error: Hugging Face pipeline (transformers) is not available. Cannot process MKV files.")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")
        except OSError as e:
            print(f"Error creating output directory {output_directory}: {e}")
            return

    mkv_files = find_mkv_files(input_directory)
    if not mkv_files:
        return

    print(f"Initializing Hugging Face pipeline with model: {config.HF_PIPELINE_MODEL}")
    print(f"Using device: {config.DEVICE}, dtype: {config.HF_TORCH_DTYPE}, attention: {config.HF_ATTENTION_IMPL}")
    try:
        # Use the imported and potentially renamed hf_pipeline
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=config.HF_PIPELINE_MODEL,
            torch_dtype=config.HF_TORCH_DTYPE,
            device=config.DEVICE,
            model_kwargs={"attn_implementation": config.HF_ATTENTION_IMPL},
        )
    except Exception as e:
        print(f"Error initializing Hugging Face pipeline: {e}")
        return

    print(f"Starting processing of {len(mkv_files)} MKV files...")
    for mkv_file in tqdm(mkv_files, desc="Processing MKV files", unit="file"):
        print(f"\nProcessing: {mkv_file}")
        base_name = os.path.splitext(os.path.basename(mkv_file))[0]
        wav_file = os.path.join(output_directory, base_name + '.wav')
        output_txt_path = os.path.join(output_directory, base_name + '.txt')

        if not convert_mkv_to_wav(mkv_file, wav_file):
            print(f"Skipping processing for '{os.path.basename(mkv_file)}' due to conversion error.")
            continue

        print(f"Processing audio: {wav_file} (Task: {config.HF_TRANSLATION_TASK})")
        generate_kwargs = {"task": config.HF_TRANSLATION_TASK}

        try:
            outputs = pipe(
                wav_file,
                chunk_length_s=config.HF_CHUNK_LENGTH_S,
                batch_size=config.HF_BATCH_SIZE,
                return_timestamps=True,
                generate_kwargs=generate_kwargs
            )
            txt_content = outputs.get("text", "").strip()

            try:
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                print(f"Text output saved: {output_txt_path}")
            except IOError as e:
                 print(f"Error writing text file {output_txt_path}: {e}")

            try:
                os.remove(wav_file)
                print(f"Removed intermediate WAV file: {wav_file}")
            except OSError as e:
                print(f"Warning: Could not remove intermediate WAV file {wav_file}: {e}")

        except Exception as e:
            print(f"Error during pipeline processing for {wav_file}: {e}")

    print("\nMKV processing finished.")


# ================================================
# Functions from diarization_processor.py
# ================================================

# Ta funkcja (load_diarization_models) została przeniesiona do utils.py i jest importowana.
# Jej definicja nie jest już potrzebna bezpośrednio w processor.py.


# --- Main Audio Processing Function (for API use) ---
async def process_audio(
    audio_path: Path,
    file_job_id: str,
    whisper_model_size: str
) -> Tuple[List[TranscriptionSegment], str, int]: # Typowanie na 3 wartości
    """
    Processes a single audio file: transcribes, diarizes, and saves the transcription.
    Returns a list of TranscriptionSegment objects, the saved transcription filename, and transcript_id.
    """
    logging.info(f"Starting audio processing for file_job_id {file_job_id}: {audio_path.name}")
    
    transcription_result = None
    diarization_result = None
    aligned_segments = []
    output_transcription_filename = None
    transcript_id = None # Inicjalizacja transcript_id

    try:
        # Load models
        whisper_model, diarization_pipeline, embedding_model = get_diarization_models()

        if whisper_model is None:
            logging.error("Whisper model is not loaded. Cannot process audio.")
            return [], None, None # Zawsze zwracaj 3 wartości: (lista, string, int)

        # Transkrypcja
        logging.info("  1/3 Performing transcription with Whisper...")
        transcription_result = await asyncio.to_thread(
            whisper_model.transcribe,
            str(audio_path),
            verbose=False,
            fp16=config.DEVICE.startswith("cuda")
        )
        logging.info("  ✅ Transcription complete.")

        # Diarization
        logging.info("  2/3 Performing diarization with Pyannote...\n")
        diarization_result = await asyncio.to_thread(diarization_pipeline, str(audio_path))
        logging.info("  ✅ Diarization complete.")

        # Aligancja i zapis do pliku
        logging.info("  3/3 Aligning transcription with diarization and saving...\n")
        aligned_segments = diarize_text(
            transcription_result,
            diarization_result,
            original_audio_path=audio_path
        )

        output_dir = Path(config.OUTPUT_DIR_API_TRANSCRIPTS) / file_job_id
        os.makedirs(output_dir, exist_ok=True)
        base_filename = audio_path.stem
        safe_base_filename = "".join([c if c.isalnum() else "_" for c in base_filename])
        current_datetime_str = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
        output_transcription_filename = Path(output_dir) / f"{current_datetime_str}_{safe_base_filename}_diarized_transcription.txt"

        full_transcription_content = "" # Akumulacja pełnej transkrypcji
        with open(output_transcription_filename, "w", encoding="utf-8") as f:
            for segment in aligned_segments:
                speaker = segment[1]
                text = segment[2]
                line = f"{speaker}: {text}\n"
                f.write(line)
                full_transcription_content += line # Akumulacja

        logging.info(f"  ✅ Diarized transcription saved to {output_transcription_filename}")

        # Dodaj transkrypcję do bazy danych
        transcript_id = agent_db.add_transcript(full_transcription_content, str(output_transcription_filename))
        logging.info(f"  ✅ Transcription added to database with ID: {transcript_id}")

        result_tuple = (aligned_segments, str(output_transcription_filename), transcript_id)
        logging.info(f"DEBUG: process_audio returning {len(result_tuple)} values.")
        return result_tuple

    except Exception as e:
        logging.error(f"Error during audio processing for {audio_path.name}: {e}")
        import traceback
        traceback.print_exc()
        error_result_tuple = ([], None, None)
        logging.error(f"DEBUG: process_audio returning {len(error_result_tuple)} values on error.")
        return error_result_tuple
    finally:
        # Clear CUDA cache if available after processing each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug(f"Cleared CUDA cache for file_job_id {file_job_id} after audio processing.")

# Reszta pliku
# def process_audio_files_with_diarization(...):
# ...
