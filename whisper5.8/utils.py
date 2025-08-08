# utils.py
# Combined utilities from srt_utils.py, utils.py, and file_utils.py

import os
import logging
from datetime import timedelta, datetime
from pathlib import Path # Import Path for use in save_minutes_to_file
import re
import torch # Dodany import
import json # Import for JSON operations
import numpy as np # <--- DODAJ TĘ LINIĘ
import torchaudio # Potrzebne do ładowania audio dla pyannote.audio
from scipy.spatial.distance import cosine # Do obliczania podobieństwa embeddingów
import shutil # <--- DODAJ TĘ LINIĘ: Do czyszczenia tymczasowych plików/katalogów
import subprocess # DODANY IMPORT: Do uruchamiania komend systemowych

import config # <--- UPEWNIJ SIĘ, ŻE TO JEST TUTAJ, POZA BLOKIEM try-except

# Pyannote i Whisper powinny być importowane w tym pliku, ponieważ ich modele są tutaj ładowane
try:
    import whisper # Import dla openai-whisper
except ImportError:
    logging.warning("Warning: openai-whisper library not found. Diarization will not work.")
    whisper = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline, Model, Inference # Import dla pyannote
except ImportError:
    logging.warning("Warning: pyannote.audio library not found. Diarization will not work.")
    PyannotePipeline = None

from typing import List, Tuple, Dict, Any
import time # Import the time module
import asyncio # <--- DODAJ TĘ LINIĘ: Do obsługi operacji asynchronicznych takich jak asyncio.to_thread

# Dodany import dla EncoderClassifier z SpeechBrain
try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    logging.warning("Warning: speechbrain.pretrained.EncoderClassifier not found. Speaker embedding will not work.")
    EncoderClassifier = None

# Import configuration variables needed
try:
    # Import the specific variable needed for saving minutes
    from config import (
        OUTPUT_DIR_MINUTES,
        WHISPER_MODEL_SIZE,
        PYANNOTE_PIPELINE,
        HF_TOKEN,
        DEVICE, # Dodany import dla DEVICE
        PYANNOTE_EMBEDDING_MODEL,
        SPEAKER_ENROLLMENT_DIR,
        PYANNOTE_IDENTIFICATION_THRESHOLD,
        GOOGLE_API_KEY,
        GEMINI_MODEL_NAME,
        QUERY_SECTIONS,
        PROMPT_SETTINGS_FILE
    )
except ImportError:
    logging.warning("Warning: Could not import necessary config variables. Make sure config.py exists and defines them.")
    # Define fallbacks if needed, or handle the error appropriately elsewhere
    OUTPUT_DIR_MINUTES = "meeting_minutes" # Example fallback, matches name in config
    WHISPER_MODEL_SIZE = "base" # Fallback
    PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1" # Fallback
    HF_TOKEN = None # Fallback
    DEVICE = "cpu" # Fallback
    PYANNOTE_EMBEDDING_MODEL = None # Fallback
    SPEAKER_ENROLLMENT_DIR = "speaker_enrollment" # Fallback
    PYANNOTE_IDENTIFICATION_THRESHOLD = 0.5 # Fallback

# Global variables to cache models
_whisper_model = None
_diarization_pipeline = None
_embedding_model = None # Nowa zmienna dla modelu embeddingu
_enrolled_speakers_profiles: Dict[str, np.ndarray] = {} # Przechowuje nazwę mówcy -> embedding

# ================================================
# Functions from srt_utils.py
# ================================================

def format_timestamp(seconds):
    """Formats seconds into SRT timestamp format (HH:MM:SS,ms)."""
    # Original instruction: format should be always 00:00:00,000
    # Let's implement the actual formatting
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        # Handle invalid input, return default or raise error
        return "00:00:00,000" # Default placeholder for invalid data

    # Calculate hours, minutes, seconds, milliseconds
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds_total = divmod(remainder, 60)
    seconds_part, milliseconds = divmod(seconds_total, 1)

    # Format parts with leading zeros
    hours_str = f"{int(hours):02d}"
    minutes_str = f"{int(minutes):02d}"
    seconds_str = f"{int(seconds_part):02d}"
    ms_str = f"{int(milliseconds * 1000):03d}"

    return f"{hours_str}:{minutes_str}:{seconds_str},{ms_str}"


def save_to_srt(outputs, srt_file_path):
    """
    Saves the transcription output (with timestamps and chunks) from a Whisper-like pipeline
    result to an SRT file. Uses corrected timestamp formatting.

    Args:
        outputs (dict): The dictionary returned by the pipeline, expected to have a "chunks"
                        key, where each chunk is a dict with "timestamp" (tuple) and "text".
        srt_file_path (str): The path where the SRT file will be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    logging.info(f"Attempting to save SRT file to: {srt_file_path}")
    if not isinstance(outputs, dict) or "chunks" not in outputs:
        logging.error("Error: Invalid output format for SRT generation. Expected a dictionary with a 'chunks' key.")
        return False

    chunks = outputs.get("chunks")
    if not isinstance(chunks, list):
         logging.error(f"Error: Expected 'chunks' to be a list, but got {type(chunks)}.")
         return False

    try:
        with open(srt_file_path, "w", encoding="utf-8") as f:
            segment_index = 1
            for segment in chunks:
                if not isinstance(segment, dict):
                    logging.warning(f"Warning: Skipping invalid segment (not a dict): {segment}")
                    continue

                timestamps = segment.get("timestamp")
                text = segment.get("text", "").strip()

                if not isinstance(timestamps, (list, tuple)) or len(timestamps) != 2:
                    logging.warning(f"Warning: Skipping segment {segment_index} due to missing or invalid timestamps: {timestamps}")
                    continue

                start_time = timestamps[0]
                end_time = timestamps[1]

                if start_time is None or end_time is None:
                     logging.warning(f"Warning: Skipping segment {segment_index} due to None value in timestamps: ({start_time}, {end_time})")
                     continue

                if not text:
                     logging.warning(f"Warning: Skipping segment {segment_index} due to empty text.")
                     continue

                # Format timestamps using the corrected helper function
                start_time_str = format_timestamp(start_time)
                end_time_str = format_timestamp(end_time)

                # Prevent invalid SRT where end time is before start time (using original floats)
                try:
                    start_s = float(start_time)
                    end_s = float(end_time)
                    if start_s > end_s:
                         logging.warning(f"Warning: Correcting segment {segment_index} end time ({end_s}s) because it's before start time ({start_s}s).")
                         # Adjust end time string to match start time string if invalid
                         end_time_str = start_time_str
                except (ValueError, TypeError):
                    logging.warning(f"Warning: Could not compare original timestamps for segment {segment_index}: ({start_time}, {end_time})")
                    continue

                # Write SRT entry
                f.write(f"{segment_index}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")
                segment_index += 1

        logging.info(f"SRT file successfully saved: {srt_file_path}")
        return True

    except IOError as e:
        logging.error(f"Error writing SRT file to {srt_file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during SRT file generation: {e}")
        return False


# ================================================
# Functions originally from utils.py (Diarization Alignment Placeholder)
# ================================================

# IMPORTANT: The following `diarize_text` is a VERY BASIC PLACEHOLDER.
# You need to replace this with a proper implementation that aligns
# Whisper's transcription segments with Pyannote's speaker diarization results.
# Libraries like `stable-ts` or custom logic involving timestamp matching are needed.

def get_diarization_models():
    """
    Loads Whisper (openai), Pyannote diarization, and Pyannote embedding models based on config settings.
    Uses caching to load models only once.
    """
    global _whisper_model, _diarization_pipeline, _embedding_model

    if _whisper_model is not None and _diarization_pipeline is not None and _embedding_model is not None:
        logging.info("All models already loaded, returning cached instances.")
        return _whisper_model, _diarization_pipeline, _embedding_model

    logging.info("Loading diarization and embedding models (first time)...")
    logging.info(f"Using device: {DEVICE}")

    # Zwolnij pamięć GPU PRZED ZAŁADOWANIEM MODELI
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared CUDA cache before loading models.")

    # Load Whisper model (openai-whisper version)
    if whisper:
        try:
            logging.info(f"Loading OpenAI Whisper model: {WHISPER_MODEL_SIZE}")
            _whisper_model = whisper.load_model(
                WHISPER_MODEL_SIZE,
                device=DEVICE,
                download_root="./models"
            )
            logging.info("OpenAI Whisper model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading OpenAI Whisper model '{WHISPER_MODEL_SIZE}': {e}")
            logging.info("Attempting to load Whisper model on CPU...")
            try:
                _whisper_model = whisper.load_model(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    download_root="./models"
                )
                logging.info("OpenAI Whisper model loaded successfully on CPU.")
            except Exception as e_cpu:
                logging.error(f"Error loading OpenAI Whisper model on CPU: {e_cpu}")
                _whisper_model = None

    # Load Pyannote diarization pipeline
    if PyannotePipeline:
        auth_token = None
        if HF_TOKEN and HF_TOKEN != "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":
            auth_token = HF_TOKEN
        else:
            logging.critical("CRITICAL: Hugging Face token (HF_TOKEN) is not set or is placeholder! "
                             "Pyannote diarization model WILL FAIL TO LOAD. "
                             "Please set HF_TOKEN in your .env file and accept terms on Hugging Face for model 'pyannote/speaker-diarization-3.1'.")

        pipeline_args = {"use_auth_token": auth_token} if auth_token else {}
        
        try:
            logging.info(f"Loading Pyannote diarization pipeline: {PYANNOTE_PIPELINE}")
            _diarization_pipeline = PyannotePipeline.from_pretrained(
                PYANNOTE_PIPELINE,
                **pipeline_args
            ).to(torch.device(DEVICE))
            logging.info("Pyannote diarization pipeline loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Pyannote diarization pipeline: {e}")
            logging.info("Attempting to load Pyannote diarization pipeline on CPU...")
            try:
                _diarization_pipeline = PyannotePipeline.from_pretrained(
                    PYANNOTE_PIPELINE,
                    **pipeline_args
                ).to(torch.device("cpu"))
                logging.info("Pyannote diarization pipeline loaded successfully on CPU.")
            except Exception as e_cpu:
                logging.error(f"Error loading Pyannote diarization pipeline on CPU: {e_cpu}")
                _diarization_pipeline = None

    # Load Pyannote embedding model using SpeechBrain's EncoderClassifier
    if EncoderClassifier:
        try:
            logging.info(f"Loading Pyannote embedding model (SpeechBrain/ECAPA-TDNN) using SpeechBrain EncoderClassifier on {DEVICE}...")
            _embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": DEVICE},
                # use_auth_token=HF_TOKEN # If this model requires specific auth (test_embedding showed it works without it for this model)
            )
            logging.info("SpeechBrain embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SpeechBrain embedding model on {DEVICE}: {e}")
            logging.info("Attempting to load SpeechBrain embedding model on CPU...")
            try:
                _embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                    # use_auth_token=HF_TOKEN # If this model requires specific auth (test_embedding showed it works without it for this model)
                )
                logging.info("SpeechBrain embedding model loaded successfully on CPU.")
            except Exception as e_cpu:
                logging.error(f"Error loading SpeechBrain embedding model on CPU: {e_cpu}")
                _embedding_model = None
    else:
        logging.error("SpeechBrain EncoderClassifier is not available. Cannot load embedding model.")


    if _whisper_model is None or _diarization_pipeline is None or _embedding_model is None:
        logging.critical("CRITICAL: One or more required models failed to load. Diarization and speaker enrollment features may not work.")
        # Możesz tutaj rzucić wyjątek, jeśli uznasz, że brakujące modele uniemożliwiają działanie aplikacji
        # raise RuntimeError("Required models failed to load.")

    return _whisper_model, _diarization_pipeline, _embedding_model

def load_enrolled_speakers():
    """
    Loads enrolled speaker profiles from the SPEAKER_ENROLLMENT_DIR.
    Each speaker's profile is expected to be in a separate .npy file within their named subdirectory.
    """
    global _enrolled_speakers_profiles
    _enrolled_speakers_profiles = {} # DODANO: Wyczyść listę przed ponownym ładowaniem
    logging.info(f"DEBUG: load_enrolled_speakers: Current _enrolled_speakers_profiles before loading: {list(_enrolled_speakers_profiles.keys())} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO

    enrollment_path = Path(SPEAKER_ENROLLMENT_DIR)
    logging.info(f"DEBUG: Loading enrolled speaker profiles from resolved path: {enrollment_path.resolve()}") # DODANO
    enrollment_path.mkdir(parents=True, exist_ok=True) # Upewnij się, że katalog istnieje

    logging.info(f"Loading enrolled speaker profiles from: {enrollment_path}")
    for speaker_dir in enrollment_path.iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            profile_file = speaker_dir / "embedding.npy" # Standardowa nazwa pliku dla embeddingu
            logging.info(f"DEBUG: Checking for profile file: {profile_file.resolve()}") # DODANO
            if profile_file.exists():
                try:
                    embedding = np.load(profile_file)
                    _enrolled_speakers_profiles[speaker_name] = embedding
                    logging.info(f"Loaded profile for speaker: {speaker_name}")
                except Exception as e:
                    logging.error(f"Error loading embedding for {speaker_name} from {profile_file}: {e}")
            else:
                logging.warning(f"No embedding.npy found for speaker: {speaker_name} in {speaker_dir.resolve()}") # ZMIENIONO
    logging.info(f"Finished loading. Total enrolled speakers: {len(_enrolled_speakers_profiles)} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO
    logging.info(f"DEBUG: load_enrolled_speakers: Final _enrolled_speakers_profiles after loading: {list(_enrolled_speakers_profiles.keys())} (ID: {id(_enrolled_speakers_profiles)})") # ZMIENIONO


def delete_speaker(speaker_name: str) -> bool:
    """
    Deletes a speaker's enrollment profile (embedding file and directory).
    Args:
        speaker_name (str): The name of the speaker to delete.
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    global _enrolled_speakers_profiles
    logging.info(f"DEBUG: delete_speaker called for '{speaker_name}'. Current _enrolled_speakers_profiles ID: {id(_enrolled_speakers_profiles)}") # DODANO
    speaker_profile_dir = Path(SPEAKER_ENROLLMENT_DIR) / speaker_name
    
    if speaker_name in _enrolled_speakers_profiles:
        del _enrolled_speakers_profiles[speaker_name]
        logging.info(f"Removed speaker '{speaker_name}' from in-memory profiles. Current _enrolled_speakers_profiles ID after deletion: {id(_enrolled_speakers_profiles)}") # ZMIENIONO
    
    if speaker_profile_dir.exists():
        try:
            shutil.rmtree(speaker_profile_dir)
            logging.info(f"Successfully deleted speaker directory: {speaker_profile_dir}")
            return True
        except Exception as e:
            logging.error(f"❌ Error deleting speaker directory {speaker_profile_dir}: {e}")
            return False
    else:
        logging.warning(f"Speaker directory for '{speaker_name}' not found at {speaker_profile_dir}. Profile might have been removed manually or never existed.")
        # If it was in memory but not on disk, we still consider it a success if it's no longer there
        return True if speaker_name not in _enrolled_speakers_profiles else False

async def enroll_speaker_from_audio(
    audio_file_path: Path,
    human_name: str
) -> bool:
    """
    Enrolls a speaker by generating an embedding from the provided audio file
    and saving it to the speaker enrollment directory.
    """
    logging.info(f"Enrolling speaker '{human_name}' from audio: {audio_file_path.name}")

    if _embedding_model is None:
        logging.error(f"❌ Pyannote embedding model not loaded. Cannot enroll speaker '{human_name}'.")
        return False

    try:
        # Load audio file using torchaudio
        # Signal will be (channels, samples)
        signal, sample_rate = await asyncio.to_thread(torchaudio.load, str(audio_file_path))

        # Ensure correct sample rate (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            logging.warning(f"Audio sample rate is {sample_rate}Hz. Resampling to 16000Hz for speaker '{human_name}'.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = await asyncio.to_thread(resampler, signal)
            sample_rate = 16000 # Update sample_rate after resampling

        # Ensure mono audio if stereo
        if signal.shape[0] > 1:
            logging.info(f"Converting stereo audio to mono for speaker '{human_name}'.")
            signal = torch.mean(signal, dim=0, keepdim=True) # Resulting shape is (1, samples)

        # Prepare signal for the model: _embedding_model.encode_batch expects (batch_size, samples)
        # So, squeeze if it's (1, samples) then unsqueeze back if needed by the model
        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0) # Convert from (1, samples) to (samples,) for `encode_batch`
        
        # Move signal to the correct device for inference
        signal = signal.to(DEVICE)

        # Generate embedding
        # The encode_batch method expects (batch_size, samples) or (batch_size, channels, samples)
        # We pass (1, samples) by unsqueezing to add batch dimension if it's (samples,)
        # And then squeeze(0) to remove the batch dimension from the output
        embedding = await asyncio.to_thread(lambda: _embedding_model.encode_batch(signal.unsqueeze(0)).squeeze(0).cpu().numpy())

        # Check if embedding was successfully generated
        if embedding is None:
            logging.error(f"❌ Failed to generate embedding for speaker '{human_name}'.")
            return False

        # --- NOWE LINIE: Zapisz embedding do pliku ---
        speaker_profile_dir = Path(SPEAKER_ENROLLMENT_DIR) / human_name
        logging.info(f"DEBUG: Attempting to create speaker directory: {speaker_profile_dir.resolve()}") # DODANO
        speaker_profile_dir.mkdir(parents=True, exist_ok=True) # Upewnij się, że katalog mówcy istnieje
        logging.info(f"DEBUG: Speaker directory created/exists: {speaker_profile_dir.resolve()}") # DODANO

        embedding_file_path = speaker_profile_dir / "embedding.npy"
        logging.info(f"DEBUG: Attempting to save embedding to: {embedding_file_path.resolve()}") # DODANO
        np.save(embedding_file_path, embedding)
        logging.info(f"Saved embedding for '{human_name}' to {embedding_file_path.resolve()}") # ZMIENIONO
        # --- KONIEC NOWYCH LINII ---

        # Save the embedding (assuming _enrolled_speakers_profiles is handled globally)
        _enrolled_speakers_profiles[human_name] = embedding
        logging.info(f"Successfully generated and stored embedding for speaker '{human_name}'. Current _enrolled_speakers_profiles ID: {id(_enrolled_speakers_profiles)}") # DODANO
        return True

    except Exception as e:
        logging.error(f"❌ Error during speaker enrollment for '{human_name}': {e}", exc_info=True)
        return False


def identify_speaker(audio_segment_path: Path) -> str:
    """
    Identifies the speaker of a given audio segment by comparing its embedding
    with known enrolled speaker profiles.
    Returns the human-readable name of the identified speaker or "SPEAKER_UNKNOWN".
    """
    global _embedding_model, _enrolled_speakers_profiles

    logging.info(f"DEBUG: identify_speaker called for: {audio_segment_path.name}")
    if _embedding_model is None:
        logging.error("❌ Pyannote embedding model not loaded. Cannot identify speaker.")
        return "SPEAKER_UNKNOWN"

    if not _enrolled_speakers_profiles:
        logging.warning("No speaker profiles enrolled. Cannot perform speaker identification.")
        return "SPEAKER_UNKNOWN"

    try:
        # Load audio file using torchaudio
        signal, sample_rate = torchaudio.load(str(audio_segment_path))

        # Ensure correct sample rate (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            logging.warning(f"Resampling audio segment from {sample_rate}Hz to 16000Hz for identification.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)
            sample_rate = 16000

        # Ensure mono audio if stereo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Prepare signal for the model
        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0)
        signal = signal.to(DEVICE)

        # Generate embedding
        segment_embedding = _embedding_model.encode_batch(signal.unsqueeze(0)).squeeze(0).cpu().numpy()

        if segment_embedding is None or segment_embedding.size == 0:
            logging.warning(f"Could not generate embedding for segment {audio_segment_path.name}. Returning UNKNOWN.")
            return "SPEAKER_UNKNOWN"
        logging.info(f"DEBUG: Generated embedding for segment with shape: {segment_embedding.shape}")


        # Porównanie z zarejestrowanymi profilami
        best_match_name = "SPEAKER_UNKNOWN"
        highest_similarity = PYANNOTE_IDENTIFICATION_THRESHOLD # Użyj progu jako minimum
        logging.info(f"DEBUG: Initial highest_similarity (threshold): {highest_similarity}")

        for enrolled_name, enrolled_embedding in _enrolled_speakers_profiles.items():
            logging.info(f"DEBUG: Comparing with enrolled speaker: {enrolled_name}")
            # Upewnij się, że oba embeddingi mają ten sam kształt i są np. 1D
            if enrolled_embedding.ndim > 1:
                enrolled_embedding = enrolled_embedding.flatten() # Spłaszcz, jeśli jest (1, D)
            if segment_embedding.ndim > 1:
                segment_embedding = segment_embedding.flatten() # Spłaszcz, jeśli jest (1, D)

            # Oblicz podobieństwo kosinusowe
            # Odległość kosinusowa = 1 - Podobieństwo kosinusowe
            # Więc podobieństwo = 1 - odległość
            similarity = 1 - cosine(enrolled_embedding, segment_embedding)
            logging.info(f"DEBUG: Similarity for {enrolled_name}: {similarity:.4f}")

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = enrolled_name
                logging.info(f"DEBUG: New best match: {best_match_name} with similarity: {highest_similarity:.4f}")


        if best_match_name != "SPEAKER_UNKNOWN":
            logging.info(f"Identified speaker as '{best_match_name}' with similarity: {highest_similarity:.2f}")
        else:
            logging.info(f"No speaker identified (max similarity: {highest_similarity:.2f}). Returning UNKNOWN.")

        return best_match_name

    except Exception as e:
        logging.error(f"❌ Error during speaker identification for {audio_segment_path.name}: {e}", exc_info=True)
        return "SPEAKER_UNKNOWN"


def diarize_text(transcription_result: Dict[str, Any], diarization_result: Any, original_audio_path: Path) -> List[Tuple[Any, str, str]]:
    """
    Aligns Whisper transcription segments with Pyannote diarization results,
    performing speaker identification for each segment and ensuring consistency
    across the entire file for identified speakers.
    """
    whisper_segments = transcription_result.get('segments', [])
    final_aligned_segments = []

    # Słownik do przechowywania mapowania etykiet Pyannote na nazwy mówców
    speaker_label_to_human_name = {}
    
    # Lista do przechowywania tymczasowych segmentów z ich etykietami Pyannote
    # Przed ostatecznym przypisaniem nazw ludzkich
    temp_segments_with_pyannote_labels = []

    try:
        speaker_turns = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_turns.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
        speaker_turns.sort(key=lambda x: x['start'])
        num_speakers = len(diarization_result.labels())
        logging.info(f"Pyannote identified {num_speakers} speakers via diarization.")

        if not speaker_turns:
            logging.warning("Pyannote diarization yielded no speaker turns. Falling back.")
            # Jeśli brak segmentów diarizacji, fallback do domyślnych etykiet
            for i, segment in enumerate(whisper_segments):
                start_time = segment.get('start')
                end_time = segment.get('end')
                text = segment.get('text', '').strip()
                if text and start_time is not None and end_time is not None:
                    segment_time_info = type('obj', (object,), {'start': start_time, 'end': end_time})()
                    final_aligned_segments.append((segment_time_info, "SPEAKER_UNKNOWN", text))
            return final_aligned_segments

    except Exception as e:
        logging.error(f"Error processing Pyannote diarization result or no turns found: {e}. Falling back to generic labels.", exc_info=True)
        # Jeśli błąd w diarizacji, fallback do domyślnych etykiet
        for i, segment in enumerate(whisper_segments):
            start_time = segment.get('start')
            end_time = segment.get('end')
            text = segment.get('text', '').strip()
            if text and start_time is not None and end_time is not None:
                segment_time_info = type('obj', (object,), {'start': start_time, 'end': end_time})()
                final_aligned_segments.append((segment_time_info, "SPEAKER_UNKNOWN", text))
        return final_aligned_segments

    # Load audio to extract segments for embedding
    try:
        waveform, sample_rate = torchaudio.load(original_audio_path)
        logging.info(f"DEBUG: Original audio loaded for diarization: {original_audio_path.name}, sample_rate: {sample_rate}, shape: {waveform.shape}")

    except Exception as e:
        logging.error(f"Error loading original audio file {original_audio_path} for speaker identification: {e}", exc_info=True)
        # Fallback if audio cannot be loaded, proceed without speaker identification
        logging.warning("Proceeding with diarization without speaker identification due to audio loading error.")
        for i, segment in enumerate(whisper_segments):
            start_time = segment.get('start')
            end_time = segment.get('end')
            text = segment.get('text', '').strip()
            if text and start_time is not None and end_time is not None:
                segment_time_info = type('obj', (object,), {'start': start_time, 'end': end_time})()
                final_aligned_segments.append((segment_time_info, "SPEAKER_UNKNOWN", text))
        return final_aligned_segments

    speaker_turn_index = 0
    # PRZEBIEG 1: Zbudowanie mapowania etykiet Pyannote na nazwy ludzkie
    for i, segment in enumerate(whisper_segments):
        start_time = segment.get('start')
        end_time = segment.get('end')
        text = segment.get('text', '').strip()
        
        if start_time is None or end_time is None or not text:
            continue # Pomiń segmenty bez tekstu lub czasów

        diarized_speaker_label = "SPEAKER_UNKNOWN" # Domyślna wartość

        # Znajdź etykietę Pyannote dla bieżącego segmentu Whisper
        best_overlap = 0
        current_diarized_speaker = "SPEAKER_UNKNOWN"

        # Przesuń speaker_turn_index, aby uniknąć ponownego sprawdzania już przetworzonych tur
        while speaker_turn_index < len(speaker_turns) and speaker_turns[speaker_turn_index]['end'] < start_time:
            speaker_turn_index += 1

        for j in range(speaker_turn_index, len(speaker_turns)):
            turn = speaker_turns[j]
            turn_start = turn['start']
            turn_end = turn['end']
            
            # Obliczanie nakładania
            overlap_start = max(start_time, turn_start)
            overlap_end = min(end_time, turn_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                current_diarized_speaker = turn['speaker']
            
            if turn_start > end_time:
                break # Jeśli bieżąca tura jest już poza segmentem, zakończ

        diarized_speaker_label = current_diarized_speaker
        logging.info(f"DEBUG (Pass 1): Segment {i} ({start_time:.2f}-{end_time:.2f}s) Pyannote label: {diarized_speaker_label}")

        # Jeśli etykieta Pyannote nie została jeszcze zmapowana na ludzką nazwę, spróbuj zidentyfikować
        if diarized_speaker_label not in speaker_label_to_human_name:
            segment_waveform_start = int(start_time * sample_rate)
            segment_waveform_end = int(end_time * sample_rate)

            if segment_waveform_start < segment_waveform_end and waveform.shape[1] > 0:
                audio_segment_tensor = waveform[:, segment_waveform_start:segment_waveform_end]
                
                if audio_segment_tensor.shape[1] > 0:
                    if audio_segment_tensor.shape[0] > 1:
                        audio_segment_tensor = torch.mean(audio_segment_tensor, dim=0, keepdim=True)

                    temp_segment_dir = Path("./temp_audio_segments")
                    temp_segment_dir.mkdir(exist_ok=True)
                    temp_audio_path = temp_segment_dir / f"segment_pass1_{i}_{start_time}-{end_time}.wav"
                    torchaudio.save(str(temp_audio_path), audio_segment_tensor, sample_rate)

                    identified_name = identify_speaker(temp_audio_path)
                    logging.info(f"DEBUG (Pass 1): Segment {i} - Identified name: {identified_name}")

                    if identified_name != "SPEAKER_UNKNOWN":
                        speaker_label_to_human_name[diarized_speaker_label] = identified_name
                        logging.info(f"DEBUG (Pass 1): Mapped {diarized_speaker_label} to {identified_name}")
                    
                    os.remove(temp_audio_path)
            else:
                logging.warning(f"Audio segment for identification has zero duration for segment {i} in Pass 1. Skipping identification.")
        
        # Zapisz segment z etykietą Pyannote do późniejszego przetworzenia w drugim przebiegu
        temp_segments_with_pyannote_labels.append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'pyannote_label': diarized_speaker_label
        })

    # PRZEBIEG 2: Przypisanie ostatecznych nazw mówców i zbudowanie finalnej transkrypcji
    for segment_data in temp_segments_with_pyannote_labels:
        start_time = segment_data['start']
        end_time = segment_data['end']
        text = segment_data['text']
        pyannote_label = segment_data['pyannote_label']

        assigned_speaker = speaker_label_to_human_name.get(pyannote_label, pyannote_label) # Użyj zmapowanej nazwy lub etykiety Pyannote

        segment_time_info = type('obj', (object,), {'start': start_time, 'end': end_time})()
        final_aligned_segments.append((segment_time_info, assigned_speaker, text))

    logging.info(f"Diarization and speaker identification complete. Total segments: {len(final_aligned_segments)}")
    return final_aligned_segments


# ================================================
# Functions from file_utils.py
# ================================================

from typing import Optional
from summarizer import generate_html_from_text # Dodaj ten import

def save_minutes_to_file(minutes_content_string: str, unique_collection_name: str, target_date_iso: str) -> str | None:
    """Saves the generated meeting minutes to a text file."""
    logging.info(f"DEBUG: save_minutes_to_file called with unique_collection_name: {unique_collection_name}, date: {target_date_iso}")
    
    minutes_file_path = None
    # metadata_file_path = None # Removed as JSON is no longer saved here

    try:
        formatted_date_for_folder = target_date_iso
        base_output_root = Path(OUTPUT_DIR_MINUTES)
        output_dir = base_output_root / formatted_date_for_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_base_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in unique_collection_name])
        safe_base_name = safe_base_name.strip('_-')

        final_file_base_name = f"{safe_base_name}"
        
        minutes_file_path = output_dir / f"{final_file_base_name}_summary.txt"
        # metadata_file_path = output_dir / f"{final_file_base_name}_metadata.json" # Removed

        # Save minutes to a text file - directly write the string
        with open(minutes_file_path, "w", encoding="utf-8") as f:
            f.write(minutes_content_string) # Write the pre-formatted string directly
        logging.info(f"✅ Meeting minutes saved to {str(minutes_file_path)}")

        # Removed all JSON metadata saving logic from here
        # with open(metadata_file_path, "w", encoding="utf-8") as f:
        #     json.dump(metadata_dict, f, indent=4, ensure_ascii=False)
        # logging.info(f"✅ Metadata saved to {str(metadata_file_path)}")

        return str(minutes_file_path)

    except Exception as e:
        minutes_path_str = str(minutes_file_path) if minutes_file_path else "N/A"
        # metadata_path_str = str(metadata_file_path) if metadata_file_path else "N/A" # Removed
        logging.error(f"❌ Failed to save minutes to file {minutes_path_str}: {e}") # Updated error message
        return None


# ================================================
# Main execution block (for testing utils.py directly)
# ================================================

if __name__ == '__main__':
    print("Testing utils.py combined functions...")

    # --- Test SRT formatting ---
    print("\n--- Testing format_timestamp ---")
    print(f"0 seconds: {format_timestamp(0)}")
    print(f"5.123 seconds: {format_timestamp(5.123)}")
    print(f"65.999 seconds: {format_timestamp(65.999)}")
    print(f"3670.001 seconds: {format_timestamp(3670.001)}")
    print(f"None input: {format_timestamp(None)}")
    print(f"Negative input: {format_timestamp(-10)}")

    # --- Test SRT saving (mock data) ---
    print("\n--- Testing save_to_srt (mock data) ---")
    mock_outputs_srt = {
        'chunks': [
            {'timestamp': (0.5, 2.1), 'text': ' Hello there.'},
            {'timestamp': (2.5, 5.0), 'text': ' General Kenobi.'},
            {'timestamp': (5.0, 4.0), 'text': ' Invalid time segment.'}, # Test invalid time
            {'timestamp': (6.0, None), 'text': ' Segment with None time.'}, # Test None time
            {'timestamp': (7.0, 8.0), 'text': ''}, # Test empty text
        ]
    }
    test_srt_path = "test_output.srt"
    save_to_srt(mock_outputs_srt, test_srt_path)
    if os.path.exists(test_srt_path):
        print(f"Check '{test_srt_path}' for SRT output.")
        # os.remove(test_srt_path) # Clean up test file
    else:
        print("SRT file saving test failed.")


    # --- Test Diarization Placeholder ---
    print("\n--- Testing diarize_text placeholder ---")
    mock_whisper_diar = {
        'text': 'Hello Speaker 1. How are you Speaker 2?',
        'segments': [
            {'start': 0.5, 'end': 2.1, 'text': ' Hello Speaker 1.'},
            {'start': 2.5, 'end': 5.0, 'text': ' How are you Speaker 2?'}
        ]
    }
    print("\nTesting with None diarization_result:")
    aligned_none = diarize_text(mock_whisper_diar, None, None)
    for seg in aligned_none:
        print(f"  Time: {seg[0].start:.2f}-{seg[0].end:.2f}, Speaker: {seg[1]}, Text: {seg[2]}")

    print("\nNOTE: Cannot fully test Pyannote alignment path without a proper mock Annotation object.")

    # --- Test File Saving (using OUTPUT_DIR_MINUTES) --- # MODIFIED Comment
    print("\n--- Testing save_minutes_to_file ---")
    logging.basicConfig(level=logging.INFO) # Configure logging for test output
    mock_minutes = {
        "Summary": "Discussed project X roadmap.",
        "Action Items": "- Alice to draft proposal by EOD.\n- Bob to schedule follow-up.",
        "Decisions": "Approved budget for Q3."
    }
    test_collection = "Project X Meeting"
    test_date = "2023-10-27"
    # Ensure the base output directory exists for the test
    output_path_minutes = Path(OUTPUT_DIR_MINUTES) # MODIFIED Use correct var
    if not output_path_minutes.exists():
        output_path_minutes.mkdir(parents=True)
    save_minutes_to_file(mock_minutes, test_collection, test_date)
    # You can manually check the output folder (e.g., meeting_minutes/2023-10-27/) for the generated file.

    print("\nUtils testing finished.")

def extract_date_from_filename(filename):
    """
    Próbuje wyciągnąć datę z nazwy pliku w formacie YYYY-MM-DD.
    Obsługuje formaty YYYY-MM-DD, YYYYMMDD.
    """
    match = re.search(r'(\d{4}[- _]?\d{2}[- _]?\d{2})', filename)
    if match:
        date_str = match.group(1).replace('-', '').replace('_', '').replace(' ', '')
        try:
            return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None

def get_file_creation_date(file_path):
    """Zwraca datę utworzenia pliku w formacie YYYY-MM-DD."""
    timestamp = os.path.getctime(file_path)
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def get_relevant_date_for_file(file_path):
    """
    Próbuje wyciągnąć datę z nazwy pliku, a jako fallback używa daty utworzenia.
    Zwraca datę w formacie YYYY-MM-DD.
    """
    filename = Path(file_path).name
    date_from_name = extract_date_from_filename(filename)
    if date_from_name:
        return date_from_name
    return get_file_creation_date(file_path)

def get_current_ram_usage_gb() -> float:
    """
    Pobiera bieżące zużycie pamięci RAM przez proces Pythona w GB (Windows).
    """
    try:
        command = f'wmic process where "ProcessId={subprocess.os.getpid()}" get WorkingSetSize /value'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)

        if result.returncode == 0:
            output = result.stdout
            for line in output.splitlines():
                if "WorkingSetSize" in line:
                    memory_bytes = int(line.split('=')[1])
                    memory_gb = memory_bytes / (1024**3)
                    return memory_gb
        else:
            logging.error(f"WMIC command failed with return code {result.returncode}: {result.stderr}")
    except Exception as e:
        logging.error(f"Error getting RAM usage: {e}")
    return 0.0

def get_current_vram_usage_gb():
    """
    Pobiera bieżące zużycie VRAM (pamięci karty graficznej) w GB,
    używając nvidia-smi (tylko dla kart NVIDIA).
    """
    try:
        # Komenda do pobrania zużycia pamięci używanej przez GPU w MiB
        command = 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
        
        # Oczekujemy, że wyjście będzie liczbą w MiB
        vram_mib = float(result.stdout.strip())
        vram_gb = vram_mib / 1024 # Konwersja MiB na GB
        return vram_gb
    except FileNotFoundError:
        logging.warning("nvidia-smi not found. VRAM usage cannot be retrieved. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"nvidia-smi command failed: {e.stderr}")
    except ValueError:
        logging.error(f"Could not parse VRAM usage from nvidia-smi output: {result.stdout}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting VRAM usage: {e}")
    return 0.0

# ===============================================
# Functions for Prompt Management
# ===============================================

PROMPT_SETTINGS_FILE = "prompt-setting.json" # Ensure this path is correct based on config

def load_prompt_config() -> Dict[str, Dict[str, str]]:
    """
    Loads prompt configurations from a JSON file.
    Handles migration from old single-level format to new nested format.
    """
    config_path = Path(PROMPT_SETTINGS_FILE)
    if not config_path.exists():
        logging.info(f"Prompt settings file not found: {config_path}. Returning empty configuration.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Heuristic to detect old, flat format: if the root is a dictionary
        # AND any of its values are NOT dictionaries themselves (i.e., they are strings, lists, etc.)
        is_old_flat_format = False
        if isinstance(data, dict):
            # Check if all values are strings (classic flat format) or any value is not a dict
            if data and all(isinstance(v, str) for v in data.values()):
                is_old_flat_format = True
            elif data and any(not isinstance(v, dict) for v in data.values()):
                # This covers cases where some values are not dicts, implying a flat structure
                # E.g., if someone accidentally put a list or a number as a top-level value
                is_old_flat_format = True
        else:
            # If the root is not even a dictionary, it's definitely the old, invalid-for-new-format structure
            is_old_flat_format = True


        if is_old_flat_format:
            logging.info("Detected old flat prompt configuration format. Migrating to nested format.")
            # Ensure data is a dict before wrapping, even if it was e.g. a list before
            migrated_data = {"Domyślny Zestaw": data if isinstance(data, dict) else {}}
            try:
                # Save the migrated data immediately to persist the new format
                with open(config_path, "w", encoding="utf-8") as wf:
                    json.dump(migrated_data, wf, ensure_ascii=False, indent=4)
                logging.info(f"Successfully migrated and saved prompt settings to new format: {config_path}")
            except Exception as save_e:
                logging.error(f"Error saving migrated prompt settings to {config_path}: {save_e}")
            return migrated_data
        else:
            return data # Already in the new nested format
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding prompt settings file {config_path}: {e}. Returning empty configuration.")
        return {} # Return empty config instead of re-raising
    except Exception as e:
        logging.error(f"Error loading prompt settings from {config_path}: {e}")
        return {} # Return empty config for other errors too

def save_prompt_config(prompt_name: str, prompts: Dict[str, str]) -> bool:
    """Saves a named set of prompt configurations to a JSON file."""
    # NOWA WALIDACJA: Nie zezwalamy na zapis pustego zestawu promptów
    if not prompts:
        logging.warning(f"Attempted to save an empty prompt set '{prompt_name}'. Operation aborted.")
        return False

    config_path = Path(PROMPT_SETTINGS_FILE)
    try:
        # Load existing configurations
        all_prompts = load_prompt_config() # This will now return the nested structure

        # Update or add the new prompt set
        all_prompts[prompt_name] = prompts

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(all_prompts, f, ensure_ascii=False, indent=4)
        logging.info(f"Prompt set '{prompt_name}' saved to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving prompt set '{prompt_name}' to {config_path}: {e}")
        return False

def delete_prompt_config(prompt_name: str) -> bool:
    """
    Deletes a prompt set from the prompt-setting.json file.
    Args:
        prompt_name (str): The name of the prompt set to delete.
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    config_data = load_prompt_config()
    if prompt_name in config_data:
        del config_data[prompt_name]
        try:
            with open(PROMPT_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ Successfully deleted prompt set '{prompt_name}' from {PROMPT_SETTINGS_FILE}")
            return True
        except Exception as e:
            logging.error(f"❌ Error deleting prompt set '{prompt_name}' from {PROMPT_SETTINGS_FILE}: {e}")
            return False
    else:
        logging.warning(f"Prompt set '{prompt_name}' not found in {PROMPT_SETTINGS_FILE}.")
        return False
