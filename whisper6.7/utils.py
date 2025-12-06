# utils.py
# Combined utilities from srt_utils.py, utils.py, and file_utils.py

import os
import logging
from datetime import timedelta, datetime
from pathlib import Path # Import Path for use in save_minutes_to_file
import re
from typing import Dict, Any, Optional, List, Tuple
import torch # Dodany import
import json # Import for JSON operations
import numpy as np # <--- DODAJ TĘ LINIĘ
import torchaudio # Potrzebne do ładowania audio dla pyannote.audio
from scipy.spatial.distance import cosine # Do obliczania podobieństwa embeddingów
import shutil # <--- DODAJ TĘ LINIĘ: Do czyszczenia tymczasowych plików/katalogów
import subprocess # DODANY IMPORT: Do uruchamiania komend systemowych
import sys # Dodany import
import platform # Dodany import

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

try:
    import assemblyai as aai
except ImportError:
    aai = None

_ASSEMBLYAI_MODEL_ENUM_MAP: Dict[str, Any] = {}
if aai is not None:
    try:
        _ASSEMBLYAI_MODEL_ENUM_MAP = {
            "best": getattr(aai.SpeechModel, "best", None),
            "nano": getattr(aai.SpeechModel, "nano", None),
            "slam-1": getattr(aai.SpeechModel, "slam_1", None) or getattr(aai.SpeechModel, "slam1", None),
            "universal": getattr(aai.SpeechModel, "universal", None),
        }
    except AttributeError:
        _ASSEMBLYAI_MODEL_ENUM_MAP = {}

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

from typing import List, Tuple, Dict, Any, Optional
import asyncio # <--- DODAJ TĘ LINIĘ: Do obsługi operacji asynchronicznych takich jak asyncio.to_thread

# Dodany import dla EncoderClassifier z SpeechBrain
try:
    from speechbrain.inference.speaker import EncoderClassifier
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
        PROMPT_SETTINGS_FILE,
        USE_OPENAI_WHISPER_API,
        OPENAI_API_KEY,
        OPENAI_API_BASE,
        OPENAI_ORG_ID,
        OPENAI_WHISPER_MODEL,
        OPENAI_WHISPER_RESPONSE_FORMAT,
        OPENAI_WHISPER_LANGUAGE,
        OPENAI_WHISPER_TEMPERATURE,
        OPENAI_WHISPER_MODEL_MAP,
        DEFAULT_TRANSCRIPTION_PROVIDER,
        DEFAULT_GPT_TRANSCRIPTION_MODEL,
        DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
        TRANSCRIPTION_PROVIDER_CATALOG,
        WHISPER_LEGACY_ALIAS_REDIRECTS,
        GEMINI_TRANSCRIPTION_PROMPT,
        GEMINI_TRANSCRIBE_API_KEY,
        GEMINI_TRANSCRIBE_MAX_RETRIES,
        GEMINI_TRANSCRIBE_RETRY_BASE_DELAY,
        DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL,
        ASSEMBLYAI_API_KEY,
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
    USE_OPENAI_WHISPER_API = False
    OPENAI_API_KEY = None
    OPENAI_API_BASE = None
    OPENAI_ORG_ID = None
    OPENAI_WHISPER_MODEL = "gpt-4o-mini-transcribe"
    OPENAI_WHISPER_RESPONSE_FORMAT = "verbose_json"
    OPENAI_WHISPER_LANGUAGE = None
    OPENAI_WHISPER_TEMPERATURE = None
    OPENAI_WHISPER_MODEL_MAP = {
        "default": None,
        "mini": "gpt-4o-mini-transcribe",
        "normal": "gpt-4o-transcribe",
    }
    DEFAULT_TRANSCRIPTION_PROVIDER = "gpt"
    DEFAULT_GPT_TRANSCRIPTION_MODEL = "normal"
    DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-1.5-flash"
    TRANSCRIPTION_PROVIDER_CATALOG = [
        {
            "provider_id": "gpt",
            "label": "OpenAI Whisper",
            "default_model": DEFAULT_GPT_TRANSCRIPTION_MODEL,
            "models": [
                {"id": "mini", "label": "Mini (gpt-4o-mini-transcribe)", "metadata": {"openai_target": "gpt-4o-mini-transcribe"}},
                {"id": "normal", "label": "Normal (gpt-4o-transcribe)", "metadata": {"openai_target": "gpt-4o-transcribe"}},
            ],
        },
        {
            "provider_id": "gemini",
            "label": "Gemini",
            "default_model": DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
            "models": [
                {"id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"},
                {"id": "gemini-1.5-pro", "label": "Gemini 1.5 Pro"},
            ],
        },
    ]
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
    GEMINI_TRANSCRIPTION_PROMPT = (
        "Transcribe the audio accurately in the original language. "
        "Return only the transcription text without any additional commentary."
    )
    GEMINI_TRANSCRIBE_API_KEY = None
    GEMINI_TRANSCRIBE_MAX_RETRIES = 3
    GEMINI_TRANSCRIBE_RETRY_BASE_DELAY = 1.0

# Global helper utilities
def _sanitize_username(username: str, fallback: str = "user") -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_.-]', '_', username.strip())
    return sanitized or fallback

# Global variables to cache models
_whisper_models: Dict[str, Any] = {}
_diarization_pipeline = None
_embedding_model = None # Nowa zmienna dla modelu embeddingu
_enrolled_speakers_profiles: Dict[str, Dict[str, np.ndarray]] = {} # Mapuje przestrzeń użytkownika -> {nazwa mówcy -> embedding}
_gemini_transcribers: Dict[str, "GeminiTranscriber"] = {}
_assemblyai_transcribers: Dict[str, "AssemblyAITranscriber"] = {}

_DEFAULT_OPENAI_WHISPER_MODEL_MAP = {
    "default": None,
    "mini": "gpt-4o-mini-transcribe",
    "normal": "gpt-4o-transcribe",
}

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


class GeminiTranscriptionError(Exception):
    """Wewnętrzny wyjątek sygnalizujący błędy transkrypcji Gemini."""


class AssemblyAITranscriptionError(Exception):
    """Wewnętrzny wyjątek sygnalizujący błędy transkrypcji AssemblyAI."""


class OpenAIWhisperAPI:
    """
    Lightweight wrapper zapewniający interfejs zbliżony do openai-whisper dla użycia
    oficjalnego OpenAI Whisper API. Zwraca strukturę kompatybilną z resztą pipeline'u.
    """

    uses_remote_api = True

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        response_format: str = "verbose_json",
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        requested_alias: Optional[str] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai package is not installed.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to use the OpenAI Whisper API.")

        self.model_name = model_name
        self.default_response_format = response_format or "verbose_json"
        self.default_language = language
        self.default_temperature = temperature
        cleaned_alias = (requested_alias or "").strip()
        self.requested_alias = cleaned_alias or None
        self.resolved_model_name = model_name
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            organization=organization or None,
        )

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        response_format = kwargs.pop("response_format", self.default_response_format)
        language = kwargs.pop("language", self.default_language)
        temperature = kwargs.pop("temperature", self.default_temperature)

        request_kwargs: Dict[str, Any] = {}
        if language:
            request_kwargs["language"] = language
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        for key, value in kwargs.items():
            if value is not None:
                request_kwargs[key] = value

        effective_response_format = response_format or "verbose_json"

        with open(audio_path, "rb") as audio_file:
            transcription = self._client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                response_format=effective_response_format,
                **request_kwargs,
            )

        return self._convert_response(transcription, effective_response_format)

    @staticmethod
    def _convert_response(transcription: Any, response_format: str) -> Dict[str, Any]:
        if hasattr(transcription, "model_dump"):
            data = transcription.model_dump()
        elif hasattr(transcription, "to_dict"):
            data = transcription.to_dict()
        elif isinstance(transcription, dict):
            data = transcription
        else:
            data = {}

        result: Dict[str, Any] = {
            "text": data.get("text"),
        }

        if "language" in data:
            result["language"] = data.get("language")
        if "duration" in data:
            result["duration"] = data.get("duration")
        if "segments" in data and response_format == "verbose_json":
            segments = []
            for segment in data.get("segments") or []:
                segment_dict = segment if isinstance(segment, dict) else {}
                if not segment_dict and hasattr(segment, "model_dump"):
                    segment_dict = segment.model_dump()
                segments.append(
                    {
                        "id": segment_dict.get("id"),
                        "seek": segment_dict.get("seek"),
                        "start": segment_dict.get("start"),
                        "end": segment_dict.get("end"),
                        "text": segment_dict.get("text"),
                        "tokens": segment_dict.get("tokens"),
                        "temperature": segment_dict.get("temperature"),
                        "avg_logprob": segment_dict.get("avg_logprob"),
                        "compression_ratio": segment_dict.get("compression_ratio"),
                        "no_speech_prob": segment_dict.get("no_speech_prob"),
                    }
                )
            result["segments"] = segments
            if not segments:
                logging.warning(
                    "OpenAI Whisper API response did not include segments; diarization alignment may be impacted."
                )
        else:
            result["segments"] = []

        return result


class GeminiTranscriber:
    """
    Wrapper na nowy interfejs Google Gemini (google-genai) dla transkrypcji audio.
    """

    uses_remote_api = True
    provider = "gemini"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        fallback_api_key: Optional[str] = None,
    ):
        if google_genai is None:
            raise ImportError("Pakiet google-genai nie jest zainstalowany.")
        if not api_key:
            raise ValueError("GEMINI_TRANSCRIBE_API_KEY jest wymagany do transkrypcji Gemini.")

        self.model_name = model_name.strip()
        self._api_key = api_key
        self._fallback_api_key = fallback_api_key
        self._client = google_genai.Client(api_key=api_key)

    def _wait_for_active_file(self, file_obj: Any, timeout_seconds: int = 60) -> Any:
        import time

        start = time.time()
        while True:
            current = self._client.files.get(name=file_obj.name)
            state = getattr(current, "state", None)
            state_name = getattr(state, "name", str(state)).upper() if state else None
            if state_name == "ACTIVE":
                return current
            if state_name == "FAILED":
                raise RuntimeError("Przetwarzanie pliku przez Gemini zakończyło się błędem (FAILED).")
            if time.time() - start > timeout_seconds:
                raise TimeoutError("Przetwarzanie pliku przez Gemini trwa zbyt długo (timeout).")
            time.sleep(1)

    def _cleanup_file(self, file_name: Optional[str]) -> None:
        if not file_name:
            return
        try:
            self._client.files.delete(name=file_name)
        except Exception as cleanup_error: # pragma: no cover - best effort cleanup
            logging.debug("Nie udało się usunąć pliku Gemini %s: %s", file_name, cleanup_error)

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        try:
            candidates = getattr(response, "candidates", [])
            for candidate in candidates or []:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", [])
                for part in parts or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return part_text.strip()
        except Exception as parse_error:
            logging.debug("Nie udało się sparsować odpowiedzi Gemini: %s", parse_error)
        return ""

    def transcribe_file(self, file_path: str, prompt: Optional[str] = None) -> str:
        prompt_text = prompt or GEMINI_TRANSCRIPTION_PROMPT

        upload = self._client.files.upload(
            file=file_path,
            config={"mime_type": "audio/wav"},
        )

        active_file = None
        try:
            active_file = self._wait_for_active_file(upload)
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[prompt_text, active_file],
            )
        except Exception as api_error:
            logging.error("Błąd podczas transkrypcji Gemini dla modelu '%s': %s", self.model_name, api_error)
            raise GeminiTranscriptionError(str(api_error)) from api_error
        finally:
            file_name = getattr(active_file, "name", None) if 'active_file' in locals() else None
            if not file_name:
                file_name = getattr(upload, "name", None)
            self._cleanup_file(file_name)

        text = self._extract_text(response)
        if not text:
            logging.warning("Gemini zwróciło pustą transkrypcję dla modelu '%s'.", self.model_name)
        return text


def _generate_segments_from_words(
    word_entries: List[Tuple[float, float, str]],
    fallback_text: str,
    fallback_duration: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Buduje listę segmentów w formacie zbliżonym do openai-whisper na podstawie listy słów AssemblyAI.
    Zwraca krotkę (segments, max_end_time).
    """
    segments: List[Dict[str, Any]] = []
    max_end = float(fallback_duration) if fallback_duration else 0.0

    if not word_entries:
        cleaned_text = (fallback_text or "").strip()
        if cleaned_text:
            end_value = max_end if max_end > 0 else 0.0
            segments.append({"start": 0.0, "end": end_value, "text": cleaned_text})
        return segments, max_end

    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_tokens: List[str] = []

    max_segment_duration = 15.0
    max_segment_words = 40

    for start, end, token in word_entries:
        max_end = max(max_end, end)
        if current_start is None:
            current_start = start
        current_end = end
        current_tokens.append(token)

        duration = (current_end - current_start) if current_end is not None and current_start is not None else 0.0
        sentence_break = token.endswith((".", "?", "!", "…"))
        should_split = (
            len(current_tokens) >= max_segment_words
            or (sentence_break and duration >= 1.5)
            or duration >= max_segment_duration
        )

        if should_split:
            segment_text = " ".join(current_tokens).strip()
            if segment_text:
                segments.append(
                    {
                        "start": float(current_start),
                        "end": float(current_end if current_end is not None else current_start),
                        "text": segment_text,
                    }
                )
            current_start = None
            current_end = None
            current_tokens = []

    if current_tokens:
        segment_text = " ".join(current_tokens).strip()
        if segment_text:
            start_time = float(current_start) if current_start is not None else 0.0
            end_time = float(current_end) if current_end is not None else max_end
            if end_time < start_time:
                end_time = start_time
            segments.append({"start": start_time, "end": end_time, "text": segment_text})

    if fallback_duration:
        max_end = max(max_end, float(fallback_duration))

    if not segments:
        cleaned_text = (fallback_text or "").strip()
        if cleaned_text:
            segments.append({"start": 0.0, "end": max_end, "text": cleaned_text})

    return segments, max_end


class AssemblyAITranscriber:
    """
    Wrapper dla SDK AssemblyAI, zwracający strukturę kompatybilną z resztą pipeline'u.
    """

    uses_remote_api = True
    provider = "assemblyai"

    def __init__(self, model_name: str, api_key: str):
        if aai is None:
            raise ImportError("Pakiet assemblyai nie jest zainstalowany.")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY jest wymagany do transkrypcji AssemblyAI.")

        self.model_name = (model_name or DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL).strip()
        self._api_key = api_key
        normalized_for_enum = self.model_name.replace("-", "_")
        self._speech_model_enum = _ASSEMBLYAI_MODEL_ENUM_MAP.get(self.model_name.lower()) or getattr(aai.SpeechModel, normalized_for_enum, None) if aai and hasattr(aai, "SpeechModel") else None

    def _build_config(self) -> "aai.TranscriptionConfig":
        config_kwargs: Dict[str, Any] = {
            "speaker_labels": False,
            "punctuate": True,
            "format_text": True,
            "language_detection": True,
        }
        if self.model_name:
            if self._speech_model_enum is not None:
                config_kwargs["speech_model"] = self._speech_model_enum
            else:
                config_kwargs["speech_model"] = self.model_name
        return aai.TranscriptionConfig(**config_kwargs)

    def transcribe_file(self, file_path: str, fallback_duration: Optional[float] = None) -> Dict[str, Any]:
        try:
            aai.settings.api_key = self._api_key
            transcriber = aai.Transcriber()
            config = self._build_config()
            transcript = transcriber.transcribe(str(file_path), config=config)
        except Exception as exc:  # pragma: no cover - SDK rzuca różne wyjątki
            raise AssemblyAITranscriptionError(str(exc)) from exc

        raw_status = getattr(transcript, "status", "")
        status = str(raw_status).lower()
        if status.endswith(".completed"):
            status = "completed"
        error_message = getattr(transcript, "error", None)
        if error_message:
            raise AssemblyAITranscriptionError(str(error_message))
        if status and all(keyword not in status for keyword in ("completed", "done")):
            raise AssemblyAITranscriptionError(f"AssemblyAI zwróciło status '{status}'.")

        full_text = getattr(transcript, "text", "") or ""
        raw_words = getattr(transcript, "words", None) or []
        word_entries: List[Tuple[float, float, str]] = []
        for word in raw_words:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            token = getattr(word, "text", "")
            if start is None or end is None:
                continue
            try:
                start_sec = float(start) / 1000.0
                end_sec = float(end) / 1000.0
            except (TypeError, ValueError):
                continue
            if end_sec < start_sec:
                end_sec = start_sec
            word_entries.append((start_sec, end_sec, token))

        segments, max_end = _generate_segments_from_words(word_entries, full_text, fallback_duration)
        return {
            "text": full_text,
            "segments": segments,
            "duration": max_end,
        }

def _resolve_openai_whisper_model(requested_alias: Optional[str]) -> Tuple[str, str]:
    """
    Mapuje alias modelu z UI na konkretny model OpenAI Whisper API.
    Zwraca krotkę (resolved_model_name, normalized_alias).
    """
    mapping_source = OPENAI_WHISPER_MODEL_MAP or _DEFAULT_OPENAI_WHISPER_MODEL_MAP
    mapping = {}
    for key, value in mapping_source.items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip().lower()
        if not normalized_key:
            continue
        mapping[normalized_key] = value if value is None or isinstance(value, str) else None

    default_model = (
        OPENAI_WHISPER_MODEL
        or mapping.get("default")
        or "gpt-4o-mini-transcribe"
    )

    alias_input = (requested_alias or "").strip()
    original_alias_lower = alias_input.lower()
    canonical_alias = original_alias_lower

    if canonical_alias in WHISPER_LEGACY_ALIAS_REDIRECTS:
        redirected = WHISPER_LEGACY_ALIAS_REDIRECTS[canonical_alias]
        logging.info(
            "Legacy Whisper alias '%s' mapped to '%s'.",
            alias_input,
            redirected,
        )
        alias_input = redirected
        canonical_alias = redirected.lower()

    alias_for_record = alias_input or "default"

    if not alias_input:
        logging.debug(
            "No Whisper model alias provided; using default OpenAI model '%s'.",
            default_model,
        )
        return default_model, alias_for_record

    if canonical_alias in mapping:
        mapped_value = mapping[canonical_alias]
        if mapped_value:
            if mapped_value.lower() != canonical_alias:
                logging.info(
                    "Mapped Whisper alias '%s' to OpenAI model '%s'.",
                    alias_input,
                    mapped_value,
                )
            return mapped_value, alias_for_record
        logging.debug(
            "Whisper alias '%s' maps to default OpenAI model '%s'.",
            alias_input,
            default_model,
        )
        return default_model, alias_for_record

    if canonical_alias.startswith("gpt-"):
        logging.info(
            "Using explicit OpenAI Whisper model '%s' requested by alias.",
            alias_input,
        )
        return alias_input, alias_for_record

    logging.warning(
        "Whisper alias '%s' not recognized for OpenAI API usage. Falling back to default '%s'.",
        alias_input,
        default_model,
    )
    return default_model, alias_for_record


def get_gemini_transcriber(model_name: Optional[str], api_key_override: Optional[str] = None) -> Optional[GeminiTranscriber]:
    """
    Zwraca (i buforuje) obiekt GeminiTranscriber dla wskazanego modelu.
    """
    normalized = (model_name or DEFAULT_GEMINI_TRANSCRIPTION_MODEL).strip()
    if not normalized:
        normalized = DEFAULT_GEMINI_TRANSCRIPTION_MODEL
    effective_key = (api_key_override or GEMINI_TRANSCRIBE_API_KEY or "").strip()
    if not effective_key:
        logging.error("Gemini transcription API key (GOOGLE_TRANSCRIBE_API_KEY) is not configured.")
        return None
    if api_key_override:
        try:
            return GeminiTranscriber(
                normalized,
                effective_key,
                fallback_api_key=GOOGLE_API_KEY if GOOGLE_API_KEY else None,
            )
        except Exception as creation_error:
            logging.error(
                "Failed to initialize Gemini transcriber '%s' with provided override key: %s",
                normalized,
                creation_error,
            )
            return None

    transcriber = _gemini_transcribers.get(normalized)
    if transcriber is None:
        try:
            transcriber = GeminiTranscriber(
                normalized,
                effective_key,
                fallback_api_key=GOOGLE_API_KEY,
            )
            _gemini_transcribers[normalized] = transcriber
        except Exception as creation_error:
            logging.error("Failed to initialize Gemini transcriber '%s': %s", normalized, creation_error)
            return None
    return transcriber


def get_assemblyai_transcriber(
    model_name: Optional[str],
    api_key_override: Optional[str] = None,
) -> Optional[AssemblyAITranscriber]:
    """
    Zwraca (i buforuje) obiekt AssemblyAITranscriber dla wskazanego modelu.
    """
    normalized = (model_name or DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL).strip()
    if not normalized:
        normalized = DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL
    normalized = normalized.lower()

    if aai is None:
        logging.error("Pakiet assemblyai nie jest zainstalowany. Nie można wykonać transkrypcji AssemblyAI.")
        return None

    effective_key = (api_key_override or ASSEMBLYAI_API_KEY or "").strip()
    if not effective_key:
        logging.error("Brak konfiguracji ASSEMBLYAI_API_KEY. Nie można wykonać transkrypcji AssemblyAI.")
        return None

    if api_key_override:
        try:
            return AssemblyAITranscriber(normalized, effective_key)
        except Exception as creation_error:
            logging.error(
                "Failed to initialize AssemblyAI transcriber '%s' with override key: %s",
                normalized,
                creation_error,
            )
            return None

    transcriber = _assemblyai_transcribers.get(normalized)
    if transcriber is None:
        try:
            transcriber = AssemblyAITranscriber(normalized, effective_key)
            _assemblyai_transcribers[normalized] = transcriber
        except Exception as creation_error:
            logging.error("Failed to initialize AssemblyAI transcriber '%s': %s", normalized, creation_error)
            return None

    return transcriber


def get_transcription_provider_catalog() -> List[Dict[str, Any]]:
    """
    Zwraca kopię katalogu dostawców i modeli transkrypcji z konfiguracji.
    """
    try:
        # shallow copy to prevent modifications
        return [dict(provider) for provider in TRANSCRIPTION_PROVIDER_CATALOG]
    except Exception as catalog_error:
        logging.error("Failed to clone transcription provider catalog: %s", catalog_error)
        return []


def resolve_transcription_choice(
    provider: Optional[str],
    model: Optional[str],
) -> Tuple[str, Optional[str]]:
    """
    Normalizuje wybór dostawcy i modelu transkrypcji, korzystając z katalogu.
    Zwraca krotkę (provider_id, model_id).
    """
    normalized_provider = (provider or DEFAULT_TRANSCRIPTION_PROVIDER).strip().lower()
    provider_entries = get_transcription_provider_catalog()
    fallback_provider = next(
        (entry for entry in provider_entries if entry.get("provider_id") == normalized_provider),
        None,
    )
    if fallback_provider is None and provider_entries:
        logging.warning("Unknown transcription provider '%s'. Falling back to '%s'.", normalized_provider, DEFAULT_TRANSCRIPTION_PROVIDER)
        fallback_provider = next(
            (entry for entry in provider_entries if entry.get("provider_id") == DEFAULT_TRANSCRIPTION_PROVIDER),
            provider_entries[0],
        )
        normalized_provider = fallback_provider.get("provider_id", DEFAULT_TRANSCRIPTION_PROVIDER)
    elif fallback_provider is None:
        # Catalog empty; return defaults
        return normalized_provider, model

    available_models = fallback_provider.get("models", [])
    normalized_model = model.strip() if isinstance(model, str) else None

    if normalized_model:
        normalized_model = normalized_model.lower()
        if any(m.get("id") == normalized_model for m in available_models):
            return normalized_provider, normalized_model

    default_model = fallback_provider.get("default_model")
    if default_model:
        return normalized_provider, str(default_model)

    if available_models:
        return normalized_provider, str(available_models[0].get("id"))

    return normalized_provider, normalized_model


def build_transcription_metadata(
    provider: Optional[str],
    model: Optional[str],
) -> Dict[str, str]:
    """
    Przygotowuje ustandaryzowane informacje o dostawcy i modelu transkrypcji
    do celów raportowych/metadanych.
    """
    provider_id, normalized_model = resolve_transcription_choice(provider, model)
    provider_entries = get_transcription_provider_catalog()
    provider_entry = next((entry for entry in provider_entries if entry.get("provider_id") == provider_id), None)
    provider_label = (provider_entry.get("label") if provider_entry else None) or provider_id.title()

    resolved_model = (normalized_model or (provider_entry.get("default_model") if provider_entry else None) or "") if provider_entry else (normalized_model or "")
    model_display = resolved_model

    if provider_entry and resolved_model:
        available_models = provider_entry.get("models", [])
        resolved_model_lower = resolved_model.strip().lower()
        for candidate in available_models:
            candidate_id = str(candidate.get("id", "")).strip().lower()
            if candidate_id == resolved_model_lower:
                candidate_metadata = candidate.get("metadata") or {}
                model_display = (
                    candidate_metadata.get("openai_target")
                    or candidate.get("label")
                    or resolved_model
                )
                break

    transcription_model_display = model_display or provider_label or "Brak danych"
    if provider_label and model_display:
        if provider_label.lower() not in model_display.lower():
            transcription_model_display = f"{provider_label} – {model_display}"
        else:
            transcription_model_display = model_display
    elif provider_label and not transcription_model_display:
        transcription_model_display = provider_label
    elif not transcription_model_display:
        transcription_model_display = "Brak danych"

    return {
        "transcription_provider": provider_id,
        "transcription_provider_label": provider_label,
        "transcription_model": resolved_model,
        "transcription_model_display": transcription_model_display,
    }


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


async def save_to_srt(outputs, srt_file_path):
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
        f = None # Inicjalizacja uchwytu pliku
        try:
            f = await asyncio.to_thread(open, srt_file_path, "w", encoding="utf-8")
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

                # Write SRT entry asynchonously
                await asyncio.to_thread(f.write, f"{segment_index}\n")
                await asyncio.to_thread(f.write, f"{start_time_str} --> {end_time_str}\n")
                await asyncio.to_thread(f.write, f"{text}\n\n")
                segment_index += 1
        finally:
            if f:
                await asyncio.to_thread(f.close)

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

async def get_diarization_models(
    requested_whisper_model_size: Optional[str] = None,
    load_whisper: bool = True,
    openai_api_key_override: Optional[str] = None,
):
    """
    Loads Whisper (openai), Pyannote diarization, and Pyannote embedding models based on config settings.
    Uses caching to load models only once.
    """
    global _whisper_models, _diarization_pipeline, _embedding_model
    using_remote = USE_OPENAI_WHISPER_API if load_whisper else False
    resolved_alias: Optional[str] = None

    if load_whisper:
        if using_remote:
            resolved_model_name, resolved_alias = _resolve_openai_whisper_model(requested_whisper_model_size)
            display_model_name = resolved_model_name
            whisper_cache_key = f"openai_api::{resolved_model_name.lower()}"
        else:
            display_model_name = (requested_whisper_model_size or WHISPER_MODEL_SIZE or "").strip()
            if not display_model_name:
                display_model_name = WHISPER_MODEL_SIZE
            whisper_cache_key = display_model_name.lower()
    else:
        display_model_name = (requested_whisper_model_size or WHISPER_MODEL_SIZE or "").strip() or "n/a"
        whisper_cache_key = f"skip::{display_model_name}"

    cached_whisper_model = None
    if load_whisper and not openai_api_key_override:
        cached_whisper_model = _whisper_models.get(whisper_cache_key)
    if load_whisper:
        if (
            cached_whisper_model is not None
            and _diarization_pipeline is not None
            and _embedding_model is not None
        ):
            logging.info(
                "Returning cached models (Whisper: %s, Pyannote pipeline, SpeechBrain embedding).",
                display_model_name,
            )
            return cached_whisper_model, _diarization_pipeline, _embedding_model
    else:
        if _diarization_pipeline is not None and _embedding_model is not None:
            return None, _diarization_pipeline, _embedding_model

    logging.info("Loading diarization and embedding models (first time)...")
    logging.info(f"Using device: {DEVICE}")

    # Zwolnij pamięć GPU PRZED ZAŁADOWANIEM MODELI
    if torch.cuda.is_available():
        await asyncio.to_thread(torch.cuda.empty_cache)
        logging.info("Cleared CUDA cache before loading models.")

    if load_whisper and using_remote and cached_whisper_model is None:
        try:
            effective_api_key = (openai_api_key_override or OPENAI_API_KEY or "").strip()
            if not effective_api_key:
                raise ValueError("OPENAI_API_KEY is not configured.")
            logging.info("Initializing OpenAI Whisper API client (model: %s).", display_model_name)
            initialized_client = OpenAIWhisperAPI(
                model_name=display_model_name,
                api_key=effective_api_key,
                base_url=OPENAI_API_BASE,
                organization=OPENAI_ORG_ID,
                response_format=OPENAI_WHISPER_RESPONSE_FORMAT,
                language=OPENAI_WHISPER_LANGUAGE,
                temperature=OPENAI_WHISPER_TEMPERATURE,
                requested_alias=resolved_alias,
            )
            if openai_api_key_override:
                cached_whisper_model = initialized_client
            else:
                cached_whisper_model = initialized_client
                _whisper_models[whisper_cache_key] = initialized_client
            logging.info("OpenAI Whisper API client initialized successfully.")
        except Exception as e:
            logging.critical("Error initializing OpenAI Whisper API client: %s", e, exc_info=True)
            _whisper_models[whisper_cache_key] = None
            cached_whisper_model = None

    if load_whisper and not using_remote and whisper:
        if cached_whisper_model is None:
            try:
                logging.info(f"Loading OpenAI Whisper model: {display_model_name}")
                loaded_model = await asyncio.to_thread(
                    whisper.load_model,
                    display_model_name,
                    device=DEVICE,
                    download_root="./models"
                )
                _whisper_models[whisper_cache_key] = loaded_model
                cached_whisper_model = loaded_model
                logging.info("OpenAI Whisper model '%s' loaded successfully.", display_model_name)
            except Exception as e:
                logging.error(f"Error loading OpenAI Whisper model '{display_model_name}': {e}")
                logging.info("Attempting to load Whisper model '%s' on CPU...", display_model_name)
                try:
                    loaded_model = await asyncio.to_thread(
                        whisper.load_model,
                        display_model_name,
                        device="cpu",
                        download_root="./models"
                    )
                    _whisper_models[whisper_cache_key] = loaded_model
                    cached_whisper_model = loaded_model
                    logging.info(
                        "OpenAI Whisper model '%s' loaded successfully on CPU.",
                        display_model_name,
                    )
                except Exception as e_cpu:
                    logging.error(
                        "Error loading OpenAI Whisper model '%s' on CPU: %s",
                        display_model_name,
                        e_cpu,
                    )
                    _whisper_models[whisper_cache_key] = None
                    cached_whisper_model = None
    elif load_whisper and not using_remote and not whisper:
        logging.error("openai-whisper library not found. Cannot load local Whisper model.")

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
            _diarization_pipeline = await asyncio.to_thread(
                PyannotePipeline.from_pretrained,
                PYANNOTE_PIPELINE,
                **pipeline_args
            )
            _diarization_pipeline = _diarization_pipeline.to(torch.device(DEVICE))
            logging.info("Pyannote diarization pipeline loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Pyannote diarization pipeline: {e}")
            logging.info("Attempting to load Pyannote diarization pipeline on CPU...")
            try:
                _diarization_pipeline = await asyncio.to_thread(
                    PyannotePipeline.from_pretrained,
                    PYANNOTE_PIPELINE,
                    **pipeline_args
                )
                _diarization_pipeline = _diarization_pipeline.to(torch.device("cpu"))
                logging.info("Pyannote diarization pipeline loaded successfully on CPU.")
            except Exception as e_cpu:
                logging.error(f"Error loading Pyannote diarization pipeline on CPU: {e_cpu}")
                _diarization_pipeline = None

    # Load Pyannote embedding model using SpeechBrain's EncoderClassifier
    if EncoderClassifier:
        try:
            logging.info(f"Loading Pyannote embedding model (SpeechBrain/ECAPA-TDNN) using SpeechBrain EncoderClassifier on {DEVICE}...")
            _embedding_model = await asyncio.to_thread(
                EncoderClassifier.from_hparams,
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": DEVICE},
                # use_auth_token=HF_TOKEN # If this model requires specific auth (test_embedding showed it works without it for this model)
            )
            logging.info("SpeechBrain embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SpeechBrain embedding model on {DEVICE}: {e}")
            logging.info("Attempting to load SpeechBrain embedding model on CPU...")
            try:
                _embedding_model = await asyncio.to_thread(
                    EncoderClassifier.from_hparams,
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


    if (load_whisper and cached_whisper_model is None) or _diarization_pipeline is None or _embedding_model is None:
        logging.critical("CRITICAL: One or more required models failed to load. Diarization and speaker enrollment features may not work.")
        # Możesz tutaj rzucić wyjątek, jeśli uznasz, że brakujące modele uniemożliwiają działanie aplikacji
        # raise RuntimeError("Required models failed to load.")

    cached_model_to_return = _whisper_models.get(whisper_cache_key) if load_whisper else None

    if load_whisper and using_remote and cached_model_to_return is not None:
        try:
            cached_model_to_return.requested_alias = resolved_alias or "default"
            cached_model_to_return.resolved_model_name = display_model_name
        except Exception as alias_error:
            logging.debug("Could not update Whisper API client alias metadata: %s", alias_error)

    return cached_model_to_return, _diarization_pipeline, _embedding_model

_SPEAKER_SHARED_NAMESPACE = "__shared__"

def _speaker_cache_key(username: Optional[str]) -> str:
    return _sanitize_username(username, fallback="user") if username else _SPEAKER_SHARED_NAMESPACE

def _speaker_storage_path(username: Optional[str]) -> Path:
    base_path = Path(SPEAKER_ENROLLMENT_DIR)
    if username:
        return base_path / _sanitize_username(username, fallback="user")
    return base_path

def _clone_speaker_map(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: embedding for name, embedding in (data or {}).items()}

async def load_enrolled_speakers(username: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Loads enrolled speaker profiles for the specified user namespace (or the shared namespace when username is None).
    Returns a mapping of speaker_name -> embedding.
    """
    cache_key = _speaker_cache_key(username)
    speaker_dir = _speaker_storage_path(username)

    await asyncio.to_thread(speaker_dir.mkdir, parents=True, exist_ok=True)

    loaded: Dict[str, np.ndarray] = {}
    try:
        for entry in await asyncio.to_thread(speaker_dir.iterdir):
            if await asyncio.to_thread(entry.is_dir):
                profile_file = entry / "embedding.npy"
                if await asyncio.to_thread(profile_file.exists):
                    try:
                        embedding = await asyncio.to_thread(np.load, profile_file)
                        loaded[entry.name] = embedding
                        logging.info(f"Loaded speaker '{entry.name}' for namespace '{cache_key}'.")
                    except Exception as e:
                        logging.error(f"Error loading embedding for '{entry.name}' from {profile_file}: {e}")
                else:
                    logging.warning(f"No embedding.npy found for speaker '{entry.name}' in {entry.resolve()}")
    except FileNotFoundError:
        logging.info(f"Speaker directory {speaker_dir} not found. It will be created on demand.")

    _enrolled_speakers_profiles[cache_key] = loaded
    return _clone_speaker_map(loaded)

async def list_enrolled_speakers(username: Optional[str] = None) -> List[str]:
    """
    Returns a sorted list of enrolled speaker names for the given user namespace.
    """
    cache_key = _speaker_cache_key(username)
    if cache_key not in _enrolled_speakers_profiles:
        await load_enrolled_speakers(username)
    profiles = _enrolled_speakers_profiles.get(cache_key, {})
    return sorted(profiles.keys())


async def delete_speaker(speaker_name: str, username: Optional[str] = None) -> bool:
    """
    Deletes a speaker's enrollment profile (embedding file and directory) for the specified namespace.
    """
    cache_key = _speaker_cache_key(username)
    speaker_profile_dir = _speaker_storage_path(username) / speaker_name

    profiles = _enrolled_speakers_profiles.setdefault(cache_key, {})
    profiles.pop(speaker_name, None)

    if await asyncio.to_thread(speaker_profile_dir.exists):
        try:
            await asyncio.to_thread(shutil.rmtree, speaker_profile_dir)
            logging.info(f"Deleted speaker '{speaker_name}' from namespace '{cache_key}'.")
            return True
        except Exception as e:
            logging.error(f"❌ Error deleting speaker directory {speaker_profile_dir}: {e}")
            return False
    else:
        logging.info(f"Speaker directory for '{speaker_name}' not found at {speaker_profile_dir}.")
        return True

async def enroll_speaker_from_audio(
    audio_file_path: Path,
    human_name: str,
    username: Optional[str] = None
) -> bool:
    """
    Enrolls a speaker by generating an embedding from the provided audio file
    and saving it to the user-specific speaker enrollment directory.
    """
    logging.info(f"Enrolling speaker '{human_name}' for namespace '{_speaker_cache_key(username)}' from audio: {audio_file_path.name}")

    if _embedding_model is None:
        logging.error(f"❌ Pyannote embedding model not loaded. Cannot enroll speaker '{human_name}'.")
        return False

    try:
        signal, sample_rate = await asyncio.to_thread(torchaudio.load, str(audio_file_path))

        if sample_rate != 16000:
            logging.warning(f"Audio sample rate is {sample_rate}Hz. Resampling to 16000Hz for speaker '{human_name}'.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = await asyncio.to_thread(resampler, signal)

        if signal.shape[0] > 1:
            logging.info(f"Converting stereo audio to mono for speaker '{human_name}'.")
            signal = torch.mean(signal, dim=0, keepdim=True)

        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0)

        signal = signal.to(DEVICE)

        embedding = await asyncio.to_thread(lambda: _embedding_model.encode_batch(signal.unsqueeze(0)).squeeze(0).cpu().numpy())

        if embedding is None:
            logging.error(f"❌ Failed to generate embedding for speaker '{human_name}'.")
            return False

        speaker_profile_dir = _speaker_storage_path(username) / human_name
        await asyncio.to_thread(speaker_profile_dir.mkdir, parents=True, exist_ok=True)
        embedding_file_path = speaker_profile_dir / "embedding.npy"
        await asyncio.to_thread(np.save, embedding_file_path, embedding)
        logging.info(f"Saved embedding for '{human_name}' to {embedding_file_path.resolve()}")

        cache_key = _speaker_cache_key(username)
        profiles = _enrolled_speakers_profiles.setdefault(cache_key, {})
        profiles[human_name] = embedding

        return True

    except Exception as e:
        logging.error(f"❌ Error during speaker enrollment for '{human_name}': {e}", exc_info=True)
        return False


async def identify_speaker(audio_segment_path: Path, username: Optional[str] = None) -> str:
    """
    Identifies the speaker of the given audio segment by comparing the embedding
    with those of enrolled speakers for the specified namespace.
    """
    if _embedding_model is None:
        logging.error("❌ Embedding model not loaded. Cannot identify speaker.")
        return "SPEAKER_UNKNOWN"

    cache_key = _speaker_cache_key(username)
    if cache_key not in _enrolled_speakers_profiles:
        await load_enrolled_speakers(username)

    profiles = _enrolled_speakers_profiles.get(cache_key, {})
    if not profiles:
        logging.info(f"No enrolled speakers available for namespace '{cache_key}'. Returning SPEAKER_UNKNOWN.")
        return "SPEAKER_UNKNOWN"

    try:
        signal, sample_rate = await asyncio.to_thread(torchaudio.load, str(audio_segment_path))

        if sample_rate != 16000:
            logging.warning(f"Resampling audio segment from {sample_rate}Hz to 16000Hz for identification.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = await asyncio.to_thread(resampler, signal)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0)

        signal = signal.to(DEVICE)
        segment_embedding = await asyncio.to_thread(lambda: _embedding_model.encode_batch(signal.unsqueeze(0)).squeeze(0).cpu().numpy())

        if segment_embedding is None or segment_embedding.size == 0:
            logging.warning(f"Could not generate embedding for segment {audio_segment_path.name}. Returning UNKNOWN.")
            return "SPEAKER_UNKNOWN"

        best_match_name = "SPEAKER_UNKNOWN"
        highest_similarity = PYANNOTE_IDENTIFICATION_THRESHOLD

        for enrolled_name, enrolled_embedding in profiles.items():
            if enrolled_embedding is None:
                continue
            if enrolled_embedding.ndim > 1:
                enrolled_embedding = enrolled_embedding.flatten()
            comparison_embedding = segment_embedding.flatten() if segment_embedding.ndim > 1 else segment_embedding

            similarity = 1 - cosine(enrolled_embedding, comparison_embedding)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = enrolled_name

        if best_match_name != "SPEAKER_UNKNOWN":
            logging.info(f"[{cache_key}] Identified speaker as '{best_match_name}' with similarity: {highest_similarity:.2f}")
        else:
            logging.info(f"[{cache_key}] No speaker identified (max similarity: {highest_similarity:.2f}). Returning UNKNOWN.")

        return best_match_name

    except Exception as e:
        logging.error(f"❌ Error during speaker identification for {audio_segment_path.name}: {e}", exc_info=True)
        return "SPEAKER_UNKNOWN"


async def diarize_text(
    transcription_result: Dict[str, Any],
    diarization_result: Any,
    original_audio_path: Path,
    username: Optional[str] = None,
    fallback_audio_duration: Optional[float] = None,
) -> List[Tuple[Any, str, str]]:
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

        if not whisper_segments:
            fallback_text = (transcription_result.get('text') or "").strip()
            if fallback_text:
                first_turn = speaker_turns[0]
                fallback_start = float(first_turn.get('start') or 0.0)
                fallback_end = float(first_turn.get('end') or fallback_start)
                if fallback_audio_duration is not None and fallback_end < fallback_start:
                    fallback_end = max(fallback_start, float(fallback_audio_duration))
                elif fallback_end < fallback_start:
                    fallback_end = fallback_start
                segment_time_info = type('obj', (object,), {'start': fallback_start, 'end': fallback_end})()
                final_aligned_segments.append((segment_time_info, first_turn.get('speaker') or "SPEAKER_UNKNOWN", fallback_text))
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
        waveform, sample_rate = await asyncio.to_thread(torchaudio.load, original_audio_path)
        # logging.debug(f"Original audio loaded for diarization: {original_audio_path.name}, sample_rate: {sample_rate}, shape: {waveform.shape}")

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
        # logging.debug(f"(Pass 1): Segment {i} ({start_time:.2f}-{end_time:.2f}s) Pyannote label: {diarized_speaker_label}")

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
                    await asyncio.to_thread(temp_segment_dir.mkdir, exist_ok=True)
                    temp_audio_path = temp_segment_dir / f"segment_pass1_{i}_{start_time}-{end_time}.wav"
                    await asyncio.to_thread(torchaudio.save, str(temp_audio_path), audio_segment_tensor, sample_rate)

                    identified_name = await identify_speaker(temp_audio_path, username=username)
                    # logging.debug(f"(Pass 1): Segment {i} - Identified name: {identified_name}")

                    if identified_name != "SPEAKER_UNKNOWN":
                        speaker_label_to_human_name[diarized_speaker_label] = identified_name
                        # logging.debug(f"(Pass 1): Mapped {diarized_speaker_label} to {identified_name}")
                    
                    await asyncio.to_thread(os.remove, temp_audio_path) # Asynchroniczne usuwanie pliku
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

async def save_minutes_to_file(minutes_content_string: str, unique_collection_name: str, target_date_iso: str) -> str | None:
    """Saves the generated meeting minutes to a text file."""
    # logging.debug(f"save_minutes_to_file called with unique_collection_name: {unique_collection_name}, date: {target_date_iso}")
    
    minutes_file_path = None
    # metadata_file_path = None # Removed as JSON is no longer saved here

    try:
        formatted_date_for_folder = target_date_iso
        base_output_root = Path(OUTPUT_DIR_MINUTES)
        output_dir = base_output_root / formatted_date_for_folder
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True) # Asynchroniczne tworzenie katalogu

        safe_base_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in unique_collection_name])
        safe_base_name = safe_base_name.strip('_-')

        final_file_base_name = f"{safe_base_name}"
        
        minutes_file_path = output_dir / f"{final_file_base_name}_summary.txt"

        f = None # Inicjalizacja uchwytu pliku
        try:
            f = await asyncio.to_thread(open, minutes_file_path, "w", encoding="utf-8")
            await asyncio.to_thread(f.write, minutes_content_string) # Write the pre-formatted string directly
        finally:
            if f:
                await asyncio.to_thread(f.close)
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


async def get_current_ram_usage_gb() -> float:
    """
    Pobiera bieżące zużycie RAM (pamięci głównej) przez proces Pythona w GB.
    Dostosowane dla systemów Windows i Linux.
    """
    try:
        if platform.system() == "Windows":
            command = f'wmic process where "ProcessId={os.getpid()}" get WorkingSetSize /value'
            result = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                output = result.stdout
                for line in output.splitlines():
                    if "WorkingSetSize" in line:
                        memory_bytes = int(line.split('=')[1])
                        memory_gb = memory_bytes / (1024**3)
                        return memory_gb
            else:
                logging.error(f"WMIC command failed with return code {result.returncode}: {result.stderr}")
        else: # Zakładamy Linux/Unix
            command = f'ps -o %mem= -p {os.getpid()}'
            result = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                ram_percent_str = result.stdout.strip()
                if ram_percent_str:
                    # ps -o %mem= zwraca procent pamięci, nie bezpośrednio MiB
                    # Aby to przeliczyć na GB, potrzebujemy całkowitej pamięci RAM
                    # Co jest bardziej złożone. Na razie, możemy zwrócić procent lub uprościć.
                    # Poniżej, próbujemy uzyskać procent i przekonwertować na fikcyjne GB (do poprawy w przyszłości)
                    ram_percent = float(ram_percent_str)
                    # Ta konwersja jest błędna. %mem to procent z CAŁKOWITEJ pamięci RAM, nie proces.
                    # Aby uzyskać GB, potrzebowalibyśmy totalnej pamięci RAM systemu. 
                    # Na razie zwracamy procent, ale logujemy jako GB, co jest mylące.
                    # Zmieniamy na zwracanie %mem jako %.
                    logging.warning("RAM usage for Linux/Unix currently returns %mem as GB. This is inaccurate.")
                    # Zwracamy po prostu procent, ale typ funkcji to float (GB). To będzie wymagało przemyślenia.
                    # Najprościej na teraz: szacunkowe przeliczenie lub po prostu zwróć 0.0 z ostrzeżeniem.
                    # Zamiast tego, dla Linuxa, spróbujemy użyć `free -m` lub podobnego, aby uzyskać dokładniejszą wartość.
                    logging.warning("Accurate RAM usage for Linux/Unix is complex to get for a single process. Returning 0.0.")
                    return 0.0 # Tymczasowo zwróć 0.0 dla Linuxa
            else:
                logging.error(f"ps command failed with return code {result.returncode}: {result.stderr}")

    except FileNotFoundError:
        logging.warning("Required command (wmic or ps) not found. RAM usage cannot be retrieved.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e.cmd} with return code {e.returncode}: {e.stderr}")
    except ValueError:
        logging.error(f"Could not parse RAM usage from command output: {result.stdout if 'result' in locals() else 'N/A'}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting RAM usage: {e}")
    return 0.0

async def get_current_vram_usage_gb():
    """
    Pobiera bieżące zużycie VRAM (pamięci karty graficznej) w GB,
    używając nvidia-smi (tylko dla kart NVIDIA).
    """
    try:
        # Komenda do pobrania zużycia pamięci używanej przez GPU w MiB
        command = 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
        result = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, shell=True)
        
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

async def get_file_creation_date(file_path):
    """
    Zwraca datę utworzenia pliku w formacie YYYY-MM-DD."""
    timestamp = await asyncio.to_thread(os.path.getctime, file_path) # Asynchroniczne pobieranie czasu utworzenia
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

async def get_relevant_date_for_file(file_path):
    """
    Próbuje wyciągnąć datę z nazwy pliku, a jako fallback używa daty utworzenia.
    Zwraca datę w formacie YYYY-MM-DD.
    """
    filename = Path(file_path).name
    date_from_name = extract_date_from_filename(filename)
    if date_from_name:
        return date_from_name
    return await get_file_creation_date(file_path)

# ===============================================
# Functions for Prompt Management
# ===============================================

PROMPT_DEFAULTS_PATH = Path(PROMPT_SETTINGS_FILE)
_CONFIGURED_USER_PROMPT_DIR = getattr(config, "PROMPT_USER_SETTINGS_DIR", None)
PROMPT_USER_SETTINGS_DIR = Path(_CONFIGURED_USER_PROMPT_DIR) if _CONFIGURED_USER_PROMPT_DIR else PROMPT_DEFAULTS_PATH.parent / "prompt-settings"

_PROMPT_DEFAULT_CACHE_KEY = "__defaults__"
_prompt_cache: Dict[str, Dict[str, Dict[str, str]]] = {}

def _clone_prompt_map(data: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Creates a shallow copy of the prompt mapping while ensuring values are dictionaries.
    """
    return {
        name: dict(prompts)
        for name, prompts in (data or {}).items()
        if isinstance(prompts, dict)
    }

def _prompt_cache_key_for_user(username: str) -> str:
    return f"user::{username}"

def _user_prompt_file_path(username: str) -> Path:
    sanitized = _sanitize_username(username)
    return PROMPT_USER_SETTINGS_DIR / f"{sanitized}.json"

async def _load_default_prompts_from_disk() -> Dict[str, Dict[str, str]]:
    config_path = PROMPT_DEFAULTS_PATH
    if not await asyncio.to_thread(config_path.exists):
        logging.info(f"Prompt settings file not found: {config_path}. Returning empty configuration.")
        return {}
    try:
        f = None
        try:
            f = await asyncio.to_thread(open, config_path, "r", encoding="utf-8")
            data = await asyncio.to_thread(json.load, f)
        finally:
            if f:
                await asyncio.to_thread(f.close)

        is_old_flat_format = False
        if isinstance(data, dict):
            if data and all(isinstance(v, str) for v in data.values()):
                is_old_flat_format = True
            elif data and any(not isinstance(v, dict) for v in data.values()):
                is_old_flat_format = True
        else:
            is_old_flat_format = True

        if is_old_flat_format:
            logging.info("Detected old flat prompt configuration format. Migrating to nested format.")
            migrated_data = {"Domyślny Zestaw": data if isinstance(data, dict) else {}}
            try:
                wf = None
                try:
                    wf = await asyncio.to_thread(open, config_path, "w", encoding="utf-8")
                    await asyncio.to_thread(json.dump, migrated_data, wf, ensure_ascii=False, indent=4)
                finally:
                    if wf:
                        await asyncio.to_thread(wf.close)
                logging.info(f"Successfully migrated and saved prompt settings to new format: {config_path}")
            except Exception as save_e:
                logging.error(f"Error saving migrated prompt settings to {config_path}: {save_e}")
            return _clone_prompt_map(migrated_data)

        return _clone_prompt_map(data)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding prompt settings file {config_path}: {e}. Returning empty configuration.")
        return {}
    except Exception as e:
        logging.error(f"Error loading prompt settings from {config_path}: {e}")
        return {}

async def _load_default_prompts() -> Dict[str, Dict[str, str]]:
    cached = _prompt_cache.get(_PROMPT_DEFAULT_CACHE_KEY)
    if cached is not None:
        return _clone_prompt_map(cached)

    data = await _load_default_prompts_from_disk()
    _prompt_cache[_PROMPT_DEFAULT_CACHE_KEY] = _clone_prompt_map(data)
    return _clone_prompt_map(data)

async def _load_user_prompts_from_disk(username: str) -> Dict[str, Dict[str, str]]:
    user_path = _user_prompt_file_path(username)
    if not await asyncio.to_thread(user_path.exists):
        return {}
    try:
        f = None
        try:
            f = await asyncio.to_thread(open, user_path, "r", encoding="utf-8")
            data = await asyncio.to_thread(json.load, f)
        finally:
            if f:
                await asyncio.to_thread(f.close)
        if not isinstance(data, dict):
            logging.warning(f"User prompt file {user_path} contained invalid data structure. Resetting to empty.")
            return {}
        return _clone_prompt_map(data)
    except Exception as e:
        logging.error(f"Error loading user prompt settings from {user_path}: {e}")
        return {}

async def _load_user_prompts(username: str) -> Dict[str, Dict[str, str]]:
    cache_key = _prompt_cache_key_for_user(username)
    cached = _prompt_cache.get(cache_key)
    if cached is not None:
        return _clone_prompt_map(cached)

    data = await _load_user_prompts_from_disk(username)
    _prompt_cache[cache_key] = _clone_prompt_map(data)
    return _clone_prompt_map(data)

async def _write_prompts_file(path: Path, data: Dict[str, Dict[str, str]]) -> None:
    if data:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        f = None
        try:
            f = await asyncio.to_thread(open, path, "w", encoding="utf-8")
            await asyncio.to_thread(json.dump, data, f, ensure_ascii=False, indent=4)
        finally:
            if f:
                await asyncio.to_thread(f.close)
    else:
        if await asyncio.to_thread(path.exists):
            await asyncio.to_thread(path.unlink)

async def load_prompt_config(username: Optional[str] = None, include_defaults: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Returns prompt configurations merged for the specified user.
    When include_defaults is True, the returned dictionary contains both default
    prompt sets and user-specific sets (user sets override defaults with the same key).
    """
    result: Dict[str, Dict[str, str]] = {}

    if include_defaults:
        defaults = await _load_default_prompts()
        result.update(_clone_prompt_map(defaults))

    if username:
        user_prompts = await _load_user_prompts(username)
        for name, prompts in user_prompts.items():
            result[name] = dict(prompts)

    return result

async def get_default_prompt_names() -> List[str]:
    defaults = await _load_default_prompts()
    return list(defaults.keys())

async def save_prompt_config(prompt_name: str, prompts: Dict[str, str], username: Optional[str] = None) -> bool:
    """
    Saves (or updates) a prompt set. When username is provided, the prompt set is
    stored in the user-specific configuration; otherwise it updates defaults.
    """
    if not prompts:
        logging.warning(f"Attempted to save an empty prompt set '{prompt_name}'. Operation aborted.")
        return False

    if username:
        user_prompts = await _load_user_prompts(username)
        user_prompts[prompt_name] = prompts
        user_path = _user_prompt_file_path(username)
        await _write_prompts_file(user_path, user_prompts)
        _prompt_cache[_prompt_cache_key_for_user(username)] = _clone_prompt_map(user_prompts)
        return True

    defaults = await _load_default_prompts()
    defaults[prompt_name] = prompts
    await _write_prompts_file(PROMPT_DEFAULTS_PATH, defaults)
    _prompt_cache[_PROMPT_DEFAULT_CACHE_KEY] = _clone_prompt_map(defaults)
    return True

async def delete_prompt_config(prompt_name: str, username: Optional[str] = None) -> bool:
    """
    Deletes a prompt set owned by the specified user. Default prompt sets cannot be deleted.
    """
    if not username:
        logging.warning(f"Attempted to delete prompt set '{prompt_name}' without specifying a user.")
        return False

    user_prompts = await _load_user_prompts(username)
    if prompt_name not in user_prompts:
        logging.info(f"Prompt set '{prompt_name}' not found in user prompt storage for '{username}'.")
        return False

    del user_prompts[prompt_name]
    user_path = _user_prompt_file_path(username)
    await _write_prompts_file(user_path, user_prompts)

    cache_key = _prompt_cache_key_for_user(username)
    if user_prompts:
        _prompt_cache[cache_key] = _clone_prompt_map(user_prompts)
    else:
        _prompt_cache.pop(cache_key, None)
    return True

# ================================================
# Main execution block (for testing utils.py directly)
# ================================================

async def _test_utils_functions():
    # logging.basicConfig(level=logging.INFO) # Configure logging for test output - USUNIĘTO

    # --- Test SRT formatting ---
    # print("\n--- Testing format_timestamp ---") # USUNIĘTO
    # print(f"0 seconds: {format_timestamp(0)}") # USUNIĘTO
    # print(f"5.123 seconds: {format_timestamp(5.123)}") # USUNIĘTO
    # print(f"65.999 seconds: {format_timestamp(65.999)}") # USUNIĘTO
    # print(f"3670.001 seconds: {format_timestamp(3670.001)}") # USUNIĘTO
    # print(f"None input: {format_timestamp(None)}") # USUNIĘTO
    # print(f"Negative input: {format_timestamp(-10)}") # USUNIĘTO

    # --- Test SRT saving (mock data) ---
    # print("\n--- Testing save_to_srt (mock data) ---") # USUNIĘTO
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
    await save_to_srt(mock_outputs_srt, test_srt_path)
    if await asyncio.to_thread(os.path.exists, test_srt_path):
        # print(f"Check '{test_srt_path}' for SRT output.") # USUNIĘTO
        await asyncio.to_thread(os.remove, test_srt_path) # Clean up test file asynchonously
    else:
        # print("SRT file saving test failed.") # USUNIĘTO
        pass

    # --- Test Diarization Placeholder ---
    # print("\n--- Testing diarize_text placeholder ---") # USUNIĘTO
    mock_whisper_diar = {
        'text': 'Hello Speaker 1. How are you Speaker 2?',
        'segments': [
            {'start': 0.5, 'end': 2.1, 'text': ' Hello Speaker 1.'},
            {'start': 2.5, 'end': 5.0, 'text': ' How are you Speaker 2?'}
        ]
    }
    # print("\nTesting with None diarization_result:") # USUNIĘTO
    aligned_none = await diarize_text(mock_whisper_diar, None, None)
    for seg in aligned_none:
        # print(f"  Time: {seg[0].start:.2f}-{seg[0].end:.2f}, Speaker: {seg[1]}, Text: {seg[2]}") # USUNIĘTO
        pass

    # print("\nNOTE: Cannot fully test Pyannote alignment path without a proper mock Annotation object.") # USUNIĘTO

    # --- Test File Saving (using OUTPUT_DIR_MINUTES) ---
    # print("\n--- Testing save_minutes_to_file ---") # USUNIĘTO
    mock_minutes = {
        "Główne Tematy Omówione": "Testowy temat 1, Testowy temat 2",
        "Kluczowe Decyzje": "Testowa decyzja 1, Testowa decyzja 2",
        "Elementy Działań": "Testowe działanie 1, Testowe działanie 2"
    }
    # Convert mock_minutes dict to a string before passing (e.g., JSON string or formatted string)
    mock_minutes_str = json.dumps(mock_minutes, ensure_ascii=False, indent=2) # NEW: Konwersja na string JSON
    mock_job_id = "test_job_123"
    await save_minutes_to_file(mock_minutes_str, mock_job_id, "2023-10-27") # Dodaj brakujący argument target_date_iso
    # print(f"  Minutes saved for job: {mock_job_id}") # USUNIĘTO

    # --- Test Asynchronous RAM/VRAM Usage ---
    # print("\n--- Asynchronous Test for get_current_ram_usage_gb ---") # USUNIĘTO
    ram_usage = await get_current_ram_usage_gb()
    # print(f"Current RAM Usage: {ram_usage} GB") # USUNIĘTO

    # print("\n--- Asynchronous Test for get_current_vram_usage_gb ---") # USUNIĘTO
    vram_usage = await get_current_vram_usage_gb()
    # print(f"Current VRAM Usage: {vram_usage} GB") # USUNIĘTO

if __name__ == "__main__":
    asyncio.run(_test_utils_functions())
