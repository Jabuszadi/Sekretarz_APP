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
import sys # NOWY IMPORT: Do przekierowania stdout
import re # NOWY IMPORT: Do parsowania wyjścia tqdm
from typing import List, Tuple, Optional, Callable, Any, Dict
from datetime import datetime
from pathlib import Path

# Import TranscriptionSegment z models.py
from models import TranscriptionSegment
# Import diarize_text i get_diarization_models z utils.py
from utils import (
    diarize_text,
    get_diarization_models,
    get_gemini_transcriber,
    get_assemblyai_transcriber,
    resolve_transcription_choice,
    identify_speaker,
    get_transcription_provider_catalog,
    GeminiTranscriptionError,
    AssemblyAITranscriptionError,
)
import agent_db # NEW: Import agent_db
import torchaudio # Import dla torchaudio

class TranscriptionProviderError(Exception):
    """Raised when the configured transcription provider cannot be used."""


# === Pomocnicze struktury do pracy z wynikami Modal ===
class SimpleSegment:
    __slots__ = ("start", "end")

    def __init__(self, start: float, end: float):
        self.start = float(start)
        self.end = float(end)


class RemoteDiarizationResult:
    """
    Minimalna implementacja interfejsu zwracanego przez Pyannote, oparta na danych z Modal.
    """

    def __init__(self, segments: List[Dict[str, Any]]):
        self._segments = segments
        self._labels = sorted(
            {
                segment.get("pyannote_label", "SPEAKER_UNKNOWN")
                for segment in segments
            }
        )

    def itertracks(self, yield_label: bool = False):
        for segment in self._segments:
            seg_obj = SimpleSegment(segment["start"], segment["end"])
            label = segment.get("pyannote_label") or "SPEAKER_UNKNOWN"
            if yield_label:
                yield seg_obj, None, label
            else:
                yield seg_obj, None

    def labels(self) -> List[str]:
        return list(self._labels)


def _normalize_modal_segments(
    raw_segments: List[Dict[str, Any]]
) -> Tuple[RemoteDiarizationResult, Dict[Tuple[float, float, str], str]]:
    normalized_segments: List[Dict[str, Any]] = []
    identified_map: Dict[Tuple[float, float, str], str] = {}

    for segment in raw_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        label = (segment.get("pyannote_label") or "SPEAKER_UNKNOWN").strip() or "SPEAKER_UNKNOWN"
        identified = (segment.get("identified_speaker") or label).strip() or label

        if end < start:
            end = start

        normalized_segments.append(
            {
                "start": start,
                "end": end,
                "pyannote_label": label,
                "identified_speaker": identified,
            }
        )
        identified_map[(round(start, 3), round(end, 3), label)] = identified

    return RemoteDiarizationResult(normalized_segments), identified_map


def _apply_remote_identifications(
    segments: List[Tuple[Any, str, str]],
    identified_map: Dict[Tuple[float, float, str], str],
) -> List[Tuple[Any, str, str]]:
    remapped_segments: List[Tuple[Any, str, str]] = []
    for segment, speaker, text in segments:
        key = (round(float(segment.start), 3), round(float(segment.end), 3), speaker)
        remapped_segments.append(
            (segment, identified_map.get(key, speaker), text)
        )
    return remapped_segments


async def _attempt_modal_diarization(
    audio_path: Path,
    username: Optional[str],
) -> Optional[Tuple[Callable[[str], RemoteDiarizationResult], Dict[Tuple[float, float, str], str]]]:
    if not config.USE_MODAL_DIARIZATION:
        return None

    try:
        from modal_client import run_modal_diarization, ModalClientError
    except ImportError as import_error:
        logging.error("Klient Modal nie jest dostępny: %s", import_error)
        return None

    try:
        payload = await run_modal_diarization(audio_path, username)
    except ModalClientError as modal_error:
        logging.error("Błąd podczas wywołania diarizacji Modal: %s", modal_error)
        return None
    except Exception as exc:
        logging.error("Nieoczekiwany błąd podczas pracy z Modal: %s", exc, exc_info=True)
        return None

    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not segments:
        logging.warning("Modal diarization zwrócił pusty wynik.")
        return None

    remote_result, identified_map = _normalize_modal_segments(segments)

    def remote_pipeline(_: str) -> RemoteDiarizationResult:
        return remote_result

    logging.info("Używam diarizacji zdalnej Modal (segmentów: %d).", len(segments))
    return remote_pipeline, identified_map


# Konfiguracja loggerów Speechbrain – ograniczamy je do ostrzeżeń i wyłączamy propagację.
_speechbrain_logger = logging.getLogger("speechbrain")
_speechbrain_logger.setLevel(logging.WARNING)
_speechbrain_logger.propagate = False
for _module in (
    "speechbrain.utils.fetching",
    "speechbrain.utils.checkpoints",
    "speechbrain.utils.parameter_transfer",
    "speechbrain.dataio.encoder",
):
    _logger = logging.getLogger(_module)
    _logger.setLevel(logging.WARNING)
    _logger.propagate = False

# Tłumimy komunikaty FutureWarning z speechbrain.utils.autocast (torch.cuda.amp.custom_fwd).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="speechbrain.utils.autocast",
)

# Klasa do przechwytywania wyjścia tqdm
class TqdmCapture:
    def __init__(self, callback: Callable[[float], None]):
        self.callback = callback
        self.buffer = ""
        self.last_progress = 0.0

    def write(self, s):
        self.buffer += s
        # Próbuj parsować procent postępu z linii tqdm
        match = re.search(r'(\d+)\%\|.*\|\s*(\d+)/(\d+)\s*\[.*\]', s)
        if match:
            try:
                current_percentage = float(match.group(1))
                # Tylko wysyłaj, jeśli postęp się zmienił
                if current_percentage > self.last_progress:
                    self.callback(current_percentage)
                    self.last_progress = current_percentage
            except ValueError:
                pass # Ignore if parsing fails
        
        # Opcjonalnie: przekazuj dalej do oryginalnego stderr/stdout
        # sys.__stderr__.write(s)

    def flush(self):
        pass # tqdm flushuje, nie musimy nic robić

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
    from transformers import pipeline, set_seed
    from transformers.utils import is_flash_attn_2_available
    hf_pipeline = pipeline # Rename to avoid conflict if needed, but seems ok
except ImportError:
    logging.warning("Warning: transformers library not found. Needed for MKV processing.")
    hf_pipeline = None

# --- Imports from diarization_processor ---
# Try importing models safely
try:
    import whisper # Import dla openai-whisper
except ImportError:
    logging.warning("Warning: openai-whisper library not found. Needed for diarization processing.")
    whisper = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline, Model, Inference # Import dla pyannote
except ImportError:
    logging.warning("Warning: pyannote.audio library not found. Needed for diarization.")
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
        process = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        logging.info(f"Successfully converted '{os.path.basename(mkv_path)}' to '{os.path.basename(output_wav_path)}'")
        return True
    except FileNotFoundError:
        logging.error(f"Error: '{ffmpeg_executable}' command not found. Is ffmpeg installed and in PATH, or is FFMPEG_PATH in config.py set correctly?")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting '{os.path.basename(mkv_path)}': Stderr: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion of '{os.path.basename(mkv_path)}': {e}")
        return False


def convert_audio_to_16k_mono(input_path: Path, ffmpeg_path: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """
    Converts any audio file to 16kHz mono WAV using ffmpeg.
    Returns a tuple (success_flag, output_path_or_none).
    """
    ffmpeg_executable = ffmpeg_path or config.FFMPEG_PATH or 'ffmpeg'
    input_path = Path(input_path)

    # Jeśli plik już jest WAV 16k mono, zwracamy go bez zmian.
    try:
        metadata = torchaudio.info(str(input_path))
        if (
            metadata.sample_rate == 16000
            and metadata.num_channels == 1
            and input_path.suffix.lower() == ".wav"
        ):
            return True, input_path
    except Exception as meta_error:
        logging.debug(f"Could not read audio metadata for {input_path.name}: {meta_error}")

    target_path = input_path.with_name(f"{input_path.stem}_16k.wav")
    command = [
        ffmpeg_executable,
        '-y',
        '-i', str(input_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        str(target_path)
    ]

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            creationflags=creationflags
        )
        if process.returncode != 0:
            logging.error(
                f"Error converting '{input_path.name}' to 16k mono WAV. "
                f"Stdout: {process.stdout or 'None'} | Stderr: {process.stderr or 'None'}"
            )
            return False, None

        logging.info(
            f"Successfully converted '{input_path.name}' to 16k mono WAV: '{target_path.name}'"
        )
        return True, target_path

    except FileNotFoundError:
        logging.error(
            f"Error: '{ffmpeg_executable}' command not found. Cannot convert {input_path.name} to 16kHz."
        )
        return False, None
    except Exception as e:
        logging.error(
            f"Unexpected error while converting '{input_path.name}' to 16k mono WAV: {e}",
            exc_info=True
        )
        return False, None

def find_mkv_files(directory):
    """Finds all MKV files recursively in the specified directory."""
    if not os.path.isdir(directory):
        logging.error(f"Error: Directory not found: {directory}")
        return []
    mkv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mkv")]
    logging.info(f"Found {len(mkv_files)} MKV files in '{directory}'.")
    return mkv_files

def process_mkv_files(input_directory, output_directory):
    """
    Finds MKV files, converts them to WAV, processes them (transcribe/translate)
    using the Hugging Face Whisper pipeline, and saves the result to a TXT file.
    """
    if hf_pipeline is None:
        logging.error("Error: Hugging Face pipeline (transformers) is not available. Cannot process MKV files.")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            logging.info(f"Created output directory: {output_directory}")
        except Exception as e:
            logging.error(f"Error creating output directory {output_directory}: {e}")
            return

    mkv_files = find_mkv_files(input_directory)
    if not mkv_files:
        return

    logging.info(f"Initializing Hugging Face pipeline with model: {config.HF_PIPELINE_MODEL}")
    logging.info(f"Using device: {config.DEVICE}, dtype: {config.HF_TORCH_DTYPE}, attention: {config.HF_ATTENTION_IMPL}")
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
        logging.error(f"Error initializing Hugging Face pipeline: {e}")
        pipeline_instance = None
        return

    logging.info(f"Starting processing of {len(mkv_files)} MKV files...")
    for mkv_file in mkv_files:
        logging.info(f"\nProcessing: {mkv_file}")
        base_name = os.path.splitext(os.path.basename(mkv_file))[0]
        wav_file = os.path.join(output_directory, base_name + '.wav')
        output_txt_path = os.path.join(output_directory, base_name + '.txt')

        if not convert_mkv_to_wav(mkv_file, wav_file):
            logging.error(f"Skipping processing for '{os.path.basename(mkv_file)}' due to conversion error.")
            continue

        logging.info(f"Processing audio: {wav_file} (Task: {config.HF_TRANSLATION_TASK})")
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
                logging.info(f"Text output saved: {output_txt_path}")
            except IOError as e:
                 logging.error(f"Error writing text file {output_txt_path}: {e}")

            try:
                os.remove(wav_file)
                logging.info(f"Removed intermediate WAV file: {wav_file}")
            except OSError as e:
                logging.warning(f"Warning: Could not remove intermediate WAV file {wav_file}: {e}")

        except Exception as e:
            logging.error(f"Error during pipeline processing for {wav_file}: {e}")

    logging.info("\nMKV processing finished.")


# ================================================
# Functions from diarization_processor.py
# ================================================

# Ta funkcja (load_diarization_models) została przeniesiona do utils.py i jest importowana.
# Jej definicja nie jest już potrzebna bezpośrednio w processor.py.


# --- Main Audio Processing Function (for API use) ---
async def process_audio(
    audio_path: Path,
    file_job_id: str,
    transcription_model: Optional[str],
    transcription_provider: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    username: Optional[str] = None,
    file_id: Optional[int] = None,
    provider_tokens: Optional[Dict[str, str]] = None,
) -> Tuple[List[TranscriptionSegment], str, int]:
    logging.info(f"Starting audio processing for file_job_id {file_job_id}: {audio_path.name}")

    if not isinstance(file_job_id, str):
        error_msg = f"Invalid file_job_id type: expected str, got {type(file_job_id)}"
        logging.error(error_msg)
        raise TypeError(error_msg)

    logging.debug(f"DEBUG: process_audio received {len(locals()) - 1} arguments.")

    sanitized_tokens: Dict[str, str] = {}
    if provider_tokens:
        for key, value in provider_tokens.items():
            if not isinstance(value, str):
                continue
            normalized_key = str(key).strip().lower()
            if normalized_key not in {"gemini", "assemblyai", "openai"}:
                continue
            sanitized_value = value.strip()
            if sanitized_value:
                sanitized_tokens[normalized_key] = sanitized_value

    try:
        conversion_success, converted_path = await asyncio.to_thread(
            convert_audio_to_16k_mono, audio_path
        )
        if conversion_success and converted_path:
            if converted_path != audio_path:
                logging.info(
                    f"Using converted 16kHz audio for processing (file_job_id {file_job_id}): {converted_path.name}"
                )
            audio_path = converted_path
        else:
            logging.warning(
                f"Falling back to original audio for file_job_id {file_job_id} due to conversion issues."
            )

        provider_id, normalized_model = resolve_transcription_choice(
            transcription_provider,
            transcription_model,
        )
        provider_catalog = get_transcription_provider_catalog()
        provider_entry = next(
            (entry for entry in provider_catalog if entry.get("provider_id") == provider_id),
            {"label": provider_id.title(), "default_model": None},
        )
        provider_label = provider_entry.get("label") or provider_id.title()
        logging.info(
            "Selected transcription provider '%s' (id=%s) with model '%s' for job %s.",
            provider_label,
            provider_id,
            normalized_model,
            file_job_id,
        )

        remote_identified_map: Optional[Dict[Tuple[float, float, str], str]] = None

        if provider_id in ("gemini", "assemblyai"):
            modal_context = await _attempt_modal_diarization(audio_path, username)
            if modal_context:
                diarization_pipeline, remote_identified_map = modal_context
                whisper_model = None
                embedding_model = None
            else:
                raise TranscriptionProviderError(
                    "Nie udało się zainicjalizować diarizacji w Modal – przerwano zadanie."
                )
        else:
            whisper_model, diarization_pipeline, embedding_model = await get_diarization_models(
                normalized_model,
                openai_api_key_override=sanitized_tokens.get("openai"),
            )

        if diarization_pipeline is None:
            logging.error("Diarization pipeline failed to load. Cannot process audio.")
            return [], None, None

        if embedding_model is None and remote_identified_map is None:
            logging.error("Embedding model unavailable and brak mapy identyfikacji. Przerywam.")
            return [], None, None

        if provider_id == "gemini":
            aligned_segments, diarization_result = await _transcribe_with_gemini_segments(
                audio_path=audio_path,
                file_job_id=file_job_id,
                diarization_pipeline=diarization_pipeline,
                embedding_model=embedding_model,
                gemini_model_name=normalized_model or config.DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
                username=username,
                progress_callback=progress_callback,
                remote_identified_map=remote_identified_map,
                gemini_api_key=sanitized_tokens.get("gemini"),
            )
            if not aligned_segments:
                logging.error("Gemini transcription produced no segments.")
                return [], None, None
        elif provider_id == "assemblyai":
            aligned_segments, diarization_result = await _transcribe_with_assemblyai(
                audio_path=audio_path,
                file_job_id=file_job_id,
                diarization_pipeline=diarization_pipeline,
                assemblyai_model_name=normalized_model or config.DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL,
                progress_callback=progress_callback,
                username=username,
                remote_identified_map=remote_identified_map,
                assemblyai_api_key=sanitized_tokens.get("assemblyai"),
            )
            if not aligned_segments:
                logging.error("AssemblyAI transcription produced no segments.")
                return [], None, None
        else:
            use_remote_whisper = getattr(whisper_model, "uses_remote_api", False)
            alias_label = normalized_model or "default"
            if use_remote_whisper:
                resolved_name = getattr(whisper_model, "resolved_model_name", None)
                requested_alias = getattr(whisper_model, "requested_alias", None) or alias_label
                alias_label = requested_alias
                if resolved_name:
                    if resolved_name.lower() != (requested_alias or "").lower():
                        alias_label = f"{requested_alias} → {resolved_name}"
                    else:
                        alias_label = resolved_name
            logging.info(f"  1/3 Performing transcription with Whisper ({alias_label})...")

            tqdm_capture = None
            original_stdout = None
            original_stderr = None
            if progress_callback and not use_remote_whisper:
                tqdm_capture = TqdmCapture(progress_callback)
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = tqdm_capture
                sys.stderr = tqdm_capture

            if use_remote_whisper and progress_callback:
                progress_callback(5.0)

            try:
                if use_remote_whisper:
                    transcription_result = await asyncio.to_thread(
                        whisper_model.transcribe,
                        str(audio_path),
                    )
                else:
                    transcription_result = await asyncio.to_thread(
                        whisper_model.transcribe,
                        str(audio_path),
                        verbose=None,
                        fp16=config.DEVICE.startswith("cuda")
                    )
            finally:
                if tqdm_capture:
                    if original_stdout is not None:
                        sys.stdout = original_stdout
                    if original_stderr is not None:
                        sys.stderr = original_stderr

            logging.info("  ✅ Transcription complete.")

            if not isinstance(transcription_result, dict):
                logging.error(
                    "Unexpected transcription result type from Whisper: %s",
                    type(transcription_result),
                )
                return [], None, None

            transcription_result.setdefault("segments", [])

            if progress_callback:
                progress_callback(100.0)

            logging.info("  2/3 Performing diarization with Pyannote...\n")
            diarization_result = await asyncio.to_thread(diarization_pipeline, str(audio_path))
            logging.info("  ✅ Diarization complete.")

            logging.info("  3/3 Aligning transcription with diarization and saving...\n")
            aligned_segments = await diarize_text(
                transcription_result,
                diarization_result,
                original_audio_path=audio_path,
                username=username
            )

        transcription_segments_objects = [
            TranscriptionSegment(
                start=segment[0].start,
                end=segment[0].end,
                speaker=segment[1],
                text=segment[2].strip()
            )
            for segment in aligned_segments
        ]

        output_dir = Path(config.OUTPUT_DIR_API_TRANSCRIPTS) / file_job_id
        await asyncio.to_thread(os.makedirs, output_dir, exist_ok=True)
        base_filename = audio_path.stem
        safe_base_filename = "".join([c if c.isalnum() else "_" for c in base_filename])
        current_datetime_str = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
        output_transcription_filename = Path(output_dir) / f"{current_datetime_str}_{safe_base_filename}_diarized_transcription.txt"

        full_transcription_content = ""
        f = None
        try:
            f = await asyncio.to_thread(open, output_transcription_filename, "w", encoding="utf-8")
            for segment in aligned_segments:
                speaker = segment[1]
                text = segment[2]
                line = f"{speaker}: {text}\n"
                await asyncio.to_thread(f.write, line)
                full_transcription_content += line
        finally:
            if f:
                await asyncio.to_thread(f.close)

        logging.info(f"  ✅ Diarized transcription saved to {output_transcription_filename}")

        transcript_id = await asyncio.to_thread(
            agent_db.add_transcript,
            full_transcription_content,
            str(output_transcription_filename),
            file_id,
        )
        logging.info(f"  ✅ Transcription added to database with ID: {transcript_id}")

        result_tuple = (transcription_segments_objects, str(output_transcription_filename), transcript_id)
        logging.info(f"DEBUG: process_audio returning {len(result_tuple)} values.")
        return result_tuple

    except TranscriptionProviderError as provider_exc:
        logging.error("Transcription provider error for %s: %s", audio_path.name, provider_exc)
        raise
    except Exception as e:
        logging.error(f"Error during audio processing for {audio_path.name}: {e}")
        import traceback
        traceback.print_exc()
        error_result_tuple = ([], None, None)
        logging.error(f"DEBUG: process_audio returning {len(error_result_tuple)} values on error.")
        return error_result_tuple
    finally:
        if torch.cuda.is_available():
            await asyncio.to_thread(torch.cuda.empty_cache)
            logging.debug(f"Cleared CUDA cache for file_job_id {file_job_id} after audio processing.")

async def get_audio_duration(file_path: Path) -> float:
    """
    Returns the duration of an audio file in seconds.
    """
    try:
        metadata = await asyncio.to_thread(torchaudio.info, str(file_path))
        return metadata.num_frames / metadata.sample_rate
    except Exception as e:
        logging.error(f"Error getting audio duration for {file_path}: {e}", exc_info=True)
        return 0.0


async def _transcribe_with_gemini_segments(
    audio_path: Path,
    file_job_id: str,
    diarization_pipeline: Callable[[str], Any],
    embedding_model: Any,
    gemini_model_name: str,
    username: Optional[str],
    progress_callback: Optional[Callable[[float], None]] = None,
    remote_identified_map: Optional[Dict[Tuple[float, float, str], str]] = None,
    gemini_api_key: Optional[str] = None,
) -> Tuple[List[Tuple[Any, str, str]], Any]:
    """
    Wykonuje diarizację audio oraz transkrypcję każdej z tur mówcy przy użyciu Gemini.
    Zwraca listę segmentów zgodnych z formatem (Segment, speaker_name, text) oraz obiekt diarization_result.
    """
    transcriber = get_gemini_transcriber(gemini_model_name, api_key_override=gemini_api_key)
    if transcriber is None:
        raise TranscriptionProviderError(
            f"Nie można zainicjalizować transkrypcji Gemini dla modelu '{gemini_model_name}'. Sprawdź klucz API i zakres uprawnień."
        )

    logging.info("  1/3 Performing diarization with Pyannote...")
    diarization_result = await asyncio.to_thread(diarization_pipeline, str(audio_path))
    logging.info("  ✅ Diarization complete.")

    speaker_turns: List[Tuple[Any, str]] = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_turns.append((turn, speaker))
    speaker_turns.sort(key=lambda item: item[0].start)

    if not speaker_turns:
        logging.warning("Diarization returned no speaker turns; transcription may be empty.")

    waveform, sample_rate = await asyncio.to_thread(torchaudio.load, str(audio_path))

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    temp_segment_dir = Path("./temp_audio_segments")
    await asyncio.to_thread(temp_segment_dir.mkdir, exist_ok=True)

    aligned_segments: List[Tuple[Any, str, str]] = []
    total_turns = len(speaker_turns)

    logging.info("  2/3 Performing transcription with Gemini (%s)...", gemini_model_name)
    max_retries = max(1, getattr(config, "GEMINI_TRANSCRIBE_MAX_RETRIES", 3))
    retry_base_delay = max(0.0, getattr(config, "GEMINI_TRANSCRIBE_RETRY_BASE_DELAY", 1.0))
    for idx, (turn, diarized_label) in enumerate(speaker_turns):
        start_time = max(0.0, float(turn.start))
        end_time = max(start_time, float(turn.end))
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)

        segment_waveform = waveform[:, start_frame:end_frame] if end_frame > start_frame else torch.zeros_like(waveform[:, :1])
        if segment_waveform.nelement() == 0 or end_frame <= start_frame:
            logging.debug("Segment %d for job %s has zero length. Skipping transcription.", idx, file_job_id)
            segment_text = ""
            key = (round(start_time, 3), round(end_time, 3), diarized_label)
            identified_speaker = (
                remote_identified_map.get(key, diarized_label)
                if remote_identified_map
                else diarized_label
            )
        else:
            temp_audio_path = temp_segment_dir / f"gemini_segment_{file_job_id}_{idx}.wav"
            await asyncio.to_thread(torchaudio.save, str(temp_audio_path), segment_waveform, sample_rate)

            segment_text = ""
            attempt = 0
            while True:
                attempt += 1
                try:
                    segment_text = await asyncio.to_thread(transcriber.transcribe_file, str(temp_audio_path))
                    break
                except GeminiTranscriptionError as transcribe_error:
                    logging.error(
                        "Gemini transcription failed for segment %d (job %s) attempt %d/%d: %s",
                        idx,
                        file_job_id,
                        attempt,
                        max_retries,
                        transcribe_error,
                    )
                    message = str(transcribe_error)
                    lower_message = message.lower()
                    if "404" in message and "models" in message:
                        raise TranscriptionProviderError(
                            f"Model Gemini '{gemini_model_name}' nie jest dostępny dla Twojego konta lub API."
                        ) from transcribe_error
                    if "api key" in lower_message or "permission" in lower_message or "unauthorized" in lower_message:
                        raise TranscriptionProviderError(
                            "Gemini odrzuciło żądanie. Sprawdź klucz transkrypcji i uprawnienia konta."
                        ) from transcribe_error
                    overloaded = "overloaded" in lower_message or "503" in message
                    if overloaded and attempt < max_retries:
                        backoff_delay = retry_base_delay * (2 ** (attempt - 1))
                        logging.warning(
                            "Gemini przeciążony dla segmentu %d (job %s). Ponawianie za %.1f s...",
                            idx,
                            file_job_id,
                            backoff_delay,
                        )
                        await asyncio.sleep(backoff_delay)
                        continue
                    if overloaded:
                        raise TranscriptionProviderError(
                            "Model Gemini jest obecnie przeciążony. Spróbuj ponownie za kilka minut."
                        ) from transcribe_error
                    raise TranscriptionProviderError(
                        f"Gemini nie zdołało przetworzyć segmentu audio: {message}"
                    ) from transcribe_error
                except Exception as transcribe_error:
                    logging.error(
                        "Gemini transcription failed for segment %d (job %s): %s",
                        idx,
                        file_job_id,
                        transcribe_error,
                    )
                    message = str(transcribe_error)
                    if "404" in message and "models/" in message:
                        raise TranscriptionProviderError(
                            f"Model Gemini '{gemini_model_name}' nie jest dostępny dla Twojego konta lub API."
                        ) from transcribe_error
                    raise

            identified_speaker = diarized_label
            key = (round(start_time, 3), round(end_time, 3), diarized_label)
            if remote_identified_map:
                identified_speaker = remote_identified_map.get(key, diarized_label)
            elif embedding_model is not None:
                try:
                    identified_candidate = await identify_speaker(temp_audio_path, username=username)
                    if identified_candidate and identified_candidate != "SPEAKER_UNKNOWN":
                        identified_speaker = identified_candidate
                except Exception as speaker_error:
                    logging.warning(
                        "Speaker identification failed for segment %d (job %s): %s",
                        idx,
                        file_job_id,
                        speaker_error,
                    )
            try:
                await asyncio.to_thread(os.remove, temp_audio_path)
            except Exception as remove_error:
                logging.debug("Could not remove temporary segment file %s: %s", temp_audio_path, remove_error)

        aligned_segments.append((turn, identified_speaker, (segment_text or "").strip()))

        if progress_callback and total_turns:
            segment_progress = ((idx + 1) / total_turns) * 100.0
            progress_callback(min(100.0, segment_progress))

    logging.info("  ✅ Gemini transcription complete.")
    return aligned_segments, diarization_result


async def _transcribe_with_assemblyai(
    audio_path: Path,
    file_job_id: str,
    diarization_pipeline: Callable[[str], Any],
    assemblyai_model_name: Optional[str],
    progress_callback: Optional[Callable[[float], None]] = None,
    username: Optional[str] = None,
    remote_identified_map: Optional[Dict[Tuple[float, float, str], str]] = None,
    assemblyai_api_key: Optional[str] = None,
) -> Tuple[List[Tuple[Any, str, str]], Any]:
    """
    Transkrybuje audio za pomocą AssemblyAI i dokonuje alignu z diarizacją Pyannote.
    """
    requested_model = (assemblyai_model_name or config.DEFAULT_ASSEMBLYAI_TRANSCRIPTION_MODEL or "").strip().lower()
    transcriber = get_assemblyai_transcriber(requested_model, api_key_override=assemblyai_api_key)
    if transcriber is None:
        raise TranscriptionProviderError(
            "AssemblyAI nie jest poprawnie skonfigurowane. Ustaw pakiet 'assemblyai' oraz zmienną ASSEMBLYAI_API_KEY."
        )

    logging.info("  1/3 Performing diarization with Pyannote...")
    diarization_result = await asyncio.to_thread(diarization_pipeline, str(audio_path))
    logging.info("  ✅ Diarization complete.")
    if progress_callback:
        progress_callback(20.0)

    audio_duration = await get_audio_duration(audio_path)

    logging.info("  2/3 Performing transcription with AssemblyAI (%s)...", requested_model or "(default)")
    try:
        transcription_result = await asyncio.to_thread(
            transcriber.transcribe_file,
            str(audio_path),
            fallback_duration=audio_duration,
        )
    except AssemblyAITranscriptionError as transcribe_error:
        raise TranscriptionProviderError(f"AssemblyAI odrzuciło żądanie: {transcribe_error}") from transcribe_error
    except Exception as transcribe_error:
        raise TranscriptionProviderError(f"Nieoczekiwany błąd AssemblyAI: {transcribe_error}") from transcribe_error

    if not isinstance(transcription_result, dict):
        raise TranscriptionProviderError("AssemblyAI zwróciło nieprawidłową strukturę odpowiedzi.")

    full_text = (transcription_result.get("text") or "").strip()
    segments_list = transcription_result.get("segments")
    if not full_text and isinstance(segments_list, list):
        joined = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments_list
            if isinstance(seg, dict) and (seg.get("text") or "").strip()
        ).strip()
        if joined:
            full_text = joined
    if not full_text:
        raise TranscriptionProviderError(
            "AssemblyAI zwróciło pustą transkrypcję (brak tekstu w odpowiedzi)."
        )

    transcription_result.setdefault("segments", segments_list if isinstance(segments_list, list) else [])
    if progress_callback:
        progress_callback(60.0)

    logging.info("  3/3 Aligning transcription with diarization and saving...")
    aligned_segments = await diarize_text(
        transcription_result,
        diarization_result,
        original_audio_path=audio_path,
        username=username,
        fallback_audio_duration=audio_duration,
    )
    if not aligned_segments:
        logging.warning("AssemblyAI transcription yielded no aligned segments; falling back to single-speaker segment.")
        fallback_end = float(audio_duration) if audio_duration else 0.0
        fallback_segment = type("obj", (object,), {"start": 0.0, "end": fallback_end})()
        aligned_segments = [
            (fallback_segment, "SPEAKER_UNKNOWN", full_text),
        ]

    if remote_identified_map:
        aligned_segments = _apply_remote_identifications(aligned_segments, remote_identified_map)

    if progress_callback:
        progress_callback(100.0)

    logging.info("  ✅ AssemblyAI transcription and alignment complete.")
    return aligned_segments, diarization_result
