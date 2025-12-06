"""
Modal stub odpowiedzialny za diarizację i opcjonalną identyfikację mówców
w oparciu o istniejące moduły projektu (Pyannote + SpeechBrain).

Ten moduł można deployować poleceniem:
    modal deploy modal_diarization.py

Funkcja `run_diarization` zakłada, że otrzyma URL (HTTP/S) lub ścieżkę do pliku
audio. Zwraca listę tur mówców oraz (opcjonalnie) dopasowane nazwy użytkowników.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from modal import App, Image, Volume, gpu

# --- Konfiguracja podstawowa ---

REPO_ROOT = Path(__file__).parent.resolve()
REMOTE_APP_PATH = "/root/app"
REMOTE_CACHE_PATH = "/vol/cache"
REMOTE_SPEAKER_DIR = "/vol/speakers"

if REMOTE_APP_PATH not in sys.path:  # pragma: no cover
    sys.path.append(REMOTE_APP_PATH)

MODELS_VOLUME_NAME = os.getenv("MODAL_MODELS_VOLUME", "sekretarz-model-cache")
SPEAKERS_VOLUME_NAME = os.getenv("MODAL_SPEAKERS_VOLUME", "sekretarz-speakers")
GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L4")

models_volume = Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
speakers_volume = Volume.from_name(SPEAKERS_VOLUME_NAME, create_if_missing=True)

app = App("sekretarz-diarization")

# --- Stan globalny w kontenerze Modal ---

_models_lock: Optional[asyncio.Lock] = None
_diarization_pipeline: Any = None
_embedding_model: Any = None
_torch_device = None


async def _ensure_models_loaded(username: Optional[str]) -> None:
    """
    Ładuje pipeline diarizacji oraz model embeddingów, cache'ując je w module.
    """
    global _models_lock, _diarization_pipeline, _embedding_model, _torch_device

    if _diarization_pipeline is not None and _embedding_model is not None:
        return

    if _models_lock is None:
        _models_lock = asyncio.Lock()

    async with _models_lock:
        if _diarization_pipeline is not None and _embedding_model is not None:
            return

        # Importy wewnątrz funkcji, aby działały zarówno lokalnie jak i na Modal
        import torch

        from config import DEVICE as CONFIG_DEVICE
        from utils import get_diarization_models, load_enrolled_speakers

        # Zabezpieczenie na wypadek niespójnej konfiguracji
        if not CONFIG_DEVICE.startswith("cuda"):
            logging.info("Przestawiam DEVICE na cuda dla środowiska Modal.")
            os.environ["DEVICE"] = "cuda"
        _torch_device = torch.device("cuda")

        # Przestaw katalog głośników na wolumen Modal
        os.environ["SPEAKER_ENROLLMENT_DIR"] = REMOTE_SPEAKER_DIR

        logging.info("Ładowanie modeli diarizacji i embeddingów w środowisku Modal...")
        whisper_model, diarization_pipeline, embedding_model = await get_diarization_models(
            requested_whisper_model_size=None,
            load_whisper=False,
        )
        if diarization_pipeline is None or embedding_model is None:
            raise RuntimeError("Nie udało się załadować pipeline diarizacji lub embeddingów.")

        _diarization_pipeline = diarization_pipeline
        _embedding_model = embedding_model

        # Załaduj profile mówców dla wskazanego namespace
        await load_enrolled_speakers(username=username)
        logging.info("Modele i profile mówców gotowe.")


async def _download_audio(audio_reference: str) -> Path:
    """
    Pobiera plik audio z URL lub odczytuje lokalną ścieżkę (jeśli jest zamontowana).
    """
    candidate_path = Path(audio_reference)
    if candidate_path.exists():
        return candidate_path

    import httpx

    tmp_dir = Path(tempfile.gettempdir()) / "modal_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target_path = tmp_dir / f"{uuid.uuid4().hex}.wav"

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("GET", audio_reference) as response:
            response.raise_for_status()
            with open(target_path, "wb") as out_file:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    if chunk:
                        out_file.write(chunk)

    return target_path


async def _materialize_audio(
    audio_reference: Optional[str],
    file_data: Optional[bytes],
    file_name: Optional[str],
) -> Path:
    """
    Tworzy lokalny plik audio na podstawie URL lub dostarczonego bufora bajtów.
    """
    if file_data:
        temp_dir = Path(tempfile.gettempdir()) / "modal_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file_name).suffix if file_name else ".wav"
        target_path = temp_dir / (file_name or f"{uuid.uuid4().hex}{suffix}")
        await asyncio.to_thread(target_path.write_bytes, file_data)
        return target_path

    if audio_reference:
        return await _download_audio(audio_reference)

    raise ValueError("Musisz przekazać audio_reference lub file_data.")


async def _serialize_diarization_result(
    diarization_result: Any,
    audio_path: Path,
    username: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Konwertuje wynik Pyannote do listy segmentów, opcjonalnie wykonując identyfikację.
    """
    import torchaudio
    import torch

    from utils import identify_speaker

    waveform, sample_rate = await asyncio.to_thread(torchaudio.load, str(audio_path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    temp_segment_dir = Path(tempfile.gettempdir()) / "modal_segments"
    temp_segment_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Dict[str, Any]] = []
    for idx, (turn, _, label) in enumerate(diarization_result.itertracks(yield_label=True)):
        start = max(0.0, float(turn.start))
        end = max(start, float(turn.end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)

        segment_waveform = waveform[:, start_frame:end_frame] if end_frame > start_frame else None
        identified = "SPEAKER_UNKNOWN"

        segment_path: Optional[Path] = None
        if segment_waveform is not None and segment_waveform.nelement() > 0:
            segment_path = temp_segment_dir / f"segment_{uuid.uuid4().hex}.wav"
            await asyncio.to_thread(
                torchaudio.save,
                str(segment_path),
                segment_waveform,
                sample_rate,
            )
            identified = await identify_speaker(segment_path, username=username)

        segments.append(
            {
                "turn_index": idx,
                "start": start,
                "end": end,
                "pyannote_label": label,
                "identified_speaker": identified,
            }
        )

        if segment_path and segment_path.exists():
            try:
                segment_path.unlink()
            except OSError:
                logging.debug("Nie usunięto pliku segmentu %s", segment_path)

    return segments


@app.function(
    image=Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.15.3",
        "soundfile==0.13.1",
        "httpx==0.28.1",
        "python-dotenv==1.1.1",
        "matplotlib==3.10.5",
    )
    .pip_install(
        "pyannote.audio==3.3.2",
        "speechbrain==1.0.3",
        "sentencepiece==0.2.1",
        "transformers==4.55.0",
        "openai==2.7.1",
    )
    .pip_install(
        "google-generativeai==0.8.5",
        "google-ai-generativelanguage==0.6.15",
        "assemblyai==0.46.0",
    )
    .add_local_dir(REPO_ROOT, REMOTE_APP_PATH),
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={
        REMOTE_CACHE_PATH: models_volume,
        REMOTE_SPEAKER_DIR: speakers_volume,
    },
    min_containers=0,
)
async def run_diarization(
    audio_reference: Optional[str] = None,
    username: Optional[str] = None,
    file_data: Optional[bytes] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uruchamia diarizację w środowisku Modal i zwraca listę segmentów.

    Args:
        audio_reference: URL (HTTP/HTTPS) lub ścieżka do pliku audio.
        username: Namespace użytkownika dla identyfikacji mówców.
        file_data: Opcjonalne bajty pliku audio (jeśli brak publicznego URL).
        file_name: Nazwa pliku użyta przy zapisie bufora bajtów.
    """
    logging.info("Rozpoczynam diarizację (ref=%s, file=%s)", audio_reference, file_name)
    await _ensure_models_loaded(username=username)

    # Re-load speaker profiles for this namespace to capture any updates since the container warmed up.
    try:
        from utils import load_enrolled_speakers

        await load_enrolled_speakers(username=username)
    except Exception as speaker_err:
        logging.warning(
            "Nie udało się odświeżyć profili mówców dla namespace '%s': %s",
            username,
            speaker_err,
        )

    audio_path = await _materialize_audio(audio_reference, file_data, file_name)
    try:
        diarization_result = await asyncio.to_thread(_diarization_pipeline, str(audio_path))
        segments = await _serialize_diarization_result(
            diarization_result=diarization_result,
            audio_path=audio_path,
            username=username,
        )
    finally:
        if audio_path.exists():
            try:
                audio_path.unlink()
            except OSError:
                logging.debug("Nie usunięto pliku tymczasowego %s", audio_path)

    logging.info("Zakończono diarizację; znalezione segmenty: %d", len(segments))
    return {"segments": segments}


@app.local_entrypoint()
def main(
    audio_reference: Optional[str] = None,
    file_path: Optional[str] = None,
    username: Optional[str] = None,
) -> None:
    """
    Pozwala przetestować pipeline lokalnie (Modal run) przed deployem.
    """
    file_bytes = None
    file_name = None
    if file_path:
        local_path = Path(file_path)
        file_bytes = local_path.read_bytes()
        file_name = local_path.name

    result = run_diarization.remote(
        audio_reference=audio_reference,
        username=username,
        file_data=file_bytes,
        file_name=file_name,
    )
    print(result)

