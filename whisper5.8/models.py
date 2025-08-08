# models.py
from pydantic import BaseModel
from typing import Optional, List

class ProcessingRequest(BaseModel):
    whisper_model_size: str
    hf_token: Optional[str] = None

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

class MinutesResponse(BaseModel):
    Główne_Tematy_Omówione: Optional[str] = None
    Kluczowe_Podjęte_Decyzje: Optional[str] = None
    Lista_Zadań_do_Wykonania: Optional[str] = None
    Ważne_Działania_Następcze_lub_Kolejne_Kroki: Optional[str] = None
    Najważniejsze_Uwagi_lub_Komentarze_od_Uczestników: Optional[str] = None

class ProcessingResult(BaseModel):
    status: str
    message: str
    transcription: Optional[List[TranscriptionSegment]] = None
    minutes: Optional[MinutesResponse] = None
    transcription_filename: Optional[str] = None
    minutes_filename: Optional[str] = None
    unique_collection_name: Optional[str] = None