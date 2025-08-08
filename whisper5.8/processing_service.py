from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from processor import process_audio as processor_process_audio
from models import TranscriptionSegment
import config
from file_handlers import cleanup_temp_dir

async def process_audio(input_file: Path, job_id: str, whisper_model_size: str = None) -> Tuple[List[TranscriptionSegment], Optional[str], Optional[int]]: # Dodano Optional[int] dla transcript_id
    """
    Process audio file, perform transcription and diarization,
    and return transcription segments, the filename of the saved transcription, and transcript_id.
    """
    # Usunięto tworzenie output_dir tutaj, ponieważ to jest teraz obsługiwane w processor.py
    # output_dir = Path("output_api_transcripts") / datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # Zapisz oryginalną wartość modelu
    original_model = config.WHISPER_MODEL_SIZE
    
    transcription_filename = None
    transcript_id = None # Inicjalizacja transcript_id
    
    try:
        # Ustaw nową wartość modelu jeśli została podana
        if whisper_model_size:
            config.WHISPER_MODEL_SIZE = whisper_model_size
            print(f"Using Whisper model: {whisper_model_size}")  # Debug print
        
        # input_dir nie jest już potrzebny bezpośrednio, ścieżka pliku jest przekazywana
        # Upewnij się, że katalog wejściowy istnieje
        # input_dir = input_file.parent
        # if not input_dir.exists():
        #     raise ValueError(f"Input directory does not exist: {input_dir}")
        
        print(f"Processing file: {input_file}")
        # print(f"Input directory: {input_dir}") # Niepotrzebne
        # print(f"Output directory: {output_dir}") # Niepotrzebne
        
        # Wywołujemy teraz nową funkcję process_audio z processor.py
        segments, transcription_filename, transcript_id = await processor_process_audio( # Zmieniono: rozpakowanie trzech wartości
            audio_path=input_file, # Przekazujemy bezpośrednio ścieżkę pliku
            file_job_id=job_id,
            whisper_model_size=config.WHISPER_MODEL_SIZE # Używamy wartości z config.py, która mogła zostać zmieniona
            # Parametry chunk_duration i chunk_overlap nie są już przekazywane tutaj,
            # ponieważ są one obsługiwane w ingest.py, wywoływanym z api_app.py
        )
        
        if not segments:
            raise ValueError("No segments were generated during processing")
        
        # Konwersja segmentów (już jest OK)
        transcription_segments = [
            TranscriptionSegment(
                start=segment[0].start,
                end=segment[0].end,
                speaker=segment[1],
                text=segment[2].strip()
            )
            for segment in segments
        ]
        
        return transcription_segments, transcription_filename, transcript_id # Zwracanie trzech wartości
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        import traceback
        traceback.print_exc() # Dodano traceback dla lepszego debugowania
        return [], None, None # Zwracanie trzech wartości w przypadku błędu
    finally:
        # Przywróć oryginalną wartość modelu
        config.WHISPER_MODEL_SIZE = original_model
        # Wyczyść katalog tymczasowy (już wywoływane w api_app.py, więc ta linia może być usunięta lub zakomentowana,
        # jeśli cleanup_temp_dir jest wywoływane globalnie dla job_id)
        # cleanup_temp_dir() # Zostawiam na razie, ale prawdopodobnie zbędne