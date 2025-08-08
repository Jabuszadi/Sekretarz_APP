import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os
import logging
import numpy as np

# Konfiguracja logowania, aby widzieć komunikaty DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Pobierz HF_TOKEN ze zmiennych środowiskowych
HF_TOKEN = os.getenv("HF_TOKEN")

# Upewnij się, że token nie jest domyślną wartością-zaślepką
if not HF_TOKEN or HF_TOKEN == "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":
    logging.critical("CRITICAL: Hugging Face token (HF_TOKEN) is not set or is placeholder! "
                     "Please set HF_TOKEN in your .env file and accept terms on Hugging Face for models like 'pyannote/speaker-diarization-3.1' and 'speechbrain/spkrec-ecapa-voxceleb'.")
    HF_TOKEN = None # Ustaw na None, aby EncoderClassifier mógł zgłosić błąd, jeśli token jest wymagany
else:
    logging.info("Hugging Face token loaded.")

# Ustaw urządzenie (GPU jeśli dostępne, w przeciwnym razie CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

_embedding_model = None

def load_embedding_model():
    """
    Loads the SpeechBrain ECAPA-TDNN embedding model.
    """
    global _embedding_model
    if _embedding_model is not None:
        logging.info("Embedding model already loaded, returning cached instance.")
        return _embedding_model

    if EncoderClassifier:
        try:
            logging.info(f"Attempting to load speechbrain/spkrec-ecapa-voxceleb model on {DEVICE}...")
            _embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": DEVICE},
                # For models that require authentication, pass the token
                # use_auth_token=HF_TOKEN # Uncomment if the model itself requires this
            )
            logging.info("SpeechBrain embedding model loaded successfully.")
            return _embedding_model
        except Exception as e:
            logging.error(f"Error loading SpeechBrain embedding model on {DEVICE}: {e}")
            logging.info("Attempting to load SpeechBrain embedding model on CPU...")
            try:
                _embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                    # use_auth_token=HF_TOKEN # Uncomment if the model itself requires this
                )
                logging.info("SpeechBrain embedding model loaded successfully on CPU.")
                return _embedding_model
            except Exception as e_cpu:
                logging.error(f"Error loading SpeechBrain embedding model on CPU: {e_cpu}")
                _embedding_model = None
                return None
    else:
        logging.error("SpeechBrain EncoderClassifier is not available. Cannot load embedding model.")
        return None

def generate_embedding(audio_path: str):
    """
    Generates a speaker embedding for a given audio file.
    """
    model = load_embedding_model()
    if model is None:
        logging.error("Embedding model not loaded. Cannot generate embedding.")
        return None

    try:
        # Load audio file
        signal, sample_rate = torchaudio.load(audio_path)
        
        # Ensure correct sample rate (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            logging.warning(f"Audio sample rate is {sample_rate}Hz. Resampling to 16000Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)
            sample_rate = 16000

        # Move signal to the correct device
        signal = signal.to(DEVICE)

        # Ensure mono audio if stereo
        if signal.shape[0] > 1:
            logging.info("Converting stereo audio to mono.")
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Generate embedding
        # EncoderClassifier expects (batch_size, samples) or (batch_size, channels, samples)
        # Squeeze to (samples) if it's (1, samples), then unsqueeze back to (1, samples) for batch
        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0) # Remove channel dimension if mono
        
        embeddings = model.encode_batch(signal.unsqueeze(0)).squeeze(0)
        logging.info(f"Embedding generated successfully. Shape: {embeddings.shape}")
        return embeddings.cpu().numpy()
    except Exception as e:
        logging.error(f"Error generating embedding for {audio_path}: {e}")
        return None

if __name__ == "__main__":
    # Przykład użycia: podaj ścieżkę do pliku audio
    # Możesz użyć pliku MP3, WAV itp. Upewnij się, że plik istnieje.
    # Na przykład: "ścieżka/do/twojego/pliku.wav"
    # DLA TESTÓW: Stwórz mały plik audio (np. nagraj siebie mówiącego przez kilka sekund)
    # i umieść go w katalogu projektu.
    
    test_audio_file = "speaker_enrollment/andrzej.mp3" # Zmień na istniejący plik audio
    
    # Utwórz fikcyjny plik audio, jeśli go nie ma, dla testów
    if not os.path.exists(test_audio_file):
        logging.warning(f"Plik '{test_audio_file}' nie istnieje. Tworzę pusty plik jako placeholder.")
        # Tworzenie bardzo prostego, krótkiego pliku WAV dla testów
        try:
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=1000) # 1 sekunda ciszy
            silence.export(test_audio_file, format="wav")
            logging.info(f"Utworzono pusty plik '{test_audio_file}'. Proszę zastąpić go prawdziwym plikiem audio.")
        except ImportError:
            logging.error("Aby utworzyć fikcyjny plik audio, potrzebna jest biblioteka pydub. Zainstaluj ją: pip install pydub")
            logging.error("Nie można przetestować generowania embeddingu bez pliku audio.")
            exit()


    logging.info(f"Attempting to generate embedding for: {test_audio_file}")
    embedding = generate_embedding(test_audio_file)

    if embedding is not None:
        print("\\n--- Wynik ---")
        print(f"Pomyślnie załadowano model i wygenerowano embedding.")
        print(f"Kształt embeddingu: {embedding.shape}")
        print(f"Pierwsze 5 wartości embeddingu: {embedding[:5]}")
    else:
        print("\\n--- Wynik ---")
        print("Nie udało się załadować modelu lub wygenerować embeddingu.")
        print("Sprawdź powyższe logi pod kątem błędów.")
