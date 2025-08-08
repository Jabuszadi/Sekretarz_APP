import sys
import os
import logging
import torch

# Ustawienie ścieżki do folderu projektu, aby można było importować moduły
sys.path.insert(0, os.path.abspath('.'))

# Ustawienie poziomu logowania na DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from utils import get_diarization_models
    logging.info("Successfully imported get_diarization_models from utils.py")
except ImportError as e:
    logging.error(f"Failed to import get_diarization_models: {e}. Make sure utils.py is in the correct path and its dependencies are met.")
    sys.exit(1) # Wyjdź, jeśli import się nie powiedzie

def run_quick_test():
    logging.info("Starting quick model loading test...")

    # Te logi potwierdzają, czy funkcja została wywołana
    logging.info("Attempting to call get_diarization_models()...")
    whisper_model, diarization_pipeline, embedding_model = get_diarization_models()
    logging.info("Finished calling get_diarization_models().")

    if whisper_model:
        logging.info("Whisper model: LOADED")
    else:
        logging.error("Whisper model: FAILED TO LOAD")

    if diarization_pipeline:
        logging.info("Pyannote Diarization Pipeline: LOADED")
    else:
        logging.error("Pyannote Diarization Pipeline: FAILED TO LOAD")

    if embedding_model:
        logging.info("Pyannote Embedding Model: LOADED")
    else:
        logging.error("Pyannote Embedding Model: FAILED TO LOAD")

    logging.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA not available, running on CPU.")

    logging.info("Quick test finished.")

if __name__ == "__main__":
    run_quick_test()
