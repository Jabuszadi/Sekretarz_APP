import google.generativeai as genai
from config import GEMINI_TRANSCRIBE_API_KEY  # zakładam, że config.py już zrobił load_dotenv()

def list_models():
    genai.configure(api_key=GEMINI_TRANSCRIBE_API_KEY)
    for model in genai.list_models():
        modalities = getattr(model, "input_modalities", None) or []
        print(f"{model.name} | modalities: {modalities} | {model.supported_generation_methods}")

if __name__ == "__main__":
    list_models()