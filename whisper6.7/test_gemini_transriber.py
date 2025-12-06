from google import genai

from config import GEMINI_TRANSCRIBE_API_KEY

if not GEMINI_TRANSCRIBE_API_KEY:
    raise RuntimeError(
        "Brak konfiguracji GEMINI_TRANSCRIBE_API_KEY. Ustaw zmienną środowiskową lub wpis w .env."
    )

client = genai.Client(api_key=GEMINI_TRANSCRIBE_API_KEY)


def _wait_for_active_file(file_obj, timeout_seconds: int = 60):
    import time

    start = time.time()
    while True:
        current = client.files.get(name=file_obj.name)
        state = getattr(current, "state", None)
        if state and getattr(state, "name", str(state)).upper() == "ACTIVE":
            return current
        if state and getattr(state, "name", str(state)).upper() == "FAILED":
            raise RuntimeError(f"Przetwarzanie pliku nie powiodło się (state={state}).")
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Przetwarzanie pliku Gemini trwa zbyt długo (timeout).")
        time.sleep(1)


uploaded_file = client.files.upload(
    file="output_1min.mp4",
    config={"mime_type": "video/mp4"},
)
active_file = _wait_for_active_file(uploaded_file)

prompt = "Generate a transcript of the speech."

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[prompt, active_file],
)

print(response.text)