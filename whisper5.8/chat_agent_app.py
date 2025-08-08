# chat_agent_app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse # Dodaj HTMLResponse
from pydantic import BaseModel
import logging
import asyncio
from typing import List, Optional # Dodaj Optional
from pathlib import Path # Dodaj Path
from fastapi.staticfiles import StaticFiles # Dodaj StaticFiles
from fastapi import UploadFile, File # Dodaj importy dla plików
import uuid # Dodaj import dla UUID
import os # Dodaj import dla os
import shutil # Dodaj import dla shutil

# Importujemy potrzebne funkcje i modele z istniejących plików
from qdrant_handler import initialize_qdrant_resources, search_all_collections
from summarizer import gemini_model # Zakładamy, że gemini_model jest już zainicjalizowany w summarizer.py
import config # Do dostępu do konfiguracji, np. dla ustawień LLM
from fastmcp.client import MCPClient # Dodaj import dla MCPClient

# Inicjalizacja klienta MCP dla komunikacji z agentem audio
mcp_client = MCPClient(base_url="http://127.0.0.1:8002")

# Ustawienie logowania dla chat_agent_app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Wycisz logi httpx
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Wycisz logi uvicorn

app = FastAPI(
    title="Chat Agent API (Internal)", # Zmieniono tytuł, aby odzwierciedlał wewnętrzne przeznaczenie
    description="Internal API for interacting with the transcription and meeting minutes data via chat tools.",
    version="1.0.0",
)

# Modele Pydantic dla zapytania i odpowiedzi czatu
class ChatQuery(BaseModel):
    query: str
    collection_name: Optional[str] = None # NOWE: Opcjonalna nazwa kolekcji
    
class ChatResponse(BaseModel):
    response: str

# Funkcja do generowania odpowiedzi LLM na podstawie kontekstu i zapytania
async def generate_chat_response(user_query: str, context_documents: List[str]) -> str:
    """
    Generuje odpowiedź Agenta na podstawie zapytania użytkownika i dostarczonych dokumentów kontekstowych.
    """
    if not gemini_model:
        logging.error("Gemini model not initialized. Cannot generate chat response.")
        return "Przepraszam, model AI nie jest dostępny. Spróbuj ponownie później."

    # Połącz dokumenty kontekstowe w jeden duży ciąg
    context_str = "\n".join(context_documents)
    if not context_str:
        return "Nie znalazłem żadnych informacji w bazie danych, które odpowiadałyby na Twoje pytanie."

    # Utwórz prompt dla modelu Gemini
    prompt = f"""
Jesteś pomocnym asystentem AI. Twoim zadaniem jest odpowiadanie na pytania użytkownika,
wykorzystując *wyłącznie* informacje zawarte w podanym fragmencie tekstu.
Jeśli informacja nie znajduje się w tekście, odpowiedz, że nie możesz znaleźć odpowiedzi na podstawie dostępnych danych.
Nie wymyślaj informacji. Odpowiadaj zwięźle i na temat.

**Zapytanie Użytkownika:**
{user_query}

**Dostępny Kontekst (fragmenty transkrypcji/protokołów):**
---
{context_str}
---

**Twoja Odpowiedź (oparta wyłącznie na Kontekście):**
"""
    try:
        # Wywołanie modelu Gemini. Używamy asyncio.to_thread, bo invoke może być blokujące.
        response = await asyncio.to_thread(gemini_model.invoke, prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error generating chat response with Gemini: {e}")
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja zasobów Qdrant i modelu embeddingowego przy starcie aplikacji."""
    logging.info("Initializing Qdrant and embedding model for Chat Agent...")
    try:
        await initialize_qdrant_resources()
        logging.info("✅ Qdrant and embedding models initialized for Chat Agent.")
    except Exception as e:
        logging.critical(f"❌ Failed to initialize Qdrant resources on startup: {e}. Chat functionality will be limited.")
        # Tutaj możesz zdecydować, czy aplikacja ma się zakończyć, czy działać w ograniczonym trybie.
        # Na razie pozwalamy na działanie, ale logujemy błąd krytyczny.

@app.get("/health")
async def health_check():
    """Endpoint do sprawdzania stanu serwera."""
    return {"status": "ok", "message": "Chat Agent (Internal) is running"}

# PONIŻSZE ENDPOINTY API ZOSTANĄ PRZENIESIONE DO api_app.py
# @app.get("/", response_class=HTMLResponse)
# async def serve_chat_interface():
#     html_file_path = Path("chat_interface.html")
#     if not html_file_path.exists():
#         raise HTTPException(status_code=404, detail="Chat interface HTML file not found.")
#     with open(html_file_path, "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read(), status_code=200)

# @app.post("/chat/query", response_model=ChatResponse)
# async def chat_query_endpoint(chat_query: ChatQuery):
#     # ... (przeniesiona logika) ...

# @app.post("/upload-audio")
# async def upload_audio_endpoint( ... ):
#     # ... (przeniesiona logika) ...

# @app.get("/get_qdrant_collections/")
# async def get_qdrant_collections_endpoint():
#     # ... (przeniesiona logika) ...
