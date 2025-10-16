# Projekt Sekretarz: Asystent AI do Transkrypcji i Analizy Spotkań

## Spis Treści
1.  [Opis Projektu](#1-opis-projektu)
2.  [Architektura Systemu](#2-architektura-systemu)
3.  [Wymagania Systemowe](#3-wymagania-systemowe)
4.  [Instalacja](#4-instalacja)
5.  [Konfiguracja](#5-konfiguracja)
6.  [Uruchamianie Komponentów](#6-uruchamianie-komponentów)
7.  [Przykłady Użycia](#7-przykłady-użycia)
8.  [Struktura Projektu](#8-struktura-projektu)
9.  [Rozwiązywanie Problemów](#9-rozwiązywanie-problemów)

---

## 1. Opis Projektu

Projekt "Sekretarz" to inteligentny asystent AI zaprojektowany do automatycznej transkrypcji, dieryzacji (rozpoznawania mówców) i podsumowywania spotkań. Wykorzystuje zaawansowane modele mowy (Whisper, Pyannote) oraz modele językowe (Google Gemini) do przetwarzania audio, analizy treści i generowania inteligentnych podsumowań oraz odpowiedzi na zapytania użytkownika.

Główne funkcjonalności:
*   Transkrypcja audio na tekst.
*   Dieryzacja (rozpoznawanie i przypisywanie segmentów mowy do konkretnych mówców).
*   Automatyczne generowanie podsumowań spotkań.
*   Interaktywny chatbot AI, który potrafi odpowiadać na pytania dotyczące przetworzonych transkrypcji i podsumowań.
*   Wyszukiwanie semantyczne i SQL po treści transkrypcji.

## 2. Architektura Systemu

System składa się z kilku wzajemnie komunikujących się komponentów, wykorzystujących protokół FastMCP oraz FastAPI do obsługi API i interfejsu użytkownika.

*   **`agent_daemon.py`**: Daemon monitorujący wejście.
*   **`agents/database_agent.py`**: Agent FastMCP odpowiedzialny za operacje na bazie danych SQLite, udostępniający narzędzia do zarządzania transkrypcjami, podsumowaniami i metadanymi plików.
*   **`minimal_mcp_server.py`**: Główny serwer FastAPI i Router LlamaIndex. Działa jako punkt wejścia dla zapytań użytkownika, zarządza wywołaniami narzędzi lokalnych i zdalnych agentów, oraz integruje się z LlamaIndex ReActAgent do inteligentnego routingu zapytań.
*   **`api_app.py`**: Aplikacja FastAPI obsługująca API do przesyłania plików, zarządzania promptami, oraz inne funkcjonalności związane z przetwarzaniem audio.
*   **`chat_agent_app.py`**: Serwer dla interfejsu czatu (chat_ui), wykorzystujący FastMCP do komunikacji z agentami.
*   **`ingestion_service.py`**: Obsługuje proces ingestowania transkrypcji do bazy danych i Qdrant.
*   **`processor.py`**: Moduł odpowiedzialny za przetwarzanie audio (konwersja, transkrypcja, dieryzacja).
*   **`summarizer.py`**: Zawiera logikę do generowania podsumowań i odpowiedzi chatbota przy użyciu modeli LLM (np. Gemini).
*   **`agent_db.py`**: Bezpośrednia warstwa dostępu do bazy danych SQLite.

## 3. Wymagania Systemowe

*   **Python 3.9+** (zalecane użycie środowiska wirtualnego, np. Conda/Miniconda).
*   **FFmpeg**: Niezbędny do przetwarzania plików audio/wideo.
*   **Karta graficzna z CUDA** (opcjonalnie, ale zalecane dla przyspieszenia PyTorch/Whisper/Pyannote).
*   Wystarczająca ilość RAM i VRAM (szczególnie dla dużych modeli Whisper i Pyannote).

## 4. Instalacja

Zaleca się użycie `conda` do zarządzania środowiskiem.

1.  **Sklonuj repozytorium:**
    ```bash
    git clone https://github.com/twoja_nazwa_uzytkownika/sekretarz.git
    cd sekretarz/whisper5.8
    ```

2.  **Utwórz i aktywuj środowisko Conda:**
    ```bash
    conda create -n sekretarz python=3.9 -y
    conda activate sekretarz
    ```

3.  **Zainstaluj PyTorch z obsługą CUDA (lub CPU):**
    *   **Sprawdź swoją wersję CUDA:**
        ```bash
        nvidia-smi
        ```
        (Jeśli nie masz `nvidia-smi` lub karty NVIDIA, możesz pominąć CUDA i zainstalować wersję CPU).

    *   **Dla CUDA 12.1 (przykładowo):**
        ```bash
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --force-reinstall
        ```
    *   **Dla CPU (bez CUDA):**
        ```bash
        conda install pytorch torchvision torchaudio cpuonly -c pytorch --force-reinstall
        ```
    *   **Zweryfikuj instalację PyTorch:**
        ```bash
        python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
        ```
        Powinno zwrócić `True` i wersję CUDA (jeśli masz GPU), lub `False` i `None` (dla CPU).

4.  **Zainstaluj FFmpeg:**
    *   **Na Windowsie:** Pobierz ze strony [ffmpeg.org](https://ffmpeg.org/download.html) i dodaj ścieżkę do `bin` do zmiennej środowiskowej PATH.
    *   **Na Linuksie/macOS:**
        ```bash
        sudo apt update && sudo apt install ffmpeg # Debian/Ubuntu
        brew install ffmpeg # macOS
        ```

5.  **Zainstaluj pozostałe zależności z `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Zainstaluj modele Pyannote:**
    *   Pyannote wymaga tokena uwierzytelniającego Hugging Face. Zaloguj się na konto Hugging Face i zaakceptuj warunki użycia dla modeli:
        *   [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization)
        *   [`pyannote/embedding`](https://huggingface.co/pyannote/embedding)
    *   Następnie, uruchom `huggingface-cli login` i wklej swój token (token z `Settings -> Access Tokens`).
    *   W pliku `config.py` ustaw ścieżkę do modelu diarization, np.:
        ```python
        PYANNOTE_PIPELINE = "pyannote/speaker-diarization"
        ```

## 5. Konfiguracja

Utwórz plik `config.py` w głównym katalogu projektu i wypełnij go niezbędnymi zmiennymi środowiskowymi i ustawieniami. Oto przykład:

```python
# config.py

# Google Gemini API Key
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL_NAME = "gemini-1.5-flash" # lub inny model, np. "gemini-pro"

# Whisper ASR Model
WHISPER_MODEL_SIZE = "small" # lub "base", "medium", "large-v3" (large-v3 wymaga więcej VRAM)

# Embedding Model (dla wyszukiwania semantycznego)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Lekki model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Użycie GPU/CPU dla modelu embeddingowego

# Qdrant (jeśli używasz, choć teraz używamy cosine similarity bezpośrednio)
QDRANT_ENDPOINT = "http://localhost:6333" # Domyślny adres Qdrant
QDRANT_API_KEY = None # Jeśli nie używasz API Key
QDRANT_SEARCH_LIMIT_PER_COLLECTION = 3
QDRANT_SEARCH_TOTAL_LIMIT = 10

# Ustawienia chunkowania
CHUNK_DURATION = 240 # sekundy
CHUNK_OVERLAP = 0 # sekundy

# Diarization Model
PYANNOTE_PIPELINE = "pyannote/speaker-diarization" # Wymaga autoryzacji Hugging Face

# Ustawienie zmiennej środowiskowej dla OMP (problem z wieloma bibliotekami OpenMP)
# Dodaj tę linię przed uruchomieniem serwerów w terminalu, np.
# set KMP_DUPLICATE_LIB_OK=TRUE (Windows cmd)
# $env:KMP_DUPLICATE_LIB_OK="TRUE" (PowerShell)
# export KMP_DUPLICATE_LIB_OK=TRUE (Linux/macOS)
```

**Ważne:** Upewnij się, że masz swój `GOOGLE_API_KEY` (z Google AI Studio lub Google Cloud) i uzupełnij go w `config.py`.

## 6. Uruchamianie Komponentów

System składa się z kilku niezależnych komponentów, które muszą być uruchomione osobno. Zaleca się uruchamianie ich w oddzielnych oknach terminala.

**Przed uruchomieniem każdego komponentu, upewnij się, że Twoje środowisko `sekretarz` jest aktywne:**
```bash
conda activate sekretarz
```
Oraz, że masz ustawioną zmienną środowiskową `KMP_DUPLICATE_LIB_OK`:
```bash
# Dla Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
# Dla Windows Command Prompt
set KMP_DUPLICATE_LIB_OK=TRUE
# Dla Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 6.1. Uruchomienie Database Agent

Otwórz nowy terminal i uruchom:
```bash
python agents/database_agent.py
```
Powinieneś zobaczyć logi informujące o uruchomieniu serwera FastMCP na porcie 8002.

### 6.2. Uruchomienie Routera (API & Chat UI)

Otwórz nowy terminal i uruchom:
```bash
python minimal_mcp_server.py
```
Ten serwer uruchomi FastAPI, załaduje narzędzia lokalne i zdalne (z Database Agent) oraz zainicjuje LlamaIndex ReActAgent. Interfejs czatu będzie dostępny pod adresem `http://127.0.0.1:8000/chat_ui`.

### 6.3. Uruchomienie API App (opcjonalnie, do przesyłania plików)

Jeśli planujesz korzystać z API do przesyłania plików i zarządzania, uruchom ten komponent w osobnym terminalu:
```bash
python api_app.py
```
Interfejs API będzie dostępny pod adresem `http://127.0.0.1:8001`.

## 7. Przykłady Użycia

Po uruchomieniu wszystkich komponentów, możesz przejść do `http://127.0.0.1:8000/chat_ui` w swojej przeglądarce, aby korzystać z interfejsu czatu.

Przykładowe zapytania do chatbota:

*   **Pobieranie wszystkich transkryptów:**
    *   "Wyswietl mi wszystkie transkrypty"
    *   "Pokaż mi listę transkryptów"
*   **Pobieranie najnowszych/najstarszych transkryptów:**
    *   "Wyswietl 3 najnowsze transkrypty"
    *   "Pokaż 5 najstarszych transkryptów"
*   **Wyszukiwanie semantyczne (po temacie, słowach kluczowych):**
    *   "Który transkrypt dotyczy cross-validation?"
    *   "Znajdź fragmenty o sztucznej inteligencji"
    *   "Jaki transkrypt wspomina o budżecie?"
*   **Czat z transkryptami (głębsza analiza):**
    *   "Podsumuj mi spotkanie o projekcie X"
    *   "Co było powiedziane o strategii marketingowej w najnowszym transkrypcie?"
*   **Wyszukiwanie SQL (ekspert):**
    *   "Wykonaj zapytanie SELECT * FROM processed_files LIMIT 2"

## 8. Struktura Projektu
├── agent_daemon.py # Daemon do monitorowania i zarządzania
├── agent_db.py # Bezpośrednia warstwa dostępu do bazy SQLite
├── agent_files.db # Plik bazy danych SQLite
├── agents/ # Katalog dla agentów FastMCP
│ ├── audio_processing_agent.py
│ ├── database_agent.py # Agent bazy danych FastMCP
│ └── hello.py # Plik testowy
├── api_app.py # Aplikacja FastAPI do zarządzania plikami i promptami
├── chat_agent_app.py # Serwer dla interfejsu czatu
├── chat_interface.html # Interfejs użytkownika czatu
├── chunker.py # Moduł do dzielenia tekstu na chunki
├── config.py # Plik konfiguracyjny (API klucze, modele, etc.)
├── data_handler.py
├── file_handlers.py
├── index.html
├── ingest.py # Logika ingestowania danych
├── ingestion_service.py # Serwis ingestowania
├── input_watch/
├── main.py
├── meeting_minutes/ # Katalog na wygenerowane podsumowania spotkań
├── minimal_mcp_server.py # Główny Router FastMCP i LlamaIndex ReActAgent
├── minutes_service.py # Serwis generowania minut
├── models/ # Modele AI (Whisper, Pyannote embeddings)
│ ├── base.pt
│ └── large-v3.pt
├── models.py # Definicje modeli Pydantic/danych
├── outdated_packages.txt
├── output_1min.mp4
├── output_api_transcripts/ # Katalog na przetworzone transkrypcje
├── output.mkv
├── processing_service.py # Serwis przetwarzania audio
├── processor.py # Logika przetwarzania audio
├── prompt-setting.json
├── qdrant_handler.py # Obsługa Qdrant (obecnie pomijana na rzecz cosine similarity)
├── qdrant_uploader.py # Uploadowanie danych do Qdrant
├── README.md # Ten plik
├── requirements.txt # Zależności projektu
├── run.py # Skrypt do uruchamiania komponentów
├── scripts/
│ └── qdrant_remover.py
├── speaker_enrollment/ # Profile zarejestrowanych mówców
├── srt_parser.py # Parser plików SRT
├── static/
├── summarizer.py # Logika podsumowywania i generowania odpowiedzi LLM
├── temp_audio_segments/
├── test_pyannote.py
├── test_pyannotepy.txt
├── test_ram.py
├── test.html
├── test.py
├── uploads/ # Katalog na przesyłane pliki
└── utils.py # Funkcje pomocnicze
## 9. Rozwiązywanie Problemów

*   **`ImportError: cannot import name 'XYZ' from 'agents.database_agent'`**: Upewnij się, że usunąłeś wszystkie zbędne importy w `minimal_mcp_server.py` z `agents.database_agent` i pozostawiłeś tylko `mcp_database`.
*   **`OMP: Error #15` / Błędy związane z bibliotekami OpenMP**: Ustaw zmienną środowiskową `KMP_DUPLICATE_LIB_OK` na `TRUE` przed uruchomieniem skryptów Python (instrukcje w sekcji [6. Uruchamianie Komponentów](#6-uruchamianie-komponentów)).
*   **`ConnectionResetError` / Problemy z połączeniem FastMCP**: Upewnij się, że wszystkie serwery (Database Agent, Router, API App) są uruchomione i działają poprawnie. Spróbuj zrestartować je w odpowiedniej kolejności (najpierw Database Agent).
*   **`RuntimeError: FFmpeg extension is not available.`**: Upewnij się, że FFmpeg jest zainstalowany i jego katalog `bin` jest dodany do zmiennej środowiskowej PATH.
*   **Agent nie wybiera poprawnego narzędzia lub źle przekazuje parametry**:
    *   Sprawdź `system_prompt` w `minimal_mcp_server.py` i upewnij się, że instrukcje są jasne i kategoryczne.
    *   Zweryfikuj sygnatury funkcji narzędzi w `agents/database_agent.py` i upewnij się, że nie używają już `params: BaseModel` do opakowywania argumentów.
    *   Jeśli problem nadal występuje, może to być związane z wewnętrznym działaniem LlamaIndex i wymagać dalszego debugowania lub dostosowania podejścia do narzędzi.
