# summarizer.py
import logging
import time # Import the time module
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import torch  # Dodaj na górze pliku
import asyncio # Needed for async operations like asyncio.to_thread
from typing import List, Optional # Needed for type hints

# Import configuration variables needed here
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME, QUERY_SECTIONS

# --- Initialize Gemini Model ---
try:
    gemini_model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    logging.info(f"✅ Successfully initialized Gemini model: {GEMINI_MODEL_NAME}")
except Exception as e:
    logging.error(f"❌ Failed to initialize Gemini model ({GEMINI_MODEL_NAME}): {e}")
    # Decide how to handle this - raise or return an indicator? Raising is safer.
    raise

# --- Minutes Generation Function ---

def generate_minutes_of_meeting(context: str, custom_prompt=None) -> dict:
    """Generates meeting minutes sections using the initialized Gemini LLM."""
    results = {}
    query_sections = {} # Domyślna wartość, jeśli custom_prompt jest podany, ale niepoprawny/pusty

    if custom_prompt is None:
        # Użytkownik nie dostarczył własnego promptu, użyj domyślnych sekcji
        query_sections = QUERY_SECTIONS
    elif isinstance(custom_prompt, str):
        # Użytkownik dostarczył string, spróbuj sparsować jako JSON
        if custom_prompt.strip() == "":
            # Pusty string oznacza, że użytkownik nie chce żadnych niestandardowych sekcji
            query_sections = {}
        else:
            try:
                parsed_prompt = json.loads(custom_prompt)
                if isinstance(parsed_prompt, dict):
                    # Poprawny słownik JSON dostarczony przez użytkownika, użyj go
                    query_sections = parsed_prompt
                else:
                    # JSON był poprawny, ale nie był słownikiem (np. "true", "[1,2]"), traktuj jako pusty zamiast domyślnych
                    logging.warning(f"Custom prompt JSON was valid but not a dictionary: '{custom_prompt}'. Generating no custom sections.")
                    query_sections = {}
            except json.JSONDecodeError as e:
                # Niepoprawna składnia JSON, traktuj jako pusty zamiast domyślnych
                logging.error(f"❌ Błąd parsowania JSON dla custom promptu: {e}. Custom prompt: '{custom_prompt}'. Generowanie pustych sekcji.")
                query_sections = {}
            except Exception as e:
                # Inny nieoczekiwany błąd podczas parsowania stringu
                logging.error(f"❌ Nieoczekiwany błąd podczas przetwarzania stringu custom promptu: {e}. Custom prompt: '{custom_prompt}'. Generowanie pustych sekcji.")
                query_sections = {}
    elif isinstance(custom_prompt, dict):
        # custom_prompt jest już słownikiem (np. z wywołań wewnętrznych), użyj go bezpośrednio
        query_sections = custom_prompt
    else:
        # Nieoczekiwany typ custom_prompt
        logging.warning(f"Nieoczekiwany typ dla custom_prompt: {type(custom_prompt)}. Generowanie pustych sekcji.")
        query_sections = {}

    for section_title, query in query_sections.items():
        prompt = f"""
Na podstawie poniższego fragmentu transkrypcji spotkania, odpowiedz na pytanie: "{query}"

**Fragment Transkrypcji:**
---
{context}
---

**Oczekiwana odpowiedź (tylko treść, bez wstępów):**
- Twoja odpowiedź ma być *wyłącznie* bezpośrednią, zwięzłą i rzeczową reakcją na pytanie, opartą *jedynie* na informacjach zawartych w podanym fragmencie transkrypcji.
- **NIE DODAJ ŻADNYCH WSTĘPÓW, TYTUŁÓW CZY FORMULACJI TYPU "Na podstawie...", "Oto lista...", "W filmie omówiono...". Zacznij odpowiedź od razu od pierwszego punktu lub zdania.**
- Jeśli brak jest informacji potrzebnych do odpowiedzi, Twoja odpowiedź powinna brzmieć dokładnie: "(Brak informacji w podanym fragmencie transkrypcji.)".
- Użyj punktorów, jeśli odpowiedź wymaga listowania.
"""
        try:
            # Use invoke for synchronous call with the initialized model
            response = gemini_model.invoke(prompt)
            # Access the content of the AIMessage response
            results[section_title] = response.content.strip()
            logging.info(f"  ✅ Generated section: '{section_title}'")

            # Add a delay after each successful API call to avoid rate limits
            logging.debug("  ⏲️ Waiting 5 seconds before next API call...")
            time.sleep(5) # Sleep for 5 seconds (12 requests per minute)

            # Czyść GPU po każdej sekcji
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"❌ Error generating section '{section_title}' with Gemini: {e}")
            results[section_title] = "(Wystąpił błąd podczas generowania odpowiedzi dla tej sekcji.)"
            # Optionally add a shorter sleep even after an error, if needed
            # logging.debug("  ⏲️ Waiting briefly after error...")
            # time.sleep(1)

    return results

def generate_html_from_text(summary_text: str) -> str:
    """Generates an HTML representation of the provided summary text using the initialized Gemini LLM."""
    prompt = f"""
Poniżej znajduje się tekst podsumowania spotkania. Przekształć ten tekst w atrakcyjny wizualnie i czytelny dokument HTML.
Użyj odpowiednich tagów HTML, takich jak <h1> dla tytułu, <h2> dla nagłówków sekcji, <p> dla akapitów i <ul>/<li> dla list.
Możesz użyć tagów <strong> lub <em> dla wyróżnienia ważnych fragmentów.
Dodaj sekcję <head> z tagami <meta charset="UTF-8"> i <title>Uporządkowane Podsumowanie Spotkania</title>.
Nie dołączaj żadnych zewnętrznych arkuszy stylów ani skryptów JavaScript. Skup się wyłącznie na strukturze i semantyce HTML.
Zacznij odpowiedź od znacznika `<!DOCTYPE html>`.
---
{summary_text}
---
"""
    try:
        logging.info("Generating HTML from summary text...")
        response = gemini_model.invoke(prompt)
        html_content = response.content.strip()

        # Simple validation to ensure it looks like HTML
        if not html_content.startswith("<!DOCTYPE html>") and not html_content.startswith("<html"):
            logging.warning("Generated content does not seem to be a valid HTML document. Attempting to wrap it.")
            html_content = f"<!DOCTYPE html>\n<html lang=\"pl\">\n<head>\n<meta charset=\"UTF-8\">\n<title>Podsumowanie Spotkania</title>\n</head>\n<body>\n{html_content}\n</body>\n</html>"

        logging.info("✅ Successfully generated HTML content.")
        return html_content
    except Exception as e:
        logging.error(f"❌ Error generating HTML from summary text with Gemini: {e}")
        return f"<p>Wystąpił błąd podczas generowania HTML: {e}</p>"

# Zmodyfikowana funkcja, aby przyjmowała kontekst bezpośrednio
async def generate_chat_response(user_query: str, context_str: str) -> str:
    """
    Generates an AI Agent's response based on the user's query and provided context documents.
    """
    if not gemini_model:
        logging.error("Gemini model not initialized. Cannot generate chat response.")
        return "Przepraszam, model AI nie jest dostępny. Spróbuj ponownie później."

    if not context_str:
        return "Nie znalazłem żadnych informacji w bazie danych, które odpowiadałyby na Twoje pytanie."

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
        response = await asyncio.to_thread(gemini_model.invoke, prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error generating chat response with Gemini: {e}")
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."