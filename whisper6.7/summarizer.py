# summarizer.py
import asyncio
import json
import logging
from html import escape
from typing import Callable, Dict, List, Optional

import google.generativeai as genai
import torch

from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME, QUERY_SECTIONS

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

try:
    _GENERATION_CONFIG = genai.GenerationConfig(temperature=0.3)
except AttributeError:
    _GENERATION_CONFIG = {"temperature": 0.3}

try:
    gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config=_GENERATION_CONFIG,
    )
    logging.info("✅ Successfully initialized Gemini model: %s", GEMINI_MODEL_NAME)
except Exception as exc:
    logging.error("❌ Failed to initialize Gemini model (%s): %s", GEMINI_MODEL_NAME, exc)
    raise


_METADATA_PREFIXES = (
    "wygenerowano",
    "model llm",
    "model embeddings",
    "model transkrypcji",
    "model diarizacji",
    "metoda",
    "narzędzie",
    "czas",
    "sprawdzenie",
)


def _extract_summary_components(summary_text: str) -> tuple[List[str], List[Dict[str, str]], List[str]]:
    """
    Dzieli tekst minut na linie ogólne, zadania i metadane.
    """
    lines: List[str] = []
    tasks: List[Dict[str, str]] = []
    metadata: List[str] = []
    current_task: Optional[Dict[str, str]] = None
    previous_blank = False

    def flush_task() -> None:
        nonlocal current_task
        if current_task:
            tasks.append(current_task)
            current_task = None

    for raw_line in summary_text.splitlines():
        line = raw_line.strip()

        # Pomijamy fence'y Markdown
        if line.startswith("```"):
            flush_task()
            continue

        if not line:
            flush_task()
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue

        previous_blank = False
        lowered = line.lower()

        if current_task and not lowered.startswith(("zadanie:", "osoba odpowiedzialna:", "termin:", "status:")):
            flush_task()

        if lowered.startswith("zadanie:"):
            flush_task()
            current_task = {
                "title": line.split(":", 1)[1].strip() if ":" in line else line.removeprefix("Zadanie").strip(),
            }
            continue

        if current_task:
            if lowered.startswith("osoba odpowiedzialna:"):
                current_task["owner"] = line.split(":", 1)[1].strip()
                continue
            if lowered.startswith("termin:"):
                current_task["deadline"] = line.split(":", 1)[1].strip()
                continue
            if lowered.startswith("status:"):
                current_task["status"] = line.split(":", 1)[1].strip()
                continue

        if lowered.startswith(_METADATA_PREFIXES):
            flush_task()
            metadata.append(line)
            continue

        lines.append(line)

    flush_task()
    return lines, tasks, metadata


def _render_summary_as_html(summary_text: str) -> str:
    """
    Konwertuje tekstowe podsumowanie na czytelny dokument HTML bez użycia modelu.
    """
    lines, tasks, metadata_lines = _extract_summary_components(summary_text)
    content_parts: List[str] = []
    in_list = False
    title_rendered = False

    def close_list_if_needed() -> None:
        nonlocal in_list
        if in_list:
            content_parts.append("</ul>")
            in_list = False

    def render_tasks_grid() -> str:
        if not tasks:
            return ""
        grid_parts: List[str] = ['<div class="tasks-grid">']
        for task in tasks:
            title = escape(task.get("title") or "Zadanie bez nazwy")
            owner = task.get("owner")
            deadline = task.get("deadline")
            status = task.get("status")

            grid_parts.append('<div class="task-card">')
            grid_parts.append(f"<h3>{title}</h3>")
            if owner:
                grid_parts.append(f"<p><span>Odpowiedzialny:</span> {escape(owner)}</p>")
            if deadline:
                grid_parts.append(f"<p><span>Termin:</span> {escape(deadline)}</p>")
            if status:
                grid_parts.append(f"<p><span>Status:</span> {escape(status)}</p>")
            grid_parts.append("</div>")
        grid_parts.append("</div>")
        return "\n".join(grid_parts)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_list_if_needed()
            continue

        # Pierwsza linia traktowana jako tytuł dokumentu
        if not title_rendered:
            content_parts.append(f"<h1>{escape(line)}</h1>")
            title_rendered = True
            continue

        if line.startswith("- "):
            if not in_list:
                content_parts.append("<ul>")
                in_list = True
            content_parts.append(f"<li>{escape(line[2:].strip())}</li>")
            continue
        else:
            close_list_if_needed()

        if line.endswith(":") and line.count(":") == 1:
            heading_text = line[:-1].strip()
            content_parts.append(f"<h2>{escape(heading_text)}</h2>")
            if tasks and heading_text.lower().startswith("lista zadań"):
                content_parts.append(render_tasks_grid())
            continue

        if ":" in line:
            label, rest = line.split(":", 1)
            content_parts.append(
                f"<p><span class=\"label\">{escape(label.strip())}:</span> {escape(rest.strip() or 'Brak danych')}</p>"
            )
        else:
            content_parts.append(f"<p>{escape(line)}</p>")

    close_list_if_needed()

    if tasks and not any("tasks-grid" in fragment for fragment in content_parts):
        content_parts.append("<h2>Lista zadań</h2>")
        content_parts.append(render_tasks_grid())

    if metadata_lines:
        content_parts.append('<div class="minutes-meta"><h3>Metadane</h3><ul class="meta-list">')
        for entry in metadata_lines:
            if ":" in entry:
                label, rest = entry.split(":", 1)
                content_parts.append(
                    f"<li><span class=\"meta-label\">{escape(label.strip())}:</span> {escape(rest.strip())}</li>"
                )
            else:
                content_parts.append(f"<li>{escape(entry)}</li>")
        content_parts.append("</ul></div>")

    body_content = "\n".join(fragment for fragment in content_parts if fragment.strip())

    style_block = """
    :root { color-scheme: light; }
    body {
        margin: 0;
        padding: 48px 24px;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        background: #f3f4f6;
        color: #1f2937;
    }
    .minutes-container {
        max-width: 920px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 20px;
        padding: 44px 52px;
        box-shadow: 0 32px 60px -20px rgba(30, 64, 175, 0.35);
    }
    h1 {
        margin: 0 0 24px 0;
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
    }
    h2 {
        margin: 36px 0 16px;
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #1d4ed8;
    }
    h3 {
        margin-top: 0;
        font-size: 18px;
        color: #1d4ed8;
    }
    p {
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.7;
    }
    .label {
        font-weight: 600;
        color: #1f2937;
        margin-right: 4px;
    }
    ul {
        margin: 12px 0 16px 1.5rem;
        padding: 0;
    }
    li {
        margin: 4px 0;
    }
    .tasks-grid {
        margin-top: 12px;
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .task-card {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 18px 30px -18px rgba(37, 99, 235, 0.45);
    }
    .task-card p {
        margin: 6px 0;
        font-size: 15px;
    }
    .task-card span {
        font-weight: 600;
        color: #1e40af;
    }
    .minutes-meta {
        margin-top: 36px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }
    .minutes-meta h3 {
        margin: 0 0 12px 0;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #6b7280;
    }
    .meta-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .meta-list li {
        margin: 6px 0;
        font-size: 14px;
        color: #4b5563;
    }
    .meta-label {
        font-weight: 600;
        color: #1f2937;
        margin-right: 6px;
    }
    @media (max-width: 640px) {
        .minutes-container {
            padding: 32px 24px;
        }
        h1 {
            font-size: 26px;
        }
    }
    """

    return (
        "<!DOCTYPE html>"
        "<html lang=\"pl\">"
        "<head>"
        "<meta charset=\"UTF-8\" />"
        "<title>Podsumowanie Spotkania</title>"
        f"<style>{style_block}</style>"
        "</head>"
        "<body>"
        f"<div class=\"minutes-container\">{body_content}</div>"
        "</body>"
        "</html>"
    )


def _extract_text_from_response(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    try:
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return part_text.strip()
    except Exception as parse_error:  # pragma: no cover - ostrożna degradacja
        logging.debug("Failed to parse Gemini response text: %s", parse_error)

    return ""


def _generate_text_sync(prompt: str, model=None) -> str:
    active_model = model or gemini_model
    response = active_model.generate_content(prompt)
    return _extract_text_from_response(response)


async def _generate_text(prompt: str, model=None) -> str:
    return await asyncio.to_thread(_generate_text_sync, prompt, model)


async def generate_minutes_of_meeting(
    context: str,
    custom_prompt=None,
    progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    api_key_override: Optional[str] = None,
) -> dict:
    results: Dict[str, str] = {}
    query_sections = {}

    override_model = None
    if api_key_override:
        override_key = api_key_override.strip()
        if override_key:
            try:
                override_client = genai.Client(api_key=override_key)
                override_model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL_NAME,
                    client=override_client,
                    generation_config=_GENERATION_CONFIG,
                )
            except Exception as override_error:
                logging.error(
                    "❌ Failed to initialize Gemini model with override API key: %s",
                    override_error,
                )
                raise

    model_for_generation = override_model or gemini_model

    if custom_prompt is None:
        query_sections = QUERY_SECTIONS
    elif isinstance(custom_prompt, str):
        prompt_text = custom_prompt.strip()
        if not prompt_text:
            query_sections = {}
        else:
            try:
                parsed_prompt = json.loads(prompt_text)
                if isinstance(parsed_prompt, dict):
                    query_sections = parsed_prompt
                elif isinstance(parsed_prompt, list):
                    query_sections = {
                        f"Sekcja {idx + 1}": str(item)
                        for idx, item in enumerate(parsed_prompt)
                        if str(item).strip()
                    }
                else:
                    logging.warning(
                        "Custom prompt JSON was valid but unexpected type (%s). Generating pojedynczą sekcję z tekstu.",
                        type(parsed_prompt),
                    )
                    query_sections = {"Sekcja niestandardowa": prompt_text}
            except json.JSONDecodeError:
                logging.info(
                    "Custom prompt nie jest poprawnym JSON-em. Interpretuję go jako pojedynczą sekcję tekstową."
                )
                lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
                if len(lines) <= 1:
                    query_sections = {"Sekcja niestandardowa": prompt_text}
                else:
                    query_sections = {
                        f"Sekcja {idx + 1}": line for idx, line in enumerate(lines)
                    }
            except Exception as unexpected:
                logging.error(
                    "❌ Nieoczekiwany błąd podczas przetwarzania stringu custom promptu: %s. Custom prompt: '%s'. Generowanie pojedynczej sekcji.",
                    unexpected,
                    custom_prompt,
                )
                query_sections = {"Sekcja niestandardowa": prompt_text}
    elif isinstance(custom_prompt, dict):
        query_sections = custom_prompt
    else:
        logging.warning("Nieoczekiwany typ dla custom_prompt: %s. Generowanie pustych sekcji.", type(custom_prompt))
        query_sections = {}

    total_sections = len(query_sections)
    processed_sections = 0

    if total_sections == 0:
        logging.info("Brak sekcji do wygenerowania minut. Oznaczam postęp jako 100%.")
        if progress_callback:
            progress_callback(100.0, None)
        return results

    for section_title, query in query_sections.items():
        current_index = processed_sections + 1
        logging.info("[Minutes] Generowanie sekcji %s/%s: '%s'", current_index, total_sections, section_title)
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
- **WAŻNE:** W transkrypcji imiona mówców są podane w formacie "[Imię]: [tekst]". Używaj tych imion w odpowiedzi zamiast ogólnych określeń typu "Mówiący" czy "Speaker".
"""
        try:
            text = await _generate_text(prompt, model_for_generation)
            text = await _generate_text(prompt, model_for_generation)
            results[section_title] = text or "(Brak informacji w podanym fragmencie transkrypcji.)"
            logging.info("  ✅ Generated section: '%s'", section_title)

            await asyncio.sleep(5)

            processed_sections += 1
            if progress_callback and total_sections > 0:
                current_progress = (processed_sections / total_sections) * 100
                logging.info(
                    "[Minutes] Postęp generowania minut: %.1f%% (sekcja '%s')",
                    current_progress,
                    section_title,
                )
                progress_callback(current_progress, section_title)
        except Exception as exc:
            logging.error("❌ Error generating section '%s' with Gemini: %s", section_title, exc)
            results[section_title] = "(Wystąpił błąd podczas generowania odpowiedzi dla tej sekcji.)"

            processed_sections += 1
            if progress_callback and total_sections > 0:
                current_progress = (processed_sections / total_sections) * 100
                logging.info(
                    "[Minutes] Postęp generowania minut (po błędzie): %.1f%% (sekcja '%s')",
                    current_progress,
                    section_title,
                )
                progress_callback(current_progress, section_title)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(100.0, None)

    return results


async def generate_html_from_text(summary_text: str) -> str:
    try:
        logging.info("Rendering HTML summary using deterministic formatter.")
        return _render_summary_as_html(summary_text)
    except Exception as exc:
        logging.error("❌ Error generating HTML summary: %s", exc, exc_info=True)
        return (
            "<!DOCTYPE html>"
            "<html lang=\"pl\"><head><meta charset=\"UTF-8\" />"
            "<title>Podsumowanie Spotkania (fallback)</title></head>"
            "<body><pre>"
            f"{escape(summary_text)}"
            "</pre></body></html>"
        )


async def generate_chat_response(user_query: str, context_str: str) -> str:
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
        return await _generate_text(prompt) or "Przepraszam, nie udało się wygenerować odpowiedzi."
    except Exception as exc:
        logging.error("Error generating chat response with Gemini: %s", exc)
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."