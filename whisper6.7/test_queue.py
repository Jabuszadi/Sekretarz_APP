"""
Skrypt testowy do sprawdzania systemu kolejkowania wiadomości.
Wysyła kilka requestów jednocześnie i sprawdza ich status.
"""
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional

# Konfiguracja
# Endpoint /chat/query jest teraz w api_app.py na porcie 7777 (uruchamianym przez run.py)
# Port 8000 to minimal_mcp_server.py (uruchamiany osobno, jeśli potrzebny)
API_BASE_URL = "http://localhost:7777"  # api_app.py - główny serwer z kolejką
TEST_QUERIES = [
    "Co to jest Python?",
    "Jak działa system kolejkowania?",
    "Wyjaśnij mi czym jest PostgreSQL",
    "Opowiedz mi o FastAPI",
    "Co to jest pgBoss?",
]

# Sprawdź czy serwer działa przed rozpoczęciem testów
async def check_server_health(session: aiohttp.ClientSession) -> bool:
    """Sprawdza czy serwer jest dostępny."""
    try:
        # Sprawdź endpoint /health (jeśli istnieje) lub główny endpoint
        async with session.get(f"{API_BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
            return response.status < 500
    except aiohttp.ClientConnectorError:
        return False
    except Exception:
        # Spróbuj sprawdzić główny endpoint
        try:
            async with session.get(f"{API_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status < 500
        except:
            return False


async def send_chat_query(session: aiohttp.ClientSession, query: str, token: Optional[str] = None) -> Optional[str]:
    """Wysyła zapytanie do API i zwraca job_id."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        async with session.post(
            f"{API_BASE_URL}/chat/query",
            json={"query": query},
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                # Wyciągnij job_id z odpowiedzi
                response_text = data.get("response", "")
                import re
                job_id_match = re.search(r"ID zadania: ([a-f0-9-]+)", response_text, re.IGNORECASE)
                if job_id_match:
                    return job_id_match.group(1)
                print(f"⚠️  Nie znaleziono job_id w odpowiedzi: {response_text}")
                return None
            else:
                # Pobierz szczegóły błędu
                try:
                    error_data = await response.json()
                    error_detail = error_data.get('detail', f'HTTP {response.status}')
                except:
                    error_text = await response.text()
                    error_detail = f'HTTP {response.status}: {error_text[:200]}'
                print(f"❌ Błąd przy wysyłaniu zapytania '{query}': {error_detail}")
                print(f"   URL: {API_BASE_URL}/chat/query")
                print(f"   Status: {response.status}")
                return None
    except Exception as e:
        print(f"❌ Wyjątek przy wysyłaniu zapytania '{query}': {e}")
        return None


async def check_job_status(session: aiohttp.ClientSession, job_id: str, token: Optional[str] = None) -> Optional[Dict]:
    """Sprawdza status zadania."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        async with session.get(
            f"{API_BASE_URL}/chat/query/status/{job_id}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return None
    except Exception as e:
        print(f"❌ Błąd przy sprawdzaniu statusu {job_id}: {e}")
        return None


async def get_job_response(session: aiohttp.ClientSession, job_id: str, token: Optional[str] = None) -> Optional[str]:
    """Pobiera odpowiedź na zadanie."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        async with session.get(
            f"{API_BASE_URL}/chat/query/response/{job_id}",
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("response", "")
            return None
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu odpowiedzi {job_id}: {e}")
        return None


async def test_queue_system():
    """Główna funkcja testująca system kolejkowania."""
    print("🚀 Rozpoczynam test systemu kolejkowania...\n")
    
    # Sprawdź czy serwer działa
    print("🔍 Sprawdzanie dostępności serwera...")
    async with aiohttp.ClientSession() as session:
        if not await check_server_health(session):
            print(f"\n❌ BŁĄD: Nie można połączyć się z serwerem na {API_BASE_URL}")
            print("\n💡 INSTRUKCJA:")
            print("   1. Uruchom serwer w osobnym terminalu:")
            print("      python minimal_mcp_server.py")
            print("\n   2. Lub użyj uvicorn bezpośrednio:")
            print("      uvicorn minimal_mcp_server:app --host 0.0.0.0 --port 8000 --reload")
            print("\n   3. Poczekaj aż zobaczysz: '✅ pgBoss uruchomiony'")
            print("\n   4. Następnie uruchom ponownie ten test")
            return
    
    print("✅ Serwer jest dostępny!\n")
    
    # Token autoryzacji (opcjonalny - usuń jeśli nie używasz autoryzacji)
    token = None  # Wstaw tutaj token jeśli potrzebujesz autoryzacji
    
    async with aiohttp.ClientSession() as session:
        # Krok 1: Wyślij wszystkie zapytania jednocześnie
        print("📤 Wysyłanie zapytań do kolejki...")
        tasks = [send_chat_query(session, query, token) for query in TEST_QUERIES]
        job_ids = await asyncio.gather(*tasks)
        
        # Filtruj None (błędy)
        valid_job_ids = [jid for jid in job_ids if jid is not None]
        
        print(f"✅ Wysłano {len(valid_job_ids)}/{len(TEST_QUERIES)} zapytań pomyślnie\n")
        print(f"📋 Job IDs: {valid_job_ids}\n")
        
        if not valid_job_ids:
            print("❌ Brak poprawnych job_id. Sprawdź czy serwer działa i czy endpoint jest dostępny.")
            return
        
        # Krok 2: Monitoruj status zadań
        print("⏳ Monitorowanie statusu zadań...\n")
        completed_jobs = set()
        max_wait_time = 300  # Maksymalny czas oczekiwania: 5 minut
        start_time = time.time()
        check_interval = 2  # Sprawdzaj co 2 sekundy
        
        while len(completed_jobs) < len(valid_job_ids) and (time.time() - start_time) < max_wait_time:
            for i, job_id in enumerate(valid_job_ids):
                if job_id in completed_jobs:
                    continue
                
                status_data = await check_job_status(session, job_id, token)
                if status_data:
                    status = status_data.get("status", "unknown")
                    retry_count = status_data.get("retry_count", 0)
                    
                    status_emoji = {
                        "pending": "⏳",
                        "processing": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                        "retrying": "🔄",
                    }.get(status, "❓")
                    
                    print(f"{status_emoji} Job {job_id[:8]}... | Status: {status} | Próba: {retry_count + 1}")
                    
                    if status == "completed":
                        completed_jobs.add(job_id)
                        response = await get_job_response(session, job_id, token)
                        if response:
                            preview = response[:100].replace("\n", " ") + "..." if len(response) > 100 else response
                            print(f"   📝 Odpowiedź: {preview}\n")
                    elif status == "failed":
                        completed_jobs.add(job_id)
                        error_msg = status_data.get("error_message", "Unknown error")
                        print(f"   ❌ Błąd: {error_msg}\n")
            
            if len(completed_jobs) < len(valid_job_ids):
                await asyncio.sleep(check_interval)
                print()  # Pusta linia dla czytelności
        
        # Podsumowanie
        print("\n" + "="*60)
        print("📊 PODSUMOWANIE TESTU")
        print("="*60)
        print(f"Wysłano zapytań: {len(valid_job_ids)}")
        print(f"Zakończone: {len(completed_jobs)}")
        print(f"Oczekujące: {len(valid_job_ids) - len(completed_jobs)}")
        
        if len(completed_jobs) == len(valid_job_ids):
            print("\n✅ Wszystkie zadania zostały przetworzone!")
        else:
            print(f"\n⚠️  Nie wszystkie zadania zostały zakończone w czasie {max_wait_time}s")


if __name__ == "__main__":
    print("="*60)
    print("TEST SYSTEMU KOLEJKOWANIA WIADOMOŚCI")
    print("="*60)
    print(f"API URL: {API_BASE_URL}")
    print(f"Liczba testowych zapytań: {len(TEST_QUERIES)}\n")
    
    try:
        asyncio.run(test_queue_system())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test przerwany przez użytkownika")
    except Exception as e:
        print(f"\n\n❌ Błąd podczas testu: {e}")
        import traceback
        traceback.print_exc()

