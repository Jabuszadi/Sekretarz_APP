"""
Moduł do zarządzania kolejką wiadomości używając Dramatiq.
Dramatiq to profesjonalny system kolejkowania zadań dla Python.

WAŻNE: Aktory muszą być zdefiniowane na poziomie modułu, żeby worker mógł je zaimportować.
Na Windows może być problem z multiprocessing - upewnij się, że worker widzi aktory.
"""
import logging
import os
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from dotenv import load_dotenv

# WAŻNE: Eksportuj aktory, żeby worker mógł je zaimportować
__all__ = ['process_chat_message', 'process_batch', 'process_single_file', 'get_broker', 'enqueue_message', 'enqueue_batch_processing', 'enqueue_single_file_processing']

# Załaduj zmienne środowiskowe (bez importowania całego config.py, który importuje torch)
load_dotenv()

# Import dramatiq BEZPOŚREDNIO - nie używamy try/except, bo jeśli dramatiq nie jest zainstalowany,
# to worker i tak nie zadziała, więc lepiej od razu rzucić błąd
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
import redis

logger = logging.getLogger(__name__)

_process_callback: Optional[Callable] = None
_broker: Optional["RedisBroker"] = None
_result_backend: Optional["RedisBackend"] = None
_actor: Optional["dramatiq.Actor"] = None
_batch_actor: Optional["dramatiq.Actor"] = None
_file_actor: Optional["dramatiq.Actor"] = None


# Aktor Dramatiq musi być zdefiniowany na poziomie modułu, żeby worker mógł go zaimportować
# Funkcja jest zdefiniowana jako zwykła funkcja, a potem będzie zadekorowana dekoratorem @dramatiq.actor
def _process_chat_message_impl(username: str, query: str, collection_name: Optional[str] = None):
    """Aktor Dramatiq do przetwarzania wiadomości."""
    import asyncio
    
    logger.info(f"Przetwarzanie wiadomości (użytkownik: {username})")
    
    try:
        # Import funkcji potrzebnych do przetwarzania
        # Używamy lazy import, żeby uniknąć problemów z importami przy starcie modułu
        from qdrant_handler import search_all_collections, initialize_qdrant_resources
        from summarizer import generate_chat_response
        
        # Pobierz wartości z zmiennych środowiskowych (bez importowania config.py, który importuje torch)
        qdrant_limit_per_collection = int(os.getenv("QDRANT_SEARCH_LIMIT_PER_COLLECTION", "5"))
        qdrant_total_limit = int(os.getenv("QDRANT_SEARCH_TOTAL_LIMIT", "20"))
        
        # Dramatiq worker działa w osobnym wątku/procesie, więc tworzymy nowy event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Przetwórz wiadomość - ta sama logika co w api_app.py
        async def process_async():
            # Inicjalizuj Qdrant client i embedding model (jeśli nie są już zainicjalizowane)
            # Worker działa w osobnym procesie, więc musi zainicjalizować zasoby samodzielnie
            try:
                await initialize_qdrant_resources()
            except Exception as e:
                logger.warning(f"Qdrant resources may already be initialized or initialization failed: {e}")
            
            # Wyszukaj dokumenty w Qdrant
            relevant_documents = await search_all_collections(
                query,
                limit_per_collection=qdrant_limit_per_collection,
                total_limit=qdrant_total_limit,
                target_collection=collection_name
            )
            
            # Przygotuj kontekst z dokumentów
            context_documents = [doc.content for doc in relevant_documents]
            context_str = "\n".join(context_documents) if context_documents else ""
            
            # Wygeneruj odpowiedź używając generate_chat_response
            response = await generate_chat_response(query, context_str)
            return response
        
        # Uruchom async funkcję
        response = loop.run_until_complete(process_async())
        
        return {"response": response}
    except Exception as e:
        error_message = str(e)
        logger.error(f"Błąd podczas przetwarzania wiadomości: {error_message}", exc_info=True)
        return {"error": error_message}


def _process_batch_impl(batch_job_id: str):
    """Aktor Dramatiq do przetwarzania batchów plików."""
    import asyncio
    import json
    from file_handlers import job_temp_storage
    import agent_db
    
    # WAŻNE: Ten log powinien się pojawić gdy worker przetwarza zadanie
    logger.info(f"🚀 [DRAMATIQ WORKER] Przetwarzanie batcha: {batch_job_id}")
    logger.info(f"🚀 [DRAMATIQ WORKER] Actor process_batch wywołany dla batch_job_id: {batch_job_id}")
    
    try:
        # Import funkcji potrzebnych do przetwarzania
        # Używamy lazy import, żeby uniknąć problemów z importami przy starcie modułu
        import sys
        from pathlib import Path
        
        # Dodaj ścieżkę do api_app.py jeśli nie jest już dodana
        api_app_path = Path(__file__).parent
        if str(api_app_path) not in sys.path:
            sys.path.insert(0, str(api_app_path))
        
        # Import funkcji z api_app.py
        # Musimy to zrobić w funkcji async, żeby uniknąć problemów z importami
        async def process_batch_async():
            # Fallback: jeśli job_temp_storage nie ma wpisu (worker w osobnym procesie)
            if batch_job_id not in job_temp_storage:
                logger.warning(f"[{batch_job_id}] job_temp_storage is empty for this batch; reconstructing from DB")
                batch_db_record = agent_db.get_batch_job(batch_job_id)
                if batch_db_record:
                    file_job_ids_from_db = json.loads(batch_db_record.get('file_job_ids_json', '[]'))
                    params_from_db = batch_db_record.get('params', {})
                    file_ids_from_db = batch_db_record.get('file_ids', [])
                    job_temp_storage[batch_job_id] = {
                        'type': 'batch',
                        'file_job_ids': file_job_ids_from_db,
                        'status': batch_db_record.get('status', 'uploaded'),
                        'params': params_from_db
                    }
                    logger.info(f"[{batch_job_id}] job_temp_storage reconstructed with {len(file_job_ids_from_db)} file_job_ids")
                    # Spróbuj odwzorować file_job_id -> processed_files z DB
                    if file_job_ids_from_db and file_ids_from_db and len(file_job_ids_from_db) == len(file_ids_from_db):
                        from pathlib import Path as _Path
                        for fjid, fid in zip(file_job_ids_from_db, file_ids_from_db):
                            file_rec = agent_db.get_file_record_by_id(fid)
                            if not file_rec:
                                logger.warning(f"[{batch_job_id}] Brak rekordu processed_files dla file_id={fid}")
                                continue
                            job_temp_storage[fjid] = {
                                'type': 'file',
                                'batch_job_id': batch_job_id,
                                'status': file_rec.get('status', 'uploaded'),
                                'db_file_id': fid,
                                'file_path': _Path(file_rec['filepath']) if file_rec.get('filepath') else None,
                                'filename': file_rec.get('filename'),
                                'params': params_from_db or {},
                            }
                        logger.info(f"[{batch_job_id}] Odtworzono {len(file_job_ids_from_db)} wpisów file_job w job_temp_storage")
                    else:
                        if not file_job_ids_from_db:
                            logger.warning(f"[{batch_job_id}] Brak file_job_ids w params_json – nie można odwzorować plików")
                        if not file_ids_from_db:
                            logger.warning(f"[{batch_job_id}] Brak file_ids w batch_job_files – nie można odwzorować plików")
                        if file_job_ids_from_db and file_ids_from_db and len(file_job_ids_from_db) != len(file_ids_from_db):
                            logger.warning(f"[{batch_job_id}] Niezgodna długość file_job_ids ({len(file_job_ids_from_db)}) vs file_ids ({len(file_ids_from_db)}) – pomijam rekonstrukcję plików")
                else:
                    logger.error(f"[{batch_job_id}] Nie znaleziono batcha w DB przy próbie rekonstrukcji job_temp_storage")

            # Import tutaj, żeby uniknąć problemów z cyklicznymi importami
            from api_app import run_batch_processing
            
            # Uruchom przetwarzanie batcha
            await run_batch_processing(batch_job_id)
            return {"status": "completed", "batch_job_id": batch_job_id}
        
        # Dramatiq worker działa w osobnym wątku/procesie, więc tworzymy nowy event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Uruchom async funkcję
        result = loop.run_until_complete(process_batch_async())
        
        return result
    except Exception as e:
        error_message = str(e)
        logger.error(f"Błąd podczas przetwarzania batcha {batch_job_id}: {error_message}", exc_info=True)
        return {"error": error_message, "batch_job_id": batch_job_id}


def _process_single_file_impl(file_job_id: str, file_path: str, filename: str, username: str, transcription_provider: str, transcription_model: Optional[str], output_name: Optional[str], provider_tokens_json: Optional[str]):
    """Aktor Dramatiq do przetwarzania pojedynczych plików."""
    import asyncio
    import json
    from pathlib import Path
    
    logger.info(f"Przetwarzanie pojedynczego pliku: {file_job_id} ({filename})")
    
    try:
        # Import funkcji potrzebnych do przetwarzania
        import sys
        
        # Dodaj ścieżkę do api_app.py jeśli nie jest już dodana
        api_app_path = Path(__file__).parent
        if str(api_app_path) not in sys.path:
            sys.path.insert(0, str(api_app_path))
        
        # Import funkcji z api_app.py
        async def process_file_async():
            # Import tutaj, żeby uniknąć problemów z cyklicznymi importami
            from api_app import process_single_file_internal
            
            # Parsuj provider_tokens
            provider_tokens = {}
            if provider_tokens_json:
                try:
                    import json as json_lib
                    provider_tokens = json_lib.loads(provider_tokens_json)
                except:
                    pass
            
            # Uruchom przetwarzanie pliku
            result = await process_single_file_internal(
                file_job_id=file_job_id,
                file_path=Path(file_path),
                filename=filename,
                username=username,
                transcription_provider=transcription_provider,
                transcription_model=transcription_model,
                output_name=output_name,
                provider_tokens=provider_tokens,
            )
            return result
        
        # Dramatiq worker działa w osobnym wątku/procesie, więc tworzymy nowy event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Uruchom async funkcję
        result = loop.run_until_complete(process_file_async())
        
        return result
    except Exception as e:
        error_message = str(e)
        logger.error(f"Błąd podczas przetwarzania pliku {file_job_id}: {error_message}", exc_info=True)
        return {"error": error_message, "file_job_id": file_job_id}


# WAŻNE: Aktory MUSZĄ być zdefiniowane na poziomie modułu, żeby Dramatiq worker mógł je wykryć!
# Nie mogą być wewnątrz bloków if/try - muszą być na poziomie modułu

# Middleware do logowania wiadomości
class LoggingMiddleware:
    """Middleware do logowania wiadomości Dramatiq."""
    def __init__(self):
        self.actor_name = "LoggingMiddleware"
        # Dramatiq oczekuje atrybutu forks nawet jeśli nic nie forkujemy
        self.forks = ()
        # Dramatiq oczekuje actor_options jako set opcji aktora
        self.actor_options = set()

    # Hooki życiowe wymagane przez Dramatiq 1.18
    def before_worker_boot(self, broker, worker=None, **kwargs):
        logger.info("🔧 [MIDDLEWARE] Worker boot start")

    def after_worker_boot(self, broker, worker=None, **kwargs):
        logger.info("🔧 [MIDDLEWARE] Worker boot done")

    def before_worker_thread_boot(self, broker, thread=None, **kwargs):
        logger.info("🔧 [MIDDLEWARE] Thread boot start")

    def after_worker_thread_boot(self, broker, thread=None, **kwargs):
        logger.info("🔧 [MIDDLEWARE] Thread boot done")

    # Dodatkowe hooki lifecycle/no-op aby nie generować wyjątków
    def after_consumer_thread_boot(self, broker, *args, **kwargs):
        return None

    def before_worker_shutdown(self, broker, worker=None, **kwargs):
        return None

    def after_worker_shutdown(self, broker, worker=None, **kwargs):
        return None

    def before_worker_thread_shutdown(self, broker, thread=None, **kwargs):
        return None

    def before_consumer_thread_shutdown(self, broker, *args, **kwargs):
        return None

    def after_consumer_thread_shutdown(self, broker, *args, **kwargs):
        return None

    def before_ack(self, broker, message=None, *args, **kwargs):
        return None

    def after_ack(self, broker, message=None, *args, **kwargs):
        return None

    # Fallback: ignoruj wszelkie inne sygnały middleware, których nie obsługujemy
    def __getattr__(self, name):
        if name.startswith(("before_", "after_")):
            return lambda *a, **k: None
        raise AttributeError(name)

    # Hooki deklaracji kolejek/aktorów wywoływane przy starcie brokera
    def before_declare_actor(self, broker, actor):
        logger.info(f"🔧 [MIDDLEWARE] Deklaracja aktora: {getattr(actor, 'actor_name', actor)}")

    def after_declare_actor(self, broker, actor):
        logger.info(f"🔧 [MIDDLEWARE] Aktor zadeklarowany: {getattr(actor, 'actor_name', actor)}")

    def before_declare_queue(self, broker, queue_name):
        logger.info(f"🔧 [MIDDLEWARE] Deklaracja kolejki: {queue_name}")

    def after_declare_queue(self, broker, queue_name):
        logger.info(f"🔧 [MIDDLEWARE] Kolejka zadeklarowana: {queue_name}")

    def after_declare_delay_queue(self, broker, queue_name):
        logger.info(f"🔧 [MIDDLEWARE] Kolejka opóźnień zadeklarowana: {queue_name}")

    # Hooki enqueue (wysyłanie wiadomości do kolejki)
    def before_enqueue(self, broker, message=None, delay=None, *args, **kwargs):
        try:
            logger.info(f"📤 [MIDDLEWARE] enqueue message_id={getattr(message,'message_id',None)} actor={getattr(message,'actor_name',None)} queue={getattr(message,'queue_name',None)} delay={delay}")
        except Exception:
            logger.info("📤 [MIDDLEWARE] enqueue (no details)")

    def after_enqueue(self, broker, message=None, delay=None, *args, **kwargs):
        try:
            logger.info(f"✅ [MIDDLEWARE] enqueued message_id={getattr(message,'message_id',None)} actor={getattr(message,'actor_name',None)} queue={getattr(message,'queue_name',None)} delay={delay}")
        except Exception:
            logger.info("✅ [MIDDLEWARE] enqueued (no details)")

    def after_process_boot(self, broker, *, process=None):
        """Hook wywoływany po starcie procesu workera."""
        logger.info("🔧 [MIDDLEWARE] Proces workera wystartował")
    
    def before_process_message(self, broker, message):
        """Wywoływane przed przetworzeniem wiadomości."""
        logger.info(f"📥 [MIDDLEWARE] Otrzymano wiadomość: actor={message.actor_name}, message_id={message.message_id}, queue={message.queue_name}")
        return message
    
    def after_process_message(self, broker, message, *, result=None, exception=None):
        """Wywoływane po przetworzeniu wiadomości."""
        if exception:
            logger.error(f"❌ [MIDDLEWARE] Błąd podczas przetwarzania wiadomości {message.message_id}: {exception}")
        else:
            logger.info(f"✅ [MIDDLEWARE] Wiadomość {message.message_id} przetworzona pomyślnie")
        return message

# Najpierw upewnijmy się, że broker jest skonfigurowany
def _setup_broker():
    """Konfiguruje broker Redis z Results middleware."""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if dramatiq.get_broker() is None:
            broker = RedisBroker(url=redis_url)
            result_backend = RedisBackend(url=redis_url)
            broker.add_middleware(Results(backend=result_backend))
            broker.add_middleware(LoggingMiddleware())
            dramatiq.set_broker(broker)
            logger.info(f"✅ Domyślny broker Redis zainicjalizowany z Results middleware: {redis_url}")
        else:
            existing_broker = dramatiq.get_broker()
            middleware_list = getattr(existing_broker, 'middleware', [])
            has_results_middleware = any(isinstance(m, Results) for m in middleware_list)
            has_logging_middleware = any(isinstance(m, LoggingMiddleware) for m in middleware_list)
            if not has_results_middleware:
                result_backend = RedisBackend(url=redis_url)
                existing_broker.add_middleware(Results(backend=result_backend))
                logger.info("✅ Results middleware dodany do istniejącego brokera")
            if not has_logging_middleware:
                existing_broker.add_middleware(LoggingMiddleware())
                logger.info("✅ Logging middleware dodany do brokera")
    except Exception as e:
        logger.warning(f"Nie można ustawić domyślnego brokera: {e}. Broker będzie ustawiony później w get_broker()")

# Skonfiguruj broker przed dekorowaniem aktorów
_setup_broker()

# WAŻNE: Logowanie przy starcie modułu - to będzie widoczne w logach workera
logger.info("=" * 80)
logger.info("🚀 queue_service.py: Inicjalizacja aktorów Dramatiq")
logger.info("=" * 80)

# KLUCZOWE: Aktory MUSZĄ być zdefiniowane BEZPOŚREDNIO na poziomie modułu z dekoratorem @dramatiq.actor!
# Dramatiq automatycznie skanuje moduł i znajduje wszystkie funkcje zadekorowane @dramatiq.actor
# Dekorator @dramatiq.actor MUSI być zastosowany bezpośrednio używając składni @, BEZ warunków!

# WAŻNE: W środowisku produkcyjnym dramatiq powinien być zawsze dostępny
# Jeśli dramatiq jest None, to dekorator @dramatiq.actor nie zadziała, ale funkcje będą zdefiniowane

# Definiuj aktory BEZPOŚREDNIO na poziomie modułu z dekoratorem @dramatiq.actor
# Używamy bezpośrednio @dramatiq.actor - to jest JEDYNY sposób, który działa poprawnie!
# Jeśli dramatiq jest None, dekorator po prostu nie zadziała, ale funkcje będą zdefiniowane

# Użyj dekoratora @dramatiq.actor BEZPOŚREDNIO na funkcjach - to jest KLUCZOWE!
# Dramatiq automatycznie wykryje te aktory podczas skanowania modułu
# Aktory MUSZĄ być zdefiniowane BEZPOŚREDNIO na poziomie modułu, BEZ żadnych warunków!

@dramatiq.actor(max_retries=3, min_backoff=60000, max_backoff=240000, queue_name="chat", store_results=True)
def process_chat_message(username: str, query: str, collection_name: Optional[str] = None):
    return _process_chat_message_impl(username, query, collection_name)

@dramatiq.actor(max_retries=3, min_backoff=60000, max_backoff=240000, queue_name="batch_processing", store_results=True)
def process_batch(batch_job_id: str):
    # WAŻNE: Ten log powinien się pojawić gdy worker przetwarza zadanie
    # Jeśli tego logu nie ma, to znaczy że worker nie wywołuje aktora
    import sys
    print(f"[DRAMATIQ WORKER] ========== ROZPOCZECIE PRZETWARZANIA BATCH ==========", file=sys.stderr)
    print(f"[DRAMATIQ WORKER] batch_job_id: {batch_job_id}", file=sys.stderr)
    print(f"[DRAMATIQ WORKER] PID: {os.getpid()}", file=sys.stderr)
    logger.info("=" * 80)
    logger.info(f"🎯 [ACTOR process_batch] ========== ROZPOCZECIE PRZETWARZANIA BATCH ==========")
    logger.info(f"🎯 [ACTOR process_batch] Funkcja aktora wywołana dla batch_job_id: {batch_job_id}")
    logger.info(f"🎯 [ACTOR process_batch] PID procesu: {os.getpid()}")
    logger.info(f"🎯 [ACTOR process_batch] Nazwa aktora: process_batch")
    logger.info(f"🎯 [ACTOR process_batch] Kolejka: batch_processing")
    logger.info("=" * 80)
    try:
        result = _process_batch_impl(batch_job_id)
        logger.info("=" * 80)
        logger.info(f"🎯 [ACTOR process_batch] Funkcja aktora zakończona dla batch_job_id: {batch_job_id}")
        logger.info("=" * 80)
        print(f"[DRAMATIQ WORKER] ========== ZAKONCZENIE PRZETWARZANIA BATCH ==========", file=sys.stderr)
        return result
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"🎯 [ACTOR process_batch] Błąd w funkcji aktora dla batch_job_id {batch_job_id}: {e}", exc_info=True)
        logger.error("=" * 80)
        print(f"[DRAMATIQ WORKER] ========== BLAD W PRZETWARZANIU BATCH ==========", file=sys.stderr)
        print(f"[DRAMATIQ WORKER] Błąd: {e}", file=sys.stderr)
        raise

@dramatiq.actor(max_retries=3, min_backoff=60000, max_backoff=240000, queue_name="file_processing", store_results=True)
def process_single_file(file_job_id: str, file_path: str, filename: str, username: str, transcription_provider: str, transcription_model: Optional[str], output_name: Optional[str], provider_tokens_json: Optional[str]):
    return _process_single_file_impl(file_job_id, file_path, filename, username, transcription_provider, transcription_model, output_name, provider_tokens_json)

logger.info("✅ Aktory zadekorowane BEZPOŚREDNIO używając @dramatiq.actor na poziomie modułu")

# Logowanie informacji o aktorach
logger.info("=" * 80)
logger.info("✅ Aktory zdefiniowane na poziomie modułu")
logger.info(f"✅ Actor process_batch: {process_batch}")
logger.info(f"   - Queue: {getattr(process_batch, 'queue_name', 'default')}")
logger.info(f"   - Actor name: {getattr(process_batch, 'actor_name', 'unknown')}")
logger.info(f"   - Type: {type(process_batch)}")
logger.info(f"✅ Actor process_single_file: {process_single_file}")
logger.info(f"   - Queue: {getattr(process_single_file, 'queue_name', 'default')}")
logger.info(f"   - Actor name: {getattr(process_single_file, 'actor_name', 'unknown')}")
logger.info(f"   - Type: {type(process_single_file)}")
logger.info(f"✅ Actor process_chat_message: {process_chat_message}")
logger.info(f"   - Queue: {getattr(process_chat_message, 'queue_name', 'default')}")
logger.info(f"   - Actor name: {getattr(process_chat_message, 'actor_name', 'unknown')}")
logger.info(f"   - Type: {type(process_chat_message)}")

# Sprawdź czy aktory są zarejestrowane w brokerze
try:
    broker = dramatiq.get_broker()
    if broker:
        logger.info("=" * 80)
        logger.info("📋 Sprawdzam zarejestrowane aktory w brokerze...")
        if hasattr(broker, 'actors'):
            # broker.actors może zwracać różne typy - sprawdźmy to
            actors_list = list(broker.actors)
            logger.info(f"📋 broker.actors zwrócił {len(actors_list)} elementów typu: {[type(a).__name__ for a in actors_list[:3]]}")
            
            # Spróbuj wyciągnąć nazwy aktorów
            registered_actors = []
            for actor in actors_list:
                if hasattr(actor, 'actor_name'):
                    registered_actors.append(actor.actor_name)
                elif isinstance(actor, str):
                    registered_actors.append(actor)
                else:
                    registered_actors.append(str(actor))
            
            logger.info(f"✅ Zarejestrowane aktory w brokerze: {registered_actors}")
            if len(registered_actors) == 0:
                logger.error("❌ BRAK ZAREJESTROWANYCH AKTORÓW W BROKERZE!")
                logger.error("   To oznacza, że Dramatiq nie wykrył aktorów!")
            else:
                logger.info(f"✅ Znaleziono {len(registered_actors)} aktorów w brokerze")
        else:
            logger.warning("⚠️  Broker nie ma atrybutu 'actors'")
        logger.info("=" * 80)
except Exception as broker_check_error:
    logger.error(f"❌ Błąd podczas sprawdzania aktorów w brokerze: {broker_check_error}", exc_info=True)
logger.info("=" * 80)


def get_broker() -> "RedisBroker":
    """Zwraca broker Redis dla Dramatiq."""
    global _broker
    
    if _broker is not None:
        return _broker
    
    # dramatiq jest teraz zawsze dostępny (import bezpośredni)
    
    # Pobierz URL Redis z zmiennych środowiskowych lub użyj domyślnego
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    try:
        _broker = RedisBroker(url=redis_url)
        # Dodaj backend dla wyników
        global _result_backend
        _result_backend = RedisBackend(url=redis_url)
        _broker.add_middleware(Results(backend=_result_backend))
        # Dodaj logging middleware
        _broker.add_middleware(LoggingMiddleware())
        dramatiq.set_broker(_broker)
        logger.info(f"✅ Broker Redis zainicjalizowany: {redis_url}")
        return _broker
    except Exception as e:
        raise RuntimeError(f"Nie można połączyć się z Redis: {e}. Upewnij się, że Redis jest uruchomiony.")


async def start_boss() -> None:
    """Inicjalizuje Dramatiq (tworzy broker)."""
    get_broker()
    logger.info("✅ Dramatiq zainicjalizowany")


async def stop_boss() -> None:
    """Zatrzymuje Dramatiq."""
    global _broker
    if _broker:
        _broker.close()
        _broker = None
    logger.info("✅ Dramatiq zatrzymany")


async def enqueue_message(
    username: str,
    query: str,
    collection_name: Optional[str] = None,
    priority: int = 0,
) -> str:
    """
    Dodaje wiadomość do kolejki Dramatiq.
    
    Args:
        username: Nazwa użytkownika
        query: Treść zapytania
        collection_name: Opcjonalna nazwa kolekcji Qdrant
        priority: Priorytet wiadomości (wyższy = pierwszy) - nie używane w Dramatiq
        
    Returns:
        str: ID zadania (message ID)
    """
    if _actor is None:
        raise RuntimeError("Handler wiadomości nie jest zarejestrowany")
    
    # Upewnij się, że broker jest ustawiony
    get_broker()
    
    # Wyślij wiadomość do aktora
    message = _actor.send(
        username=username,
        query=query,
        collection_name=collection_name,
    )
    
    job_id = message.message_id
    logger.info(f"Dodano wiadomość do kolejki Dramatiq: {job_id} (użytkownik: {username})")
    return job_id


async def enqueue_batch_processing(batch_job_id: str) -> str:
    """
    Dodaje batch do kolejki Dramatiq do przetwarzania.
    
    Args:
        batch_job_id: ID batcha do przetworzenia
        
    Returns:
        str: ID zadania (message ID)
    """
    logger.info(f"[{batch_job_id}] enqueue_batch_processing wywołane")
    
    if _batch_actor is None:
        logger.error(f"[{batch_job_id}] ❌ _batch_actor jest None - handler batchów nie jest zarejestrowany!")
        logger.error(f"[{batch_job_id}] Wywołaj register_batch_handler() przed użyciem enqueue_batch_processing")
        raise RuntimeError("Handler batchów nie jest zarejestrowany")
    
    logger.info(f"[{batch_job_id}] _batch_actor jest dostępny: {_batch_actor}, funkcja: {_batch_actor.fn.__name__}")
    
    # Upewnij się, że broker jest ustawiony
    broker = get_broker()
    logger.info(f"[{batch_job_id}] Broker: {broker}")
    
    # Wyślij batch do aktora
    try:
        logger.info(f"[{batch_job_id}] ========== WYSYŁANIE BATCH DO AKTORA ==========")
        logger.info(f"[{batch_job_id}] Actor: {_batch_actor}")
        logger.info(f"[{batch_job_id}] Actor name: {getattr(_batch_actor, 'actor_name', 'unknown')}")
        logger.info(f"[{batch_job_id}] Actor queue: {getattr(_batch_actor, 'queue_name', 'unknown')}")
        logger.info(f"[{batch_job_id}] Broker: {broker}")
        logger.info(f"[{batch_job_id}] Wysyłam batch_job_id do aktora _batch_actor.send()...")
        
        # Sprawdź stan kolejki PRZED wysłaniem
        # Dramatiq używa formatu: dramatiq:{queue_name}.msgs
        try:
            import redis as redis_lib
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            r = redis_lib.from_url(redis_url)
            # Sprawdź różne formaty kluczy
            queue_keys_to_check = [
                f"dramatiq:batch_processing.msgs",  # Format używany przez Dramatiq
                f"dramatiq:queue:batch_processing",  # Standardowy format
                f"dramatiq:batch_processing",        # Alternatywny format
            ]
            queue_length_before = 0
            for queue_key in queue_keys_to_check:
                try:
                    length = r.llen(queue_key)
                    if length > 0:
                        queue_length_before = length
                        logger.info(f"[{batch_job_id}] 📊 Długość kolejki '{queue_key}' PRZED wysłaniem: {queue_length_before} zadań")
                        break
                except:
                    pass
            if queue_length_before == 0:
                logger.info(f"[{batch_job_id}] 📊 Kolejka 'batch_processing' PRZED wysłaniem: 0 zadań")
        except Exception as e:
            logger.warning(f"[{batch_job_id}] Nie można sprawdzić kolejki przed wysłaniem: {e}")
        
        message = _batch_actor.send(batch_job_id=batch_job_id)
        job_id = message.message_id
        logger.info(f"[{batch_job_id}] ✅ Wiadomość wysłana! Message ID: {job_id}")
        logger.info(f"[{batch_job_id}] ✅ Dodano batch do kolejki Dramatiq: {job_id} (batch_job_id: {batch_job_id})")
        logger.info(f"[{batch_job_id}] ✅ Batch dodany do kolejki 'batch_processing' przez aktora {_batch_actor.fn.__name__}")
        
        # Sprawdź czy zadanie jest w kolejce Redis PO wysłaniu
        # WAŻNE: Jeśli worker jest aktywny, zadanie może być od razu pobrane z kolejki, więc kolejka będzie pusta
        # To jest normalne zachowanie - zadanie jest wtedy w procesie przetwarzania przez worker
        try:
            import redis as redis_lib
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            r = redis_lib.from_url(redis_url)
            queue_key = f"dramatiq:queue:batch_processing"
            queue_length_after = r.llen(queue_key)
            logger.info(f"[{batch_job_id}] 📊 Długość kolejki 'batch_processing' PO wysłaniu: {queue_length_after} zadań")
            
            # Sprawdź też inne klucze Dramatiq, które mogą zawierać informacje o zadaniu
            delayed_key = f"dramatiq:delayed:batch_processing"
            delayed_count = r.zcard(delayed_key)
            if delayed_count > 0:
                logger.info(f"[{batch_job_id}] 📊 Opóźnione zadania w 'batch_processing': {delayed_count}")
            
            # Sprawdź czy zadanie jest w przetwarzaniu (w in-progress)
            in_progress_keys = r.keys(f"dramatiq:in-progress:*")
            if in_progress_keys:
                logger.info(f"[{batch_job_id}] 📊 Zadania w trakcie przetwarzania: {len(in_progress_keys)}")
            
            if queue_length == 0 and delayed_count == 0:
                logger.info(f"[{batch_job_id}] ℹ️  Kolejka jest pusta - zadanie zostało pobrane przez worker lub jest w trakcie przetwarzania")
                logger.info(f"[{batch_job_id}] ℹ️  Sprawdź logi workera Dramatiq - powinien pokazać '🚀 [DRAMATIQ WORKER] Przetwarzanie batcha: {batch_job_id}'")
        except Exception as redis_check_error:
            logger.debug(f"[{batch_job_id}] Nie można sprawdzić długości kolejki w Redis: {redis_check_error}")
        
        return job_id
    except Exception as e:
        logger.error(f"[{batch_job_id}] ❌ Błąd podczas dodawania batcha do kolejki Dramatiq: {e}", exc_info=True)
        raise


async def enqueue_single_file_processing(
    file_job_id: str,
    file_path: str,
    filename: str,
    username: str,
    transcription_provider: str,
    transcription_model: Optional[str] = None,
    output_name: Optional[str] = None,
    provider_tokens_json: Optional[str] = None,
) -> str:
    """
    Dodaje pojedynczy plik do kolejki Dramatiq do przetwarzania.
    
    Args:
        file_job_id: ID zadania pliku
        file_path: Ścieżka do pliku
        filename: Nazwa pliku
        username: Nazwa użytkownika
        transcription_provider: Provider transkrypcji
        transcription_model: Model transkrypcji (opcjonalny)
        output_name: Nazwa wyjściowa (opcjonalna)
        provider_tokens_json: JSON z tokenami providerów (opcjonalny)
        
    Returns:
        str: ID zadania (message ID)
    """
    if _file_actor is None:
        raise RuntimeError("Handler plików nie jest zarejestrowany")
    
    # Upewnij się, że broker jest ustawiony
    get_broker()
    
    # Wyślij plik do aktora
    message = _file_actor.send(
        file_job_id=file_job_id,
        file_path=file_path,
        filename=filename,
        username=username,
        transcription_provider=transcription_provider,
        transcription_model=transcription_model,
        output_name=output_name,
        provider_tokens_json=provider_tokens_json or "",
    )
    
    job_id = message.message_id
    logger.info(f"Dodano plik do kolejki Dramatiq: {job_id} (file_job_id: {file_job_id}, filename: {filename})")
    return job_id


async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Pobiera status zadania z Dramatiq.
    
    Args:
        job_id: ID zadania (message ID)
        
    Returns:
        Dict z danymi zadania lub None jeśli nie znaleziono
    """
    if _result_backend is None:
        return None
    
    try:
        # Pobierz wynik z Redis bezpośrednio
        # Dramatiq przechowuje wyniki w Redis z kluczem w formacie zależnym od wersji
        # Używamy Redis bezpośrednio, ponieważ get_result() wymaga obiektu Message
        import redis as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis_lib.from_url(redis_url)
        
        # Dramatiq przechowuje wyniki w Redis z kluczem w formacie zależnym od wersji
        # Sprawdzamy różne możliwe formaty kluczy
        result = None
        # Dramatiq używa klucza w formacie: dramatiq:result:{queue_name}:{message_id}
        # Dla domyślnej kolejki queue_name to "default"
        possible_keys = [
            f"dramatiq:result:default:{job_id}",
            f"dramatiq:results:default:{job_id}",
            f"dramatiq:result:{job_id}",
            f"dramatiq:results:{job_id}",
        ]
        
        for key_pattern in possible_keys:
            result_data = r.get(key_pattern)
            if result_data:
                try:
                    import pickle
                    result = pickle.loads(result_data)
                    break
                except Exception as e:
                    logger.debug(f"Błąd podczas deserializacji wyniku z klucza {key_pattern}: {e}")
                    # Spróbuj jako JSON
                    try:
                        import json
                        result = json.loads(result_data.decode('utf-8'))
                        break
                    except:
                        pass
        
        if result is None:
            # Jeśli nie ma wyniku, zadanie jest jeszcze przetwarzane lub nie istnieje
            return {
                "job_id": job_id,
                "status": "pending",  # Jeśli nie ma wyniku, prawdopodobnie jeszcze przetwarzane
                "response": None,
                "error_message": None,
                "retry_count": 0,
                "max_retries": 3,
            }
        
        # Sprawdź czy jest błąd
        if isinstance(result, Exception):
            return {
                "job_id": job_id,
                "status": "failed",
                "response": None,
                "error_message": str(result),
                "retry_count": 0,
                "max_retries": 3,
            }
        
        # Sukces
        if isinstance(result, dict):
            return {
                "job_id": job_id,
                "status": "completed",
                "response": result.get("response"),
                "error_message": result.get("error"),
                "retry_count": 0,
                "max_retries": 3,
            }
        
        return {
            "job_id": job_id,
            "status": "completed",
            "response": str(result) if result else None,
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3,
        }
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statusu zadania {job_id}: {e}", exc_info=True)
        return None


def register_message_handler(process_callback: Callable) -> None:
    """
    Rejestruje handler do przetwarzania wiadomości.
    
    Args:
        process_callback: Funkcja async do przetwarzania wiadomości.
                          Powinna przyjmować (query: str, collection_name: Optional[str], username: str)
                          i zwracać str (odpowiedź) lub rzucać wyjątek.
    """
    global _process_callback, _actor, _batch_actor, _file_actor
    
    # Upewnij się, że broker jest ustawiony
    get_broker()
    
    _process_callback = process_callback
    # Aktory są już zdefiniowane i udekorowane na poziomie modułu
    _actor = process_chat_message
    _batch_actor = process_batch
    _file_actor = process_single_file
    
    logger.info("✅ Handler wiadomości zarejestrowany w Dramatiq")
    logger.info("✅ Handler batchów zarejestrowany w Dramatiq")
    logger.info("✅ Handler pojedynczych plików zarejestrowany w Dramatiq")


async def get_queue_stats() -> Dict[str, Any]:
    """
    Pobiera statystyki kolejki Dramatiq z Redis.
    
    Returns:
        Dict ze statystykami kolejki:
        - pending: liczba zadań oczekujących
        - completed: liczba zadań zakończonych
        - failed: liczba zadań zakończonych błędem
        - total: całkowita liczba zadań
        - queue_size: rozmiar kolejki
    """
    # dramatiq jest teraz zawsze dostępny (import bezpośredni)
    
    try:
        import redis as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis_lib.from_url(redis_url)
        
        # Dramatiq przechowuje zadania w różnych kolekcjach Redis
        # Sprawdzamy klucze związane z Dramatiq
        pending_count = 0
        completed_count = 0
        failed_count = 0
        
        # Dramatiq używa różnych prefiksów dla różnych typów danych
        # Sprawdzamy klucze związane z wynikami
        result_keys = r.keys("dramatiq:result:*")
        result_keys.extend(r.keys("dramatiq:results:*"))
        
        for key in result_keys:
            try:
                result_data = r.get(key)
                if result_data:
                    try:
                        import pickle
                        result = pickle.loads(result_data)
                        if isinstance(result, Exception):
                            failed_count += 1
                        else:
                            completed_count += 1
                    except:
                        # Jeśli nie pickle, spróbuj JSON
                        try:
                            import json
                            result = json.loads(result_data.decode('utf-8'))
                            completed_count += 1
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Błąd podczas przetwarzania klucza {key}: {e}")
        
        # Sprawdzamy kolejkę zadań oczekujących
        # Dramatiq używa listy dla kolejki
        queue_keys = r.keys("dramatiq:queue:*")
        for queue_key in queue_keys:
            try:
                queue_length = r.llen(queue_key)
                pending_count += queue_length
            except:
                pass
        
        # Sprawdzamy również delayed messages
        delayed_keys = r.keys("dramatiq:delayed:*")
        for delayed_key in delayed_keys:
            try:
                delayed_count = r.zcard(delayed_key)
                pending_count += delayed_count
            except:
                pass
        
        total = pending_count + completed_count + failed_count
        
        return {
            "pending": pending_count,
            "completed": completed_count,
            "failed": failed_count,
            "total": total,
            "queue_size": pending_count,
            "redis_connected": True,
        }
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk kolejki: {e}", exc_info=True)
        return {
            "error": str(e),
            "pending": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
            "queue_size": 0,
            "redis_connected": False,
        }


async def list_active_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Pobiera listę aktywnych zadań z kolejki.
    
    Args:
        limit: Maksymalna liczba zadań do zwrócenia
        
    Returns:
        Lista słowników z informacjami o zadaniach
    """
    # dramatiq jest teraz zawsze dostępny (import bezpośredni)
    
    try:
        import redis as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis_lib.from_url(redis_url)
        
        jobs = []
        
        # Pobierz zadania z kolejki
        queue_keys = r.keys("dramatiq:queue:*")
        for queue_key in queue_keys:
            try:
                # Pobierz pierwsze N elementów z kolejki
                queue_items = r.lrange(queue_key, 0, limit - 1)
                for item in queue_items:
                    try:
                        import pickle
                        message = pickle.loads(item)
                        job_id = getattr(message, 'message_id', None)
                        if job_id:
                            jobs.append({
                                "job_id": job_id,
                                "status": "pending",
                                "queue": queue_key.decode('utf-8') if isinstance(queue_key, bytes) else queue_key,
                            })
                    except Exception as e:
                        logger.debug(f"Błąd podczas parsowania wiadomości z kolejki: {e}")
            except Exception as e:
                logger.debug(f"Błąd podczas pobierania z kolejki {queue_key}: {e}")
        
        # Pobierz opóźnione zadania
        delayed_keys = r.keys("dramatiq:delayed:*")
        for delayed_key in delayed_keys:
            try:
                delayed_items = r.zrange(delayed_key, 0, limit - 1, withscores=True)
                for item, score in delayed_items:
                    try:
                        import pickle
                        message = pickle.loads(item)
                        job_id = getattr(message, 'message_id', None)
                        if job_id:
                            jobs.append({
                                "job_id": job_id,
                                "status": "delayed",
                                "scheduled_at": datetime.fromtimestamp(score).isoformat() if score else None,
                                "queue": delayed_key.decode('utf-8') if isinstance(delayed_key, bytes) else delayed_key,
                            })
                    except Exception as e:
                        logger.debug(f"Błąd podczas parsowania opóźnionej wiadomości: {e}")
            except Exception as e:
                logger.debug(f"Błąd podczas pobierania opóźnionych zadań {delayed_key}: {e}")
        
        return jobs[:limit]
    except Exception as e:
        logger.error(f"Błąd podczas pobierania listy aktywnych zadań: {e}", exc_info=True)
        return []
