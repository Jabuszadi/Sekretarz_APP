#!/usr/bin/env python
"""
Osobny skrypt do uruchamiania Dramatiq workera.
Uruchom to w osobnym terminalu, żeby widzieć logi i mieć pełną kontrolę nad workerem.
"""
import os
import sys
import logging
from pathlib import Path

# Ustaw kodowanie dla Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def main():
    """Uruchom Dramatiq worker."""
    logger.info("=" * 80)
    logger.info("🚀 Uruchamianie Dramatiq Workera")
    logger.info("=" * 80)
    
    # Sprawdź czy jesteśmy w odpowiednim katalogu
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    logger.info(f"📁 Katalog roboczy: {os.getcwd()}")
    
    # Sprawdź zmienne środowiskowe
    from dotenv import load_dotenv
    load_dotenv()
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    logger.info(f"🔗 Redis URL: {redis_url}")
    
    # Sprawdź czy dramatiq jest zainstalowany
    try:
        import dramatiq
        logger.info(f"✅ Dramatiq zainstalowany: {dramatiq.__version__}")
    except ImportError:
        logger.error("❌ Dramatiq nie jest zainstalowany!")
        logger.error("   Zainstaluj: pip install dramatiq[redis]")
        sys.exit(1)
    
    # Sprawdź połączenie z Redis
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        logger.info("✅ Połączenie z Redis działa")
    except Exception as e:
        logger.error(f"❌ Nie można połączyć się z Redis: {e}")
        logger.error("   Upewnij się, że Redis jest uruchomiony: redis-cli ping")
        sys.exit(1)
    
    # Importuj queue_service - to zarejestruje aktory
    logger.info("📦 Importowanie queue_service...")
    try:
        import queue_service
        logger.info("✅ queue_service zaimportowany")
        
        # Sprawdź aktory
        broker = dramatiq.get_broker()
        if broker and hasattr(broker, 'actors'):
            actors_list = list(broker.actors)
            logger.info(f"✅ Znaleziono {len(actors_list)} zarejestrowanych aktorów:")
            for actor in actors_list:
                if hasattr(actor, 'actor_name'):
                    logger.info(f"   - {actor.actor_name} (queue: {getattr(actor, 'queue_name', 'default')})")
                else:
                    logger.info(f"   - {actor}")
    except Exception as e:
        logger.error(f"❌ Błąd podczas importowania queue_service: {e}", exc_info=True)
        sys.exit(1)
    
    # Kolejki do nasłuchiwania (możesz nadpisać przez zmienną WORKER_QUEUES)
    queues = os.getenv("WORKER_QUEUES", "batch_processing")
    # Domyślnie ograniczamy do 1 procesu i 1 wątku (stabilniej na Windows)
    processes = os.getenv("WORKER_PROCESSES", "1")
    threads = os.getenv("WORKER_THREADS", "1")
    logger.info(f"📋 Worker będzie nasłuchiwał na kolejkach: {queues}")
    
    logger.info("=" * 80)
    logger.info("✅ Worker gotowy do uruchomienia")
    logger.info("=" * 80)
    logger.info("")
    logger.info("💡 Wskazówki:")
    logger.info("   - Worker będzie przetwarzał zadania z kolejek: chat, batch_processing, file_processing")
    logger.info("   - Logi z przetwarzania zadań będą widoczne w tym terminalu")
    logger.info("   - Naciśnij CTRL+C, aby zatrzymać workera")
    logger.info("")
    logger.info("🔄 Uruchamiam worker...")
    logger.info("=" * 80)
    logger.info("")
    
    # Uruchom worker używając Dramatiq CLI
    # To jest równoważne z: python -m dramatiq queue_service --queues chat,batch_processing,file_processing
    try:
        import dramatiq.cli
        from dramatiq.cli import main as dramatiq_main
        
        # Ustaw argumenty dla Dramatiq CLI (parametry można nadpisać zmiennymi środowiskowymi)
        sys.argv = [
            'dramatiq',
            'queue_service',
            '--queues', queues,
            '--processes', processes,
            '--threads', threads,
            '--verbose',
        ]
        logger.info(f"🔧 Parametry workera: {' '.join(sys.argv)}")
        logger.info(f"📋 Worker będzie nasłuchiwał TYLKO na kolejkach: {queues}")
        logger.info("")
        
        # Uruchom Dramatiq CLI
        dramatiq_main()
        
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("⏹️  Worker zatrzymany przez użytkownika (CTRL+C)")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ Błąd podczas uruchamiania workera: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
