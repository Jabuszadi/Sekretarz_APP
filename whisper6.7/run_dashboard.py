#!/usr/bin/env python3
"""
Uruchamia Dramatiq Dashboard - oficjalne narzędzie do monitorowania kolejek.
Dashboard dostępny pod: http://localhost:8080
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import dramatiq
    from dramatiq.brokers.redis import RedisBroker
    from dramatiq_dashboard import DashboardApp
except ImportError as e:
    logger.error(f"Brakuje wymaganych pakietów: {e}")
    logger.error("Zainstaluj: pip install dramatiq_dashboard")
    exit(1)

try:
    # Użyj waitress jeśli dostępny (szybki i działa na Windows), w przeciwnym razie użyj wsgiref
    try:
        from waitress import serve
        USE_WAITRESS = True
    except ImportError:
        from wsgiref.simple_server import make_server
        USE_WAITRESS = False
        logger.info("waitress nie jest zainstalowany, używam wsgiref (wolniejszy)")
        logger.info("Dla lepszej wydajności zainstaluj: pip install waitress")
except ImportError:
    from wsgiref.simple_server import make_server
    USE_WAITRESS = False


def main():
    """Uruchamia dashboard Dramatiq."""
    # Pobierz URL Redis z zmiennych środowiskowych
    # WAŻNE: Dashboard używa TEGO SAMEGO Redis co Twój system!
    # Wszystkie komponenty (API, Worker, Dashboard) czytają z tego samego Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
    dashboard_host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    
    logger.info("=" * 60)
    logger.info("🚀 Uruchamianie Dramatiq Dashboard")
    logger.info("=" * 60)
    logger.info(f"🔧 Redis URL: {redis_url}")
    logger.info("   ⚠️  Dashboard używa TEGO SAMEGO Redis co Twój system!")
    logger.info("   📊 Dashboard tylko CZYTA z Redis (pokazuje statystyki)")
    logger.info("   ✅ Worker i API zapisują/czytają zadania z tego Redis")
    logger.info("=" * 60)
    logger.info(f"🌐 Dashboard będzie dostępny pod: http://{dashboard_host}:{dashboard_port}")
    
    try:
        # Skonfiguruj broker Redis
        broker = RedisBroker(url=redis_url)
        
        # Zadeklaruj wszystkie kolejki używane w systemie
        # Dashboard musi wiedzieć o wszystkich kolejkach, żeby je pokazać
        queues_to_declare = ["chat", "batch_processing", "file_processing", "default"]
        for queue_name in queues_to_declare:
            try:
                broker.declare_queue(queue_name)
                logger.info(f"✅ Zadeklarowano kolejkę: {queue_name}")
            except Exception as e:
                logger.debug(f"Kolejka {queue_name} może już istnieć: {e}")
        
        # Ustaw broker jako domyślny
        dramatiq.set_broker(broker)
        
        # Inicjalizuj aplikację Dashboard z opcjonalnymi parametrami optymalizacji
        # Dashboard automatycznie ogranicza ilość danych do wyświetlenia
        app = DashboardApp(broker=broker, prefix="")
        
        logger.info("✅ Dramatiq Dashboard uruchomiony!")
        logger.info(f"📊 Otwórz w przeglądarce: http://{dashboard_host}:{dashboard_port}")
        if not USE_WAITRESS:
            logger.info("💡 Wskazówka: Zainstaluj 'waitress' dla lepszej wydajności: pip install waitress")
        logger.info("Naciśnij CTRL+C, aby zakończyć")
        
        # Uruchom serwer
        if USE_WAITRESS:
            logger.info(f"Serwer Waitress uruchomiony na {dashboard_host}:{dashboard_port}")
            serve(app, host=dashboard_host, port=dashboard_port)
        else:
            with make_server(dashboard_host, dashboard_port, app) as httpd:
                logger.info(f"Serwer WSGI uruchomiony na {dashboard_host}:{dashboard_port}")
                httpd.serve_forever()
                
    except Exception as e:
        logger.error(f"❌ Błąd podczas uruchamiania dashboardu: {e}", exc_info=True)
        logger.error("Sprawdź czy Redis jest uruchomiony: redis-cli ping")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Zakończono dashboard.")
        exit(0)
