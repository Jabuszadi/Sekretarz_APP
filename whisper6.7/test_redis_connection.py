#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test połączenia Redis i sprawdzenie czy API i Worker używają tego samego Redis."""
import os
import sys
from dotenv import load_dotenv

# Ustaw kodowanie dla Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

load_dotenv()

try:
    import redis
    import dramatiq
    from dramatiq.brokers.redis import RedisBroker
except ImportError as e:
    print(f"[ERROR] Brakuje modułu: {e}")
    sys.exit(1)

def test_redis_connection():
    """Test połączenia Redis i konfiguracji."""
    print("=" * 80)
    print("Test połączenia Redis i konfiguracji Dramatiq")
    print("=" * 80)
    
    # 1. Sprawdź REDIS_URL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"\n1. REDIS_URL z .env: {redis_url}")
    
    # 2. Test połączenia Redis
    print("\n2. Test połączenia Redis:")
    try:
        r = redis.from_url(redis_url)
        r.ping()
        print(f"   [OK] Połączenie z Redis działa: {redis_url}")
        
        # Sprawdź informacje o Redis
        info = r.info()
        print(f"   [OK] Redis wersja: {info.get('redis_version', 'unknown')}")
        print(f"   [OK] Używana baza danych: {r.connection_pool.connection_kwargs.get('db', 0)}")
    except Exception as e:
        print(f"   [ERROR] Nie można połączyć się z Redis: {e}")
        return
    
    # 3. Sprawdź broker Dramatiq
    print("\n3. Sprawdź broker Dramatiq:")
    try:
        import queue_service
        broker = dramatiq.get_broker()
        if broker:
            print(f"   [OK] Broker znaleziony: {type(broker).__name__}")
            if hasattr(broker, 'connection_pool'):
                broker_url = getattr(broker.connection_pool, 'connection_kwargs', {})
                print(f"   [OK] Broker Redis URL: {broker_url}")
            else:
                print(f"   [WARN] Broker nie ma connection_pool - nie można sprawdzić URL")
        else:
            print(f"   [ERROR] Broker nie jest skonfigurowany!")
    except Exception as e:
        print(f"   [ERROR] Błąd podczas sprawdzania brokera: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Test wysłania wiadomości testowej
    print("\n4. Test wysłania wiadomości testowej:")
    try:
        import queue_service
        broker = dramatiq.get_broker()
        if broker and hasattr(queue_service, 'process_batch'):
            actor = queue_service.process_batch
            
            # Sprawdź stan kolejki PRZED - Dramatiq może używać różnych formatów kluczy
            queue_keys_to_check = [
                f"dramatiq:queue:batch_processing",  # Standardowy format
                f"dramatiq:batch_processing",        # Alternatywny format
                f"dramatiq:batch_processing.msgs",  # Format z .msgs
            ]
            
            queue_lengths_before = {}
            for queue_key in queue_keys_to_check:
                try:
                    length = r.llen(queue_key)
                    queue_lengths_before[queue_key] = length
                    if length > 0:
                        print(f"   {queue_key} PRZED: {length} zadań")
                except:
                    pass
            
            # Wyślij testową wiadomość
            print(f"   Wysyłam testową wiadomość...")
            test_batch_id = "test-connection-check"
            message = actor.send(batch_job_id=test_batch_id)
            print(f"   [OK] Wiadomość wysłana! Message ID: {message.message_id}")
            
            # Sprawdź stan kolejki PO
            import time
            time.sleep(0.5)  # Czekaj chwilę
            print(f"   Sprawdzam kolejki PO wysłaniu:")
            queue_lengths_after = {}
            for queue_key in queue_keys_to_check:
                try:
                    length = r.llen(queue_key)
                    queue_lengths_after[queue_key] = length
                    if length > 0:
                        print(f"   {queue_key} PO: {length} zadań")
                        # Pokaż pierwsze zadanie
                        first_item = r.lindex(queue_key, 0)
                        if first_item:
                            print(f"      Pierwsze zadanie (pierwsze 200 znaków):")
                            print(f"      {str(first_item)[:200]}...")
                except:
                    pass
            
            # Sprawdź czy wiadomość trafiła do kolejki
            found_queue = None
            for queue_key in queue_keys_to_check:
                before = queue_lengths_before.get(queue_key, 0)
                after = queue_lengths_after.get(queue_key, 0)
                if after > before:
                    found_queue = queue_key
                    print(f"   [OK] Wiadomość trafiła do kolejki: {queue_key} ({before} -> {after})")
                    break
            
            if not found_queue:
                # Sprawdź czy któraś kolejka ma zadania
                for queue_key in queue_keys_to_check:
                    after = queue_lengths_after.get(queue_key, 0)
                    if after > 0:
                        found_queue = queue_key
                        print(f"   [WARN] Kolejka {queue_key} ma {after} zadań, ale nie widać wzrostu")
                        break
                
                if not found_queue:
                    print(f"   [WARN] Wiadomość może nie trafić do kolejki lub została natychmiast pobrana")
            
            # Usuń testową wiadomość jeśli jest w kolejce (opcjonalnie)
            # Nie usuwamy, żeby zobaczyć czy worker ją przetworzy
        else:
            print(f"   [ERROR] Nie można znaleźć aktora process_batch")
    except Exception as e:
        print(f"   [ERROR] Błąd podczas testu wysyłania: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Sprawdź wszystkie klucze Dramatiq
    print("\n5. Wszystkie klucze Dramatiq w Redis:")
    try:
        all_keys = r.keys("dramatiq:*")
        print(f"   Znaleziono {len(all_keys)} kluczy Dramatiq")
        if len(all_keys) > 0:
            print(f"   Przykładowe klucze:")
            for key in sorted(all_keys)[:10]:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                print(f"     - {key_str}")
            if len(all_keys) > 10:
                print(f"     ... i {len(all_keys) - 10} więcej")
    except Exception as e:
        print(f"   [ERROR] Błąd podczas sprawdzania kluczy: {e}")
    
    print("\n" + "=" * 80)
    print("Test zakończony")
    print("=" * 80)

if __name__ == "__main__":
    test_redis_connection()
