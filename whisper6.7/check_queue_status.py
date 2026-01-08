#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sprawdź stan kolejek Dramatiq w Redis."""
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
except ImportError:
    print("[ERROR] redis nie jest zainstalowany. Zainstaluj: pip install redis")
    sys.exit(1)

def check_queue_status():
    """Sprawdź stan kolejek Dramatiq w Redis."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    try:
        r = redis.from_url(redis_url)
        r.ping()
        
        print("=" * 80)
        print("Sprawdzanie stanu kolejek Dramatiq w Redis")
        print("=" * 80)
        print(f"Redis URL: {redis_url}\n")
        
        # Sprawdź kolejki - Dramatiq może używać różnych formatów kluczy
        queues = ["chat", "batch_processing", "file_processing", "default"]
        print("1. Kolejki zadań (pending):")
        print("-" * 80)
        total_pending = 0
        for queue_name in queues:
            # Sprawdź różne formaty kluczy, które Dramatiq może używać
            queue_keys_to_check = [
                f"dramatiq:queue:{queue_name}",     # Standardowy format
                f"dramatiq:{queue_name}",            # Alternatywny format
                f"dramatiq:{queue_name}.msgs",      # Format z .msgs
            ]
            
            queue_length = 0
            found_key = None
            for queue_key in queue_keys_to_check:
                try:
                    length = r.llen(queue_key)
                    if length > 0:
                        queue_length = length
                        found_key = queue_key
                        break
                except:
                    pass
            
            total_pending += queue_length
            if queue_length > 0:
                print(f"   [{queue_name}] {queue_length} zadań oczekujących (klucz: {found_key})")
                # Pokaż pierwsze zadanie
                first_item = r.lindex(found_key, 0)
                if first_item:
                    print(f"      Pierwsze zadanie (pierwsze 200 znaków):")
                    print(f"      {str(first_item)[:200]}...")
            else:
                print(f"   [{queue_name}] 0 zadań (pusta)")
        print(f"\n   RAZEM: {total_pending} zadań oczekujących\n")
        
        # Sprawdź opóźnione zadania
        print("2. Opóźnione zadania (delayed):")
        print("-" * 80)
        total_delayed = 0
        for queue_name in queues:
            delayed_key = f"dramatiq:delayed:{queue_name}"
            delayed_count = r.zcard(delayed_key)
            total_delayed += delayed_count
            if delayed_count > 0:
                print(f"   [{queue_name}] {delayed_count} zadań opóźnionych")
        print(f"\n   RAZEM: {total_delayed} zadań opóźnionych\n")
        
        # Sprawdź zadania w trakcie przetwarzania
        print("3. Zadania w trakcie przetwarzania (in-progress):")
        print("-" * 80)
        in_progress_keys = r.keys("dramatiq:in-progress:*")
        if in_progress_keys:
            print(f"   Znaleziono {len(in_progress_keys)} zadań w trakcie przetwarzania")
            for key in in_progress_keys[:5]:  # Pokaż pierwsze 5
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                print(f"   - {key_str}")
            if len(in_progress_keys) > 5:
                print(f"   ... i {len(in_progress_keys) - 5} więcej")
        else:
            print("   Brak zadań w trakcie przetwarzania")
        print()
        
        # Sprawdź wyniki
        print("4. Wyniki zadań (results):")
        print("-" * 80)
        result_keys = r.keys("dramatiq:result:*")
        if result_keys:
            print(f"   Znaleziono {len(result_keys)} wyników zadań")
        else:
            print("   Brak wyników zadań")
        print()
        
        # Podsumowanie
        print("=" * 80)
        print("PODSUMOWANIE:")
        print("=" * 80)
        if total_pending > 0:
            print(f"⚠️  W kolejce są {total_pending} zadania oczekujące na przetworzenie!")
            print("   Sprawdź czy worker jest uruchomiony: python run_worker.py")
        elif total_delayed > 0:
            print(f"⏰ Są {total_delayed} zadania opóźnione")
        elif in_progress_keys:
            print(f"🔄 Są {len(in_progress_keys)} zadania w trakcie przetwarzania")
            print("   To oznacza, że worker przetwarza zadania!")
        else:
            print("✅ Wszystkie kolejki są puste - brak zadań do przetworzenia")
        
        print("=" * 80)
        
    except redis.ConnectionError:
        print(f"[ERROR] Nie można połączyć się z Redis: {redis_url}")
        print("   Upewnij się, że Redis jest uruchomiony: redis-cli ping")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Błąd podczas sprawdzania kolejek: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_queue_status()
