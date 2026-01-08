#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Skrypt do czyszczenia kolejki Dramatiq w Redis."""
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
    print("❌ redis nie jest zainstalowany. Zainstaluj: pip install redis")
    sys.exit(1)

def clear_dramatiq_queue():
    """Czyści wszystkie zadania Dramatiq z Redis."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    try:
        r = redis.from_url(redis_url)
        
        print("=" * 80)
        print("Czyszczenie kolejki Dramatiq w Redis")
        print("=" * 80)
        print(f"Redis URL: {redis_url}")
        
        # Sprawdź co jest w Redis przed czyszczeniem
        print("\n📊 Stan przed czyszczeniem:")
        dramatiq_keys = r.keys("dramatiq:*")
        print(f"   Znaleziono {len(dramatiq_keys)} kluczy Dramatiq")
        
        # Pokaż klucze
        if dramatiq_keys:
            print("\n   Klucze Dramatiq:")
            for key in sorted(dramatiq_keys)[:20]:  # Pokaż pierwsze 20
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                key_type = r.type(key).decode('utf-8') if isinstance(r.type(key), bytes) else r.type(key)
                if key_type == 'list':
                    length = r.llen(key)
                    print(f"     - {key_str} (list, {length} elementów)")
                elif key_type == 'zset':
                    length = r.zcard(key)
                    print(f"     - {key_str} (sorted set, {length} elementów)")
                elif key_type == 'string':
                    print(f"     - {key_str} (string)")
                else:
                    print(f"     - {key_str} ({key_type})")
            if len(dramatiq_keys) > 20:
                print(f"     ... i {len(dramatiq_keys) - 20} więcej")
        
        # Zapytaj użytkownika
        print("\n[WARN] UWAGA: To usunie WSZYSTKIE zadania Dramatiq z Redis!")
        print("   - Kolejki zadań (pending)")
        print("   - Opóźnione zadania (delayed)")
        print("   - Wyniki zadań (results)")
        print("   - Zadania w trakcie przetwarzania (in-progress)")
        
        response = input("\nCzy na pewno chcesz wyczyscic kolejke? (tak/nie): ")
        if response.lower() not in ['tak', 'yes', 'y', 't']:
            print("[OK] Anulowano czyszczenie.")
            return
        
        # Wyczyść wszystkie klucze Dramatiq
        print("\nCzyszczenie...")
        deleted_count = 0
        for key in dramatiq_keys:
            try:
                r.delete(key)
                deleted_count += 1
            except Exception as e:
                print(f"   [WARN] Blad podczas usuwania {key}: {e}")
        
        print(f"\n[OK] Usunieto {deleted_count} kluczy Dramatiq")
        
        # Sprawdź stan po czyszczeniu
        print("\n📊 Stan po czyszczeniu:")
        remaining_keys = r.keys("dramatiq:*")
        print(f"   Pozostało {len(remaining_keys)} kluczy Dramatiq")
        
        if remaining_keys:
            print("   [WARN] Niektore klucze nie zostaly usuniete:")
            for key in remaining_keys[:10]:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                print(f"     - {key_str}")
        
        print("\n" + "=" * 80)
        print("[OK] Czyszczenie zakonczone")
        print("=" * 80)
        
    except redis.ConnectionError:
        print(f"[ERROR] Nie mozna polaczyc sie z Redis: {redis_url}")
        print("   Upewnij sie, ze Redis jest uruchomiony.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Blad podczas czyszczenia kolejki: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    clear_dramatiq_queue()
