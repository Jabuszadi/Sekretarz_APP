#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Skrypt diagnostyczny do sprawdzania stanu Dramatiq workera i aktorów."""
import os
import sys
from dotenv import load_dotenv

# Ustaw kodowanie dla Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

load_dotenv()

print("=" * 80)
print("Diagnostyka Dramatiq Workera")
print("=" * 80)

# 1. Sprawdź czy dramatiq jest zainstalowany
print("\n1️⃣  Sprawdzanie instalacji Dramatiq...")
try:
    import dramatiq
    from dramatiq.brokers.redis import RedisBroker
    print(f"   [OK] Dramatiq zainstalowany: {dramatiq.__version__}")
except ImportError as e:
    print(f"   [ERROR] Dramatiq nie jest zainstalowany: {e}")
    print("   Zainstaluj: pip install dramatiq[redis]")
    sys.exit(1)

# 2. Sprawdź połączenie z Redis
print("\n2️⃣  Sprawdzanie połączenia z Redis...")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    import redis
    r = redis.from_url(redis_url)
    r.ping()
    print(f"   [OK] Redis dziala: {redis_url}")
except Exception as e:
    print(f"   [ERROR] Nie mozna polaczyc sie z Redis: {e}")
    print("   Upewnij się, że Redis jest uruchomiony: redis-cli ping")
    sys.exit(1)

# 3. Sprawdź aktory w module queue_service
print("\n3️⃣  Sprawdzanie aktorów w queue_service...")
try:
    # Importuj queue_service - to powinno zarejestrować aktory
    import queue_service
    
    # Sprawdź czy aktory są zdefiniowane
    if hasattr(queue_service, 'process_batch'):
        actor = queue_service.process_batch
        print(f"   [OK] Actor process_batch znaleziony: {actor}")
        print(f"      - Type: {type(actor)}")
        print(f"      - Queue: {getattr(actor, 'queue_name', 'unknown')}")
        print(f"      - Actor name: {getattr(actor, 'actor_name', 'unknown')}")
    else:
        print("   [ERROR] Actor process_batch NIE zostal znaleziony w queue_service!")
    
    if hasattr(queue_service, 'process_chat_message'):
        print(f"   [OK] Actor process_chat_message znaleziony")
    else:
        print("   [ERROR] Actor process_chat_message NIE zostal znaleziony!")
    
    if hasattr(queue_service, 'process_single_file'):
        print(f"   [OK] Actor process_single_file znaleziony")
    else:
        print("   [ERROR] Actor process_single_file NIE zostal znaleziony!")
        
except Exception as e:
    print(f"   [ERROR] Blad podczas importowania queue_service: {e}")
    import traceback
    traceback.print_exc()

# 4. Sprawdź broker i zarejestrowane aktory
print("\n4️⃣  Sprawdzanie brokera i zarejestrowanych aktorów...")
try:
    broker = dramatiq.get_broker()
    if broker:
        print(f"   [OK] Broker znaleziony: {type(broker).__name__}")
        
        if hasattr(broker, 'actors'):
            actors_list = list(broker.actors)
            print(f"   Broker ma {len(actors_list)} zarejestrowanych aktorow")
            
            registered_actors = []
            for actor in actors_list:
                if hasattr(actor, 'actor_name'):
                    registered_actors.append(actor.actor_name)
                elif isinstance(actor, str):
                    registered_actors.append(actor)
                else:
                    registered_actors.append(str(actor))
            
            print(f"   Zarejestrowane aktory: {registered_actors}")
            
            if 'process_batch' in registered_actors or any('process_batch' in str(a) for a in registered_actors):
                print("   [OK] Actor 'process_batch' jest zarejestrowany w brokerze!")
            else:
                print("   [ERROR] Actor 'process_batch' NIE jest zarejestrowany w brokerze!")
                print("   [WARN] To oznacza, ze worker nie bedzie mogl przetwarzac zadan batch!")
        else:
            print("   [WARN] Broker nie ma atrybutu 'actors'")
    else:
        print("   [ERROR] Broker nie jest skonfigurowany!")
        print("   [WARN] Uruchom _setup_broker() w queue_service")
except Exception as e:
    print(f"   [ERROR] Blad podczas sprawdzania brokera: {e}")
    import traceback
    traceback.print_exc()

# 5. Sprawdź stan kolejek w Redis
print("\n5️⃣  Sprawdzanie stanu kolejek w Redis...")
try:
    # Sprawdź kolejki
    queue_keys = r.keys("dramatiq:queue:*")
    print(f"   Znaleziono {len(queue_keys)} kolejek w Redis:")
    
    for queue_key in queue_keys:
        queue_name = queue_key.decode('utf-8') if isinstance(queue_key, bytes) else queue_key
        queue_length = r.llen(queue_key)
        print(f"      - {queue_name}: {queue_length} zadan")
        
        # Jeśli są zadania, pokaż pierwsze
        if queue_length > 0:
            print(f"        [WARN] W kolejce sa zadania oczekujace na przetworzenie!")
            # Pokaż pierwsze zadanie (bez deserializacji)
            first_item = r.lindex(queue_key, 0)
            if first_item:
                print(f"        Pierwsze zadanie (pierwsze 100 znakow): {str(first_item)[:100]}...")
    
    # Sprawdź delayed messages
    delayed_keys = r.keys("dramatiq:delayed:*")
    if delayed_keys:
        print(f"   Znaleziono {len(delayed_keys)} kolejek opoznionych:")
        for delayed_key in delayed_keys:
            delayed_name = delayed_key.decode('utf-8') if isinstance(delayed_key, bytes) else delayed_key
            delayed_count = r.zcard(delayed_key)
            if delayed_count > 0:
                print(f"      - {delayed_name}: {delayed_count} zadan opoznionych")
    
    # Sprawdź wyniki
    result_keys = r.keys("dramatiq:result:*")
    if result_keys:
        print(f"   Znaleziono {len(result_keys)} wynikow zadan w Redis")
    
except Exception as e:
    print(f"   [ERROR] Blad podczas sprawdzania kolejek: {e}")
    import traceback
    traceback.print_exc()

# 6. Sprawdź czy są procesy workera
print("\n6️⃣  Sprawdź procesy workera...")
print("   💡 Uruchom w PowerShell: Get-Process python | Where-Object {$_.CommandLine -like '*dramatiq*'}")
print("   💡 Lub sprawdź ręcznie w Task Manager")

print("\n" + "=" * 80)
print("[OK] Diagnostyka zakonczona")
print("=" * 80)
print("\nWskazowki:")
print("   1. Jesli aktory nie sa zarejestrowane, sprawdz logi przy starcie workera")
print("   2. Jesli kolejka ma zadania, ale worker nie przetwarza, sprawdz czy worker jest uruchomiony")
print("   3. Uruchom worker w osobnym terminalu: python -m dramatiq queue_service --queues chat,batch_processing,file_processing")
print("   4. Sprawdz logi workera - powinien pokazac zarejestrowane aktory przy starcie")
