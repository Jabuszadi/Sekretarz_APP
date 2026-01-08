# Instrukcja uruchamiania Dramatiq Workera

## Co to jest Dramatiq Worker?

Dramatiq Worker to osobny proces, który:
- Nasłuchuje na kolejkach Redis (chat, batch_processing, file_processing)
- Pobiera zadania z kolejek
- Przetwarza zadania używając zarejestrowanych aktorów
- Zapisuje wyniki z powrotem do Redis

**WAŻNE:** Worker NIE działa na porcie HTTP - to proces, który łączy się z Redis.

## Uruchomienie Workera

### Opcja 1: Użyj prostego skryptu (ZALECANE)

```powershell
# W osobnym terminalu
python run_worker.py
```

Skrypt automatycznie:
- Sprawdzi połączenie z Redis
- Zaimportuje i zarejestruje aktory
- Uruchomi worker z odpowiednimi parametrami
- Pokaże wszystkie logi z przetwarzania

### Opcja 2: Uruchom bezpośrednio

```powershell
# W osobnym terminalu
python -m dramatiq queue_service --queues chat,batch_processing,file_processing
```

## Struktura uruchomienia

```
Terminal 1: API + Daemon
  python run.py
  → Uruchamia API (port 7777)
  → Uruchamia Daemon

Terminal 2: Dramatiq Worker
  python run_worker.py
  → Nasłuchuje na kolejkach Redis
  → Przetwarza zadania
```

## Sprawdzanie czy Worker działa

### 1. Sprawdź logi workera

Worker powinien pokazać przy starcie:
```
✅ Znaleziono 3 zarejestrowanych aktorów:
   - process_chat_message (queue: chat)
   - process_batch (queue: batch_processing)
   - process_single_file (queue: file_processing)
```

### 2. Sprawdź procesy

```powershell
.\check_worker.ps1
```

### 3. Uruchom diagnostykę

```powershell
python diagnose_worker.py
```

## Logi z przetwarzania

Gdy worker przetwarza zadanie batch, zobaczysz:

```
[DRAMATIQ WORKER] ========== ROZPOCZECIE PRZETWARZANIA BATCH ==========
🎯 [ACTOR process_batch] Funkcja aktora wywołana dla batch_job_id: ...
🚀 [DRAMATIQ WORKER] Przetwarzanie batcha: ...
```

## Rozwiązywanie problemów

### Worker nie przetwarza zadań

1. **Sprawdź czy worker jest uruchomiony:**
   ```powershell
   .\check_worker.ps1
   ```

2. **Sprawdź czy Redis działa:**
   ```powershell
   redis-cli ping
   # Powinno zwrócić: PONG
   ```

3. **Sprawdź aktory:**
   ```powershell
   python diagnose_worker.py
   ```

4. **Wyczyść kolejkę (jeśli jest zasmieczona):**
   ```powershell
   python clear_dramatiq_queue.py
   ```

### Worker nie widzi aktorów

1. Upewnij się, że `queue_service.py` jest poprawnie zaimportowany
2. Sprawdź logi przy starcie workera - powinien pokazać zarejestrowane aktory
3. Uruchom diagnostykę: `python diagnose_worker.py`

### Worker się nie uruchamia

1. Sprawdź czy Dramatiq jest zainstalowany:
   ```powershell
   pip install dramatiq[redis]
   ```

2. Sprawdź czy Redis działa:
   ```powershell
   redis-cli ping
   ```

3. Sprawdź logi błędów w terminalu workera

## Zatrzymywanie Workera

Naciśnij `CTRL+C` w terminalu, gdzie działa worker.

## Zalety osobnego workera

✅ **Widoczne logi** - wszystkie logi z przetwarzania są widoczne w terminalu  
✅ **Łatwe debugowanie** - możesz zobaczyć dokładnie co się dzieje  
✅ **Niezależność** - możesz zrestartować API bez przerywania workera  
✅ **Lepsze na Windows** - unika problemów z multiprocessing w subprocess

## Przykładowy workflow

1. **Uruchom Redis** (jeśli nie działa jako serwis)
2. **Uruchom API + Daemon:**
   ```powershell
   python run.py
   ```
3. **W osobnym terminalu uruchom Worker:**
   ```powershell
   python run_worker.py
   ```
4. **Wyślij zadanie** przez API (np. upload batch)
5. **Obserwuj logi workera** - zobaczysz przetwarzanie w czasie rzeczywistym
