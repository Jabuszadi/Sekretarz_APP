# Monitorowanie kolejek Dramatiq

Ten dokument opisuje jak monitorować kolejki zadań w systemie.

## ✅ Co jest monitorowane przez Dramatiq?

**Dashboard Dramatiq pokazuje WSZYSTKIE zadania!**

- ✅ **Widoczne w Dashboard**: Zadania z `/chat/query` (wiadomości czatu)
- ✅ **Widoczne w Dashboard**: Przetwarzanie batchów plików (`/upload_multiple/`)

Wszystko używa Dramatiq i jest widoczne w Dashboard!

## Sposoby monitorowania

### 0. 🎯 Dramatiq Dashboard (OFICJALNE NARZĘDZIE - ZALECANE)

Dramatiq ma oficjalne narzędzie do monitorowania - **dramatiq_dashboard**. To jest webowy interfejs z pełnym monitoringiem.

#### ⚠️ WAŻNE: Dashboard używa TEGO SAMEGO Redis co Twój system!

Dashboard **tylko czyta** z Redis, żeby pokazać statystyki. Nie potrzebujesz osobnego Redis!

**Przepływ:**
- API zapisuje zadania → Redis
- Worker czyta zadania z Redis → przetwarza → zapisuje wyniki do Redis
- Dashboard czyta z Redis → pokazuje statystyki

Wszystkie komponenty używają tej samej zmiennej `REDIS_URL` z `.env`.

#### Instalacja:
```bash
pip install dramatiq_dashboard
# Opcjonalnie dla lepszej wydajności:
pip install bjoern
```

#### Uruchomienie:
```bash
python run_dashboard.py
```

Dashboard będzie dostępny pod: **http://localhost:8080**

#### Konfiguracja (opcjonalna):
Możesz ustawić zmienne środowiskowe w `.env`:
```
DASHBOARD_PORT=8080
DASHBOARD_HOST=127.0.0.1
REDIS_URL=redis://localhost:6379/0  # Ten sam Redis co używa Twój system!
```

#### Co oferuje dashboard:
- ✅ Wizualizacja wszystkich kolejek
- ✅ Lista zadań w czasie rzeczywistym
- ✅ Status zadań (pending, processing, completed, failed)
- ✅ Szczegóły każdego zadania
- ✅ Statystyki i metryki
- ✅ Możliwość ponownego uruchomienia zadań
- ✅ Filtrowanie i wyszukiwanie

**To jest najprostszy i najbardziej kompletny sposób monitorowania!**

Zobacz więcej w [QUEUE_ARCHITECTURE.md](QUEUE_ARCHITECTURE.md) - szczegółowe wyjaśnienie jak to działa.

### 1. API Endpoints (alternatywa)

#### Statystyki kolejki
```bash
GET /queue/stats
```

Zwraca:
```json
{
  "pending": 5,
  "completed": 120,
  "failed": 2,
  "total": 127,
  "queue_size": 5,
  "redis_connected": true
}
```

#### Lista aktywnych zadań
```bash
GET /queue/jobs?limit=50
```

Zwraca:
```json
{
  "jobs": [
    {
      "job_id": "abc123...",
      "status": "pending",
      "queue": "dramatiq:queue:default"
    }
  ],
  "count": 1,
  "limit": 50
}
```

### 2. Skrypt CLI - monitor_queue.py (alternatywa)

Najprostszy sposób monitorowania w czasie rzeczywistym.

#### Podstawowe użycie:
```bash
python monitor_queue.py
```

#### Opcje:
```bash
# Odświeżaj co 5 sekund
python monitor_queue.py -i 5

# Tylko statystyki, bez listy zadań
python monitor_queue.py --no-jobs

# Pokaż do 50 zadań
python monitor_queue.py -j 50

# Użyj innego URL API
python monitor_queue.py --api-url http://localhost:8000
```

#### Przykładowy output:
```
============================================================
📊 STATYSTYKI KOLEJKI DRAMATIQ
============================================================
Redis: ✅ Połączono

⏳ Oczekujące:          5
✅ Zakończone:        120
❌ Błędy:              2
📊 Razem:            127
📦 Rozmiar kolejki:    5

============================================================
📋 AKTYWNE ZADANIA (5)
============================================================
  1. ⏳ abc123def456... | Status: pending
  2. ⏳ def456ghi789... | Status: pending
  ...

============================================================
🕐 Ostatnie odświeżenie: 14:45:30
Naciśnij CTRL+C, aby zakończyć
```

### 3. Sprawdzanie statusu pojedynczego zadania

```bash
GET /chat/query/status/{job_id}
```

Zwraca:
```json
{
  "job_id": "abc123...",
  "status": "completed",
  "response": "Odpowiedź...",
  "error_message": null,
  "retry_count": 0,
  "max_retries": 3
}
```

### 4. Bezpośrednie monitorowanie Redis

Możesz również monitorować Redis bezpośrednio:

```bash
# Połącz się z Redis
redis-cli

# Sprawdź wszystkie klucze Dramatiq
KEYS dramatiq:*

# Sprawdź rozmiar kolejki
LLEN dramatiq:queue:default

# Sprawdź opóźnione zadania
ZCARD dramatiq:delayed:default

# Sprawdź wyniki
KEYS dramatiq:result:*
```

## Statusy zadań

- **pending** - Zadanie czeka w kolejce
- **delayed** - Zadanie jest opóźnione (zaplanowane na później)
- **processing** - Zadanie jest aktualnie przetwarzane
- **completed** - Zadanie zakończone sukcesem
- **failed** - Zadanie zakończone błędem

## Rozwiązywanie problemów

### Problem: "Nie można połączyć się z API"
- Upewnij się, że API działa: `curl http://localhost:7777/health`
- Sprawdź czy port 7777 nie jest zajęty
- Sprawdź zmienną środowiskową `API_URL`

### Problem: "Redis nie jest połączony"
- Upewnij się, że Redis działa: `redis-cli ping` (powinno zwrócić `PONG`)
- Sprawdź `REDIS_URL` w `.env`
- Sprawdź czy Redis nie jest zablokowany przez firewall

### Problem: Worker nie przetwarza zadań
- Sprawdź czy worker jest uruchomiony: `python -m dramatiq queue_service`
- Sprawdź logi workera
- Sprawdź czy są błędy w statystykach (`failed` > 0)

### Problem: Zadania utknęły w kolejce
- Sprawdź czy worker jest uruchomiony
- Sprawdź logi workera pod kątem błędów
- Sprawdź czy Redis ma wystarczającą pamięć
- Możesz zrestartować workera: zatrzymaj (CTRL+C) i uruchom ponownie

## Automatyczne monitorowanie

Możesz użyć `monitor_queue.py` w skrypcie bash/PowerShell do automatycznego monitorowania:

```bash
# PowerShell
while ($true) {
    python monitor_queue.py --no-jobs
    Start-Sleep -Seconds 10
}
```

```bash
# Bash
while true; do
    python monitor_queue.py --no-jobs
    sleep 10
done
```

## Monitorowanie przetwarzania plików (NIE Dramatiq)

Przetwarzanie plików audio NIE używa Dramatiq, więc nie jest widoczne w Dashboard Dramatiq.

### Przez API:
```bash
# Status batcha
GET /processed_batches/{batch_job_id}/details

# Status pojedynczego pliku  
GET /processed_files/{file_job_id}
```

### Przez logi:
Wszystkie zadania przetwarzania plików są logowane z `[batch_job_id]` i `[file_job_id]`:
```
INFO:root:[6baed89e-d9e6-4106-ae7b-4899088238da] Processing file_job_id: 90f49f98-0289-4aa7-9ef4-38b7ce58f673
```

### Przez bazę danych:
```sql
-- Sprawdź batch jobs
SELECT * FROM batch_jobs WHERE status = 'processing';

-- Sprawdź pliki
SELECT * FROM processed_files WHERE status = 'processing';
```

## Integracja z innymi narzędziami

### Prometheus/Grafana
Możesz użyć endpointu `/queue/stats` jako źródła danych dla Prometheus.

### Alerting
Możesz skonfigurować alerty na podstawie statystyk:
- Jeśli `failed` > 10 - wyślij alert
- Jeśli `pending` > 100 - wyślij alert
- Jeśli `redis_connected` == false - wyślij alert

## Przykłady użycia w kodzie

```python
import httpx
from queue_service import get_queue_stats, list_active_jobs

# Pobierz statystyki (tylko zadania czatu!)
stats = await get_queue_stats()
print(f"Oczekujące zadania czatu: {stats['pending']}")

# Pobierz listę aktywnych zadań (tylko zadania czatu!)
jobs = await list_active_jobs(limit=10)
for job in jobs:
    print(f"Zadanie czatu {job['job_id']}: {job['status']}")
```
