# Instrukcja konfiguracji systemu kolejkowania (Dramatiq)

## Wymagania

1. **Redis** - musi być zainstalowany i uruchomiony
2. **Dramatiq** - zainstalowany przez `pip install -r requirements.txt`

## Instalacja Redis

### Windows:
```bash
# Używając Chocolatey
choco install redis-64

# Lub pobierz z: https://github.com/microsoftarchive/redis/releases
# Lub użyj WSL: wsl --install, potem: sudo apt-get install redis-server
```

### Linux/Mac:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# Mac (Homebrew)
brew install redis

# Uruchom Redis
redis-server
```

## Konfiguracja

1. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

2. **Uruchom Redis** (jeśli nie działa jako serwis):
```bash
redis-server
```

3. **Opcjonalnie** - ustaw `REDIS_URL` w `.env`:
```
REDIS_URL=redis://localhost:6379/0
```

## Uruchomienie

### Automatyczne (zalecane):
```bash
python run.py
```

`run.py` automatycznie uruchomi:
- API (port 7777)
- Daemon
- Dramatiq worker

### Ręczne uruchomienie:

**Terminal 1 - API:**
```bash
python -m uvicorn api_app:app --host 0.0.0.0 --port 7777
```

**Terminal 2 - Daemon:**
```bash
python agent_daemon.py
```

**Terminal 3 - Dramatiq Worker:**
```bash
python -m dramatiq queue_service
```

## Testowanie

```bash
python test_queue.py
```

## Monitorowanie

### Dramatiq Dashboard (zalecane)

Oficjalne narzędzie do monitorowania kolejek:

```bash
# Zainstaluj
pip install dramatiq_dashboard

# Uruchom
python run_dashboard.py
```

Dashboard dostępny pod: **http://localhost:8080**

Zobacz więcej w [QUEUE_MONITORING.md](QUEUE_MONITORING.md)

## Rozwiązywanie problemów

### Błąd: "Nie można połączyć się z Redis"
- Upewnij się, że Redis jest uruchomiony: `redis-cli ping` (powinno zwrócić `PONG`)
- Sprawdź `REDIS_URL` w `.env`

### Błąd: "dramatiq nie jest zainstalowany"
```bash
pip install dramatiq[redis] redis
```

### Worker nie przetwarza zadań
- Sprawdź czy worker jest uruchomiony: `python -m dramatiq queue_service`
- Sprawdź logi workera
- Sprawdź połączenie z Redis: `redis-cli`

