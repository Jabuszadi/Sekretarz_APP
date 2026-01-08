# Instrukcja uruchomienia serwera do testów kolejki

## Szybki start

### 1. Uruchom serwer główny (minimal_mcp_server.py)

W osobnym terminalu:

```bash
python minimal_mcp_server.py
```

Lub z uvicorn:

```bash
uvicorn minimal_mcp_server:app --host 0.0.0.0 --port 8000 --reload
```

**Poczekaj aż zobaczysz w logach:**
- `✅ pgBoss uruchomiony`
- `✅ Handler wiadomości zarejestrowany w pgBoss`
- `🎉 Minimal FastMCP Server startup complete!`

### 2. Uruchom test kolejki

W innym terminalu:

```bash
python test_queue.py
```

## Porty używane w systemie

- **8000** - minimal_mcp_server.py (główny serwer z kolejką)
- **7777** - api_app.py (główny API)
- **8002** - database_agent.py (agent bazodanowy)
- **8003** - audio_processing_agent.py (agent audio)

## Wymagane zależności

```bash
pip install pgboss>=8.0.0 aiohttp>=3.9.0
```

## Sprawdzenie czy serwer działa

```bash
# Sprawdź czy serwer odpowiada
curl http://localhost:8000/

# Lub w PowerShell:
Invoke-WebRequest -Uri http://localhost:8000/
```

## Rozwiązywanie problemów

### Problem: "Cannot connect to host"
- Sprawdź czy serwer jest uruchomiony
- Sprawdź czy port 8000 nie jest zajęty
- Sprawdź czy firewall nie blokuje połączenia

### Problem: "pgboss nie jest zainstalowany"
```bash
pip install pgboss>=8.0.0
```

### Problem: "Brak SUPABASE_DB_URL"
- Upewnij się, że masz ustawione zmienne środowiskowe:
  - `SUPABASE_DB_URL`
  - `USE_SUPABASE=1`

