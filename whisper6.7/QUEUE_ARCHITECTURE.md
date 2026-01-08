# Architektura systemu kolejek - Jak to działa?

## Proste wyjaśnienie

**Wszystko używa TEGO SAMEGO Redis!**

```
┌─────────────────────────────────────────────────────────┐
│                    REDIS (localhost:6379)               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Kolejka zadań (dramatiq:queue:default)         │   │
│  │  Wyniki zadań (dramatiq:result:*)                │   │
│  │  Opóźnione zadania (dramatiq:delayed:*)          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
           ▲                    ▲                    ▲
           │                    │                    │
           │                    │                    │
    ┌──────┴──────┐      ┌──────┴──────┐      ┌──────┴──────┐
    │             │      │             │      │             │
    │  API        │      │  Worker     │      │  Dashboard  │
    │  (7777)     │      │  Dramatiq   │      │  (8080)     │
    │             │      │             │      │             │
    │  Zapisuje   │      │  Czyta i    │      │  Czyta      │
    │  zadania    │      │  przetwarza │      │  i pokazuje │
    │  do Redis   │      │  zadania    │      │  statystyki │
    └─────────────┘      └─────────────┘      └─────────────┘
```

## Jak to działa krok po kroku:

### 1. **API (api_app.py) - port 7777**
   - Gdy przychodzi zapytanie do `/chat/query`, API:
   - Tworzy zadanie i **zapisuje je do Redis** (do kolejki `dramatiq:queue:default`)
   - Zwraca `job_id` użytkownikowi

### 2. **Worker Dramatiq** (`python -m dramatiq queue_service`)
   - **Czyta zadania z Redis** (z kolejki `dramatiq:queue:default`)
   - Przetwarza zadanie (wyszukuje w Qdrant, generuje odpowiedź)
   - **Zapisuje wynik z powrotem do Redis** (do `dramatiq:result:default:{job_id}`)

### 3. **Dashboard** (`python run_dashboard.py`) - port 8080
   - **Czyta z tego samego Redis**
   - Pokazuje:
     - Ile zadań jest w kolejce
     - Które zadania są przetwarzane
     - Które są zakończone
     - Które mają błędy

## Wszystko używa tej samej konfiguracji:

Wszystkie komponenty czytają z `.env`:
```
REDIS_URL=redis://localhost:6379/0
```

- ✅ `queue_service.py` - używa tego Redis
- ✅ `api_app.py` - używa tego Redis (przez queue_service)
- ✅ `run_dashboard.py` - używa tego Redis
- ✅ Worker Dramatiq - używa tego Redis

## Przykład przepływu:

1. **Użytkownik wysyła zapytanie:**
   ```
   POST /chat/query
   → API zapisuje zadanie do Redis
   → Zwraca job_id: "abc123"
   ```

2. **Worker przetwarza:**
   ```
   Worker czyta "abc123" z Redis
   → Przetwarza zadanie
   → Zapisuje wynik do Redis (dramatiq:result:default:abc123)
   ```

3. **Dashboard pokazuje:**
   ```
   Dashboard czyta z Redis:
   - Widzi zadanie "abc123" w kolejce → Status: "pending"
   - Widzi wynik w Redis → Status: "completed"
   ```

## Ważne:

- **Jeden Redis** - wszystkie komponenty używają tego samego
- **Ta sama konfiguracja** - wszystkie czytają `REDIS_URL` z `.env`
- **Dashboard tylko czyta** - nie modyfikuje zadań (tylko pokazuje)
- **Worker i API** - zapisują/czytają zadania

## Sprawdzenie czy wszystko działa:

```bash
# 1. Sprawdź czy Redis działa
redis-cli ping
# Powinno zwrócić: PONG

# 2. Sprawdź czy są zadania w Redis
redis-cli
> KEYS dramatiq:*
> LLEN dramatiq:queue:default

# 3. Uruchom dashboard - zobaczy te same dane
python run_dashboard.py
```

## Podsumowanie:

**Dashboard Dramatiq = narzędzie do podglądu tego, co jest w Twoim Redisie**

Nie potrzebujesz osobnego Redis dla dashboardu - używa tego samego, co Twój system!
