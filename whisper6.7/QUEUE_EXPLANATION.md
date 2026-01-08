# Wyjaśnienie: Dlaczego Dashboard Dramatiq pokazuje 0 jobs?

## Problem

Dashboard Dramatiq pokazuje 0 jobs, mimo że widzisz w logach przetwarzanie plików.

## Przyczyna

**Dramatiq jest używany TYLKO dla wiadomości czatu (`/chat/query`), a NIE dla przetwarzania plików audio!**

## Dwa różne systemy:

### 1. **Dramatiq** - tylko dla czatu
   - Endpoint: `/chat/query`
   - Używa: `enqueue_message()` → Dramatiq → Redis
   - Widoczne w: Dashboard Dramatiq

### 2. **Asyncio Tasks** - dla przetwarzania plików
   - Endpoint: `/upload_multiple/`, `/process_file/`
   - Używa: `asyncio.create_task(run_batch_processing(...))`
   - NIE używa Dramatiq!
   - NIE widoczne w Dashboard Dramatiq

## Jak to sprawdzić:

### Zadania czatu (widoczne w Dashboard):
```bash
# Wyślij wiadomość przez czat
POST /chat/query
{
  "query": "Test wiadomości"
}

# To zadanie pojawi się w Dashboard Dramatiq!
```

### Przetwarzanie plików (NIE widoczne w Dashboard):
```bash
# Prześlij plik audio
POST /upload_multiple/

# To zadanie NIE pojawi się w Dashboard Dramatiq!
# Jest przetwarzane przez asyncio.create_task()
```

## Jak monitorować przetwarzanie plików?

### 1. Przez API endpointy:
```bash
# Status batcha
GET /processed_batches/{batch_job_id}/details

# Status pojedynczego pliku
GET /processed_files/{file_job_id}
```

### 2. Przez logi:
- Wszystkie zadania przetwarzania plików są logowane z `[batch_job_id]` i `[file_job_id]`
- Sprawdź logi API, żeby zobaczyć postęp

### 3. Przez bazę danych:
```sql
-- Sprawdź batch jobs
SELECT * FROM batch_jobs WHERE status = 'processing';

-- Sprawdź pliki
SELECT * FROM processed_files WHERE status = 'processing';
```

## ✅ Rozwiązanie: Przetwarzanie plików dodane do Dramatiq!

**Zrobione!** Przetwarzanie plików teraz używa Dramatiq:

1. ✅ Stworzono aktora Dramatiq `process_batch` dla przetwarzania batchów
2. ✅ Zmieniono `/upload_multiple/` żeby używał `enqueue_batch_processing()` zamiast `asyncio.create_task()`
3. ✅ Batchy są teraz widoczne w Dashboard Dramatiq!

**Uwaga:** Worker Dramatiq musi być uruchomiony, żeby batchy były przetwarzane:
```bash
python -m dramatiq queue_service
```

## Podsumowanie:

- ✅ Dashboard Dramatiq pokazuje zadania z `/chat/query`
- ❌ Dashboard Dramatiq NIE pokazuje przetwarzania plików
- 📊 Przetwarzanie plików monitoruj przez API/logi/bazę danych
