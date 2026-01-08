# Rate Limiting dla Gemini API

## Problem

Jeśli masz niskie limity API dla Google Gemini:
- **RPM (Rate Per Minute)**: np. 10 requestów na minutę
- **RPD (Rate Per Day)**: np. 84 requesty na dzień

System automatycznie kontroluje te limity i czeka, jeśli trzeba, żeby nie przekroczyć limitów.

## Jak to działa

System używa **rate limitera**, który:
1. **Sprawdza limity** przed każdym wywołaniem Gemini API
2. **Czeka automatycznie** jeśli limit jest przekroczony
3. **Zwiększa liczniki** po każdym requestcie
4. **Używa Redis** do synchronizacji między procesami (jeśli dostępny)

## Konfiguracja

### Zmienne środowiskowe

Dodaj do `.env`:

```bash
# Limity Gemini API
GEMINI_RPM_LIMIT=10      # Maksymalna liczba requestów na minutę (domyślnie 10)
GEMINI_RPD_LIMIT=84      # Maksymalna liczba requestów na dzień (domyślnie 84)
```

### Przykład dla Twoich limitów

Jeśli masz:
- RPM: 10/5 (10 requestów na minutę)
- RPD: 84/20 (84 requesty na dzień)

Ustaw w `.env`:
```bash
GEMINI_RPM_LIMIT=10
GEMINI_RPD_LIMIT=84
```

## Gdzie działa rate limiting

Rate limiting jest automatycznie dodany do:

1. **Transkrypcja audio** (`GeminiTranscriber.transcribe_file()`)
   - Każdy segment audio jest transkrybowany przez Gemini
   - Rate limiter czeka między segmentami, jeśli trzeba

2. **Generowanie podsumowań** (`generate_minutes_of_meeting()`)
   - Każda sekcja podsumowania używa Gemini
   - Rate limiter kontroluje tempo generowania

3. **Odpowiedzi czatu** (`generate_chat_response()`)
   - Każda odpowiedź czatu używa Gemini
   - Rate limiter czeka, jeśli limit jest przekroczony

## Jak sprawdzić statystyki

Możesz sprawdzić użycie limitów:

```python
from gemini_rate_limiter import get_gemini_rate_limiter

rate_limiter = get_gemini_rate_limiter()
stats = rate_limiter.get_stats()

print(f"RPM: {stats['rpm_used']}/{stats['rpm_limit']} (pozostało: {stats['rpm_remaining']})")
print(f"RPD: {stats['rpd_used']}/{stats['rpd_limit']} (pozostało: {stats['rpd_remaining']})")
```

## Co się dzieje przy przekroczeniu limitu?

### Limit minutowy (RPM)
- System **automatycznie czeka** do końca minuty
- Loguje informację: `"Przekroczono limit RPM (10/min). Poczekaj X sekund."`
- Po upływie minuty, request jest wykonywany

### Limit dzienny (RPD)
- System **blokuje requesty** do następnego dnia
- Loguje błąd: `"Przekroczono limit RPD (84/dzień). Spróbuj jutro."`
- Request nie jest wykonywany

## Backend rate limitera

### Redis (zalecane)
- Jeśli Redis jest dostępny, rate limiter używa Redis do przechowywania liczników
- Działa między procesami (API, Worker, itp.)
- Synchronizuje limity między wszystkimi instancjami

### Pamięć lokalna (fallback)
- Jeśli Redis nie jest dostępny, używa lokalnej pamięci
- Działa tylko w obrębie jednego procesu
- Nie synchronizuje między procesami

## Przykładowe logi

```
✅ Gemini Rate Limiter zainicjalizowany: RPM=10, RPD=84
✅ Gemini Rate Limiter: Używam Redis (redis://localhost:6379/0)
Gemini Rate Limiter: Przekroczono limit RPM (10/min). Poczekaj 45 sekund. Czekam 5.0s...
```

## Ważne uwagi

1. **Rate limiter działa automatycznie** - nie musisz nic robić
2. **Może spowolnić przetwarzanie** - jeśli masz dużo segmentów audio, system będzie czekał między requestami
3. **Redis jest zalecany** - bez Redis, rate limiting działa tylko w jednym procesie
4. **Limity są globalne** - wszystkie wywołania Gemini (transkrypcja, podsumowania, czat) używają tych samych limitów

## Rozwiązywanie problemów

### Problem: "Rate limit Gemini API: Przekroczono limit RPD"
**Rozwiązanie**: Poczekaj do następnego dnia lub zwiększ limit w Google Cloud Console.

### Problem: Rate limiter nie działa między procesami
**Rozwiązanie**: Upewnij się, że Redis jest uruchomiony i `REDIS_URL` jest poprawnie skonfigurowany.

### Problem: Przetwarzanie jest bardzo wolne
**Rozwiązanie**: To normalne przy niskich limitach. System automatycznie czeka między requestami. Rozważ:
- Zwiększenie limitów w Google Cloud Console
- Użycie innego providera transkrypcji (np. AssemblyAI)
- Przetwarzanie plików w mniejszych partiach
