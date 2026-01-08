"""
Rate limiter dla Google Gemini API.
Kontroluje RPM (Rate Per Minute) i RPD (Rate Per Day) limity.
"""
import time
import logging
import os
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis nie jest dostępny. Rate limiting będzie działał tylko w pamięci (nie będzie działał między procesami).")


class GeminiRateLimiter:
    """
    Rate limiter dla Gemini API.
    
    Kontroluje:
    - RPM (Rate Per Minute): maksymalna liczba requestów na minutę
    - RPD (Rate Per Day): maksymalna liczba requestów na dzień
    """
    
    def __init__(
        self,
        rpm_limit: int = 10,
        rpd_limit: int = 84,
        redis_url: Optional[str] = None,
    ):
        """
        Args:
            rpm_limit: Maksymalna liczba requestów na minutę (domyślnie 10)
            rpd_limit: Maksymalna liczba requestów na dzień (domyślnie 84)
            redis_url: URL Redis do przechowywania liczników (opcjonalne)
        """
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        
        # Redis connection (jeśli dostępny)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test połączenia
                self.redis_client.ping()
                logger.info(f"✅ Gemini Rate Limiter: Używam Redis ({redis_url})")
            except Exception as e:
                logger.warning(f"⚠️  Gemini Rate Limiter: Nie można połączyć z Redis: {e}. Używam pamięci lokalnej.")
                self.redis_client = None
        
        # Fallback: lokalne liczniki w pamięci (działa tylko w obrębie jednego procesu)
        if not self.redis_client:
            self._local_minute_requests = []
            self._local_day_requests = []
            logger.info("⚠️  Gemini Rate Limiter: Używam pamięci lokalnej (nie działa między procesami)")
    
    def _get_redis_key_minute(self) -> str:
        """Klucz Redis dla licznika minutowego."""
        current_minute = int(time.time() // 60)
        return f"gemini:rate_limit:minute:{current_minute}"
    
    def _get_redis_key_day(self) -> str:
        """Klucz Redis dla licznika dziennego."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"gemini:rate_limit:day:{today}"
    
    def _check_limits_redis(self) -> Tuple[bool, Optional[str]]:
        """
        Sprawdza limity używając Redis.
        
        Returns:
            (allowed, error_message) - allowed=True jeśli można wykonać request
        """
        try:
            current_minute = int(time.time() // 60)
            minute_key = self._get_redis_key_minute()
            today = datetime.now().strftime("%Y-%m-%d")
            day_key = self._get_redis_key_day()
            
            # Sprawdź licznik minutowy
            minute_count = self.redis_client.get(minute_key)
            if minute_count is None:
                minute_count = 0
            else:
                minute_count = int(minute_count)
            
            if minute_count >= self.rpm_limit:
                wait_seconds = 60 - (time.time() % 60)
                return False, f"Przekroczono limit RPM ({self.rpm_limit}/min). Poczekaj {wait_seconds:.0f} sekund."
            
            # Sprawdź licznik dzienny
            day_count = self.redis_client.get(day_key)
            if day_count is None:
                day_count = 0
            else:
                day_count = int(day_count)
            
            if day_count >= self.rpd_limit:
                return False, f"Przekroczono limit RPD ({self.rpd_limit}/dzień). Spróbuj jutro."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Błąd podczas sprawdzania limitów w Redis: {e}")
            # Fallback: pozwól request (lepiej niż blokować wszystko)
            return True, None
    
    def _check_limits_local(self) -> Tuple[bool, Optional[str]]:
        """
        Sprawdza limity używając lokalnej pamięci.
        
        Returns:
            (allowed, error_message) - allowed=True jeśli można wykonać request
        """
        current_time = time.time()
        current_minute = int(current_time // 60)
        
        # Wyczyść stare wpisy (starsze niż 1 minuta)
        self._local_minute_requests = [
            ts for ts in self._local_minute_requests
            if int(ts // 60) == current_minute
        ]
        
        # Sprawdź limit minutowy
        if len(self._local_minute_requests) >= self.rpm_limit:
            wait_seconds = 60 - (current_time % 60)
            return False, f"Przekroczono limit RPM ({self.rpm_limit}/min). Poczekaj {wait_seconds:.0f} sekund."
        
        # Wyczyść stare wpisy dzienne (starsze niż 24 godziny)
        day_ago = current_time - 86400
        self._local_day_requests = [
            ts for ts in self._local_day_requests
            if ts > day_ago
        ]
        
        # Sprawdź limit dzienny
        if len(self._local_day_requests) >= self.rpd_limit:
            return False, f"Przekroczono limit RPD ({self.rpd_limit}/dzień). Spróbuj jutro."
        
        return True, None
    
    def _increment_redis(self):
        """Zwiększa liczniki w Redis."""
        try:
            minute_key = self._get_redis_key_minute()
            day_key = self._get_redis_key_day()
            
            # Zwiększ licznik minutowy (wygaśnij po 2 minutach)
            self.redis_client.incr(minute_key)
            self.redis_client.expire(minute_key, 120)  # 2 minuty
            
            # Zwiększ licznik dzienny (wygaśnij po 25 godzinach)
            self.redis_client.incr(day_key)
            self.redis_client.expire(day_key, 90000)  # ~25 godzin
            
        except Exception as e:
            logger.error(f"Błąd podczas zwiększania liczników w Redis: {e}")
    
    def _increment_local(self):
        """Zwiększa liczniki w lokalnej pamięci."""
        current_time = time.time()
        self._local_minute_requests.append(current_time)
        self._local_day_requests.append(current_time)
    
    def acquire(self) -> Tuple[bool, Optional[str]]:
        """
        Sprawdza czy można wykonać request i zwiększa liczniki.
        
        Returns:
            (allowed, error_message) - allowed=True jeśli można wykonać request
        """
        if self.redis_client:
            allowed, error = self._check_limits_redis()
            if allowed:
                self._increment_redis()
            return allowed, error
        else:
            allowed, error = self._check_limits_local()
            if allowed:
                self._increment_local()
            return allowed, error
    
    def wait_if_needed(self) -> Optional[str]:
        """
        Czeka jeśli trzeba, żeby nie przekroczyć limitów.
        
        Returns:
            error_message jeśli nie można wykonać requestu (po oczekiwaniu)
        """
        max_wait_time = 60  # Maksymalnie 60 sekund czekania
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            allowed, error = self.acquire()
            if allowed:
                return None
            
            # Jeśli przekroczono limit dzienny, nie ma sensu czekać
            if "dzień" in error.lower() or "jutro" in error.lower():
                return error
            
            # Czekaj chwilę i spróbuj ponownie
            wait_time = min(5.0, 60 - (time.time() % 60))
            logger.info(f"Gemini Rate Limiter: {error}. Czekam {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        return "Przekroczono limit czasu oczekiwania na rate limiter."
    
    def get_stats(self) -> dict:
        """
        Zwraca statystyki użycia.
        
        Returns:
            Dict z informacjami o użyciu limitów
        """
        if self.redis_client:
            try:
                minute_key = self._get_redis_key_minute()
                day_key = self._get_redis_key_day()
                
                minute_count = int(self.redis_client.get(minute_key) or 0)
                day_count = int(self.redis_client.get(day_key) or 0)
                
                return {
                    "rpm_used": minute_count,
                    "rpm_limit": self.rpm_limit,
                    "rpm_remaining": max(0, self.rpm_limit - minute_count),
                    "rpd_used": day_count,
                    "rpd_limit": self.rpd_limit,
                    "rpd_remaining": max(0, self.rpd_limit - day_count),
                    "backend": "redis",
                }
            except Exception as e:
                logger.error(f"Błąd podczas pobierania statystyk z Redis: {e}")
                return {"error": str(e)}
        else:
            current_time = time.time()
            current_minute = int(current_time // 60)
            day_ago = current_time - 86400
            
            minute_requests = [
                ts for ts in self._local_minute_requests
                if int(ts // 60) == current_minute
            ]
            day_requests = [
                ts for ts in self._local_day_requests
                if ts > day_ago
            ]
            
            return {
                "rpm_used": len(minute_requests),
                "rpm_limit": self.rpm_limit,
                "rpm_remaining": max(0, self.rpm_limit - len(minute_requests)),
                "rpd_used": len(day_requests),
                "rpd_limit": self.rpd_limit,
                "rpd_remaining": max(0, self.rpd_limit - len(day_requests)),
                "backend": "local_memory",
            }


# Globalna instancja rate limitera
_gemini_rate_limiter: Optional[GeminiRateLimiter] = None


def get_gemini_rate_limiter() -> GeminiRateLimiter:
    """Zwraca globalną instancję rate limitera."""
    global _gemini_rate_limiter
    
    if _gemini_rate_limiter is None:
        # Pobierz limity z zmiennych środowiskowych
        rpm_limit = int(os.getenv("GEMINI_RPM_LIMIT", "10"))
        rpd_limit = int(os.getenv("GEMINI_RPD_LIMIT", "84"))
        
        _gemini_rate_limiter = GeminiRateLimiter(
            rpm_limit=rpm_limit,
            rpd_limit=rpd_limit,
        )
        logger.info(f"✅ Gemini Rate Limiter zainicjalizowany: RPM={rpm_limit}, RPD={rpd_limit}")
    
    return _gemini_rate_limiter
