"""
supabase_client.py
==================

Moduł pomocniczy inicjalizujący i keszujący klienta Supabase.
"""
from functools import lru_cache
from typing import Optional

import logging

try:
    from supabase import Client, create_client  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - biblioteka opcjonalna
    Client = None  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]

import config


@lru_cache(maxsize=1)
def get_supabase_client() -> Optional["Client"]:
    """
    Zwraca zainicjalizowanego klienta Supabase lub None, jeśli konfiguracja jest niekompletna.
    """
    if not config.SUPABASE_URL or not config.SUPABASE_API_KEY:
        logging.debug("Supabase client not created: missing URL or API key.")
        return None

    if Client is None or create_client is None:
        logging.error(
            "Biblioteka 'supabase' nie jest zainstalowana. Dodaj ją do requirements.txt."
        )
        return None

    try:
        client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_API_KEY)
    except Exception as initialization_error:  # pragma: no cover - log diagnostyczny
        logging.error(
            "Nie udało się utworzyć klienta Supabase: %s",
            initialization_error,
            exc_info=True,
        )
        raise

    return client


def reset_supabase_client_cache() -> None:
    """
    Czyści cache klienta Supabase (przydatne w testach).
    """
    get_supabase_client.cache_clear()

