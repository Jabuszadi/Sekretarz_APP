"""
db.postgres
===========

Warstwa dostępu do bazy Postgres (Supabase) z wykorzystaniem psycopg-pool.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, Sequence

import logging
import os

import config

try:
    from psycopg_pool import ConnectionPool  # type: ignore[import-untyped]
    from psycopg.rows import dict_row  # type: ignore[import-untyped]
    import psycopg  # type: ignore[import-untyped]
except ImportError as import_error:  # pragma: no cover - moduł opcjonalny
    ConnectionPool = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    psycopg = None  # type: ignore[assignment]
    logging.debug(
        "psycopg nie jest dostępny (%s). Jeśli USE_SUPABASE=1, doinstaluj psycopg.",
        import_error,
    )


_pool: Optional["ConnectionPool"] = None


def _ensure_pool() -> "ConnectionPool":
    """
    Tworzy (lazy) pulę połączeń Postgresa.
    """
    global _pool
    if _pool is not None:
        return _pool

    if ConnectionPool is None or psycopg is None:
        raise RuntimeError(
            "psycopg_pool nie jest zainstalowany. Dodaj 'psycopg[binary]' do dependencies."
        )
    if not config.SUPABASE_DB_URL:
        raise RuntimeError("Brak SUPABASE_DB_URL – nie można utworzyć połączenia Postgres.")

    max_size = int(os.getenv("SUPABASE_POOL_MAX", "10"))
    kwargs = {
        "conninfo": config.SUPABASE_DB_URL,
        "max_size": max_size,
        "kwargs": {
            "application_name": "sekretarz_backend",
            "connect_timeout": int(os.getenv("SUPABASE_CONNECT_TIMEOUT", "10")),
            "prepare_threshold": 0,
        },
    }
    _pool = ConnectionPool(**kwargs)
    logging.info("Utworzono pulę połączeń Postgres (Supabase).")
    return _pool


@contextmanager
def connection() -> Generator["psycopg.Connection", None, None]:
    """
    Zwraca połączenie z puli. Autocommit = False, commit po wyjściu z kontekstu.
    """
    pool = _ensure_pool()
    with pool.connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


@contextmanager
def cursor(
    *,
    row_mode=dict_row,
) -> Generator["psycopg.Cursor[Any]", None, None]:
    """
    Zwraca kursor z dict_row (domyślnie), zarządzany przez kontekst.
    """
    with connection() as conn:
        with conn.cursor(row_factory=row_mode) as cur:
            yield cur


def execute(
    query: str,
    params: Optional[Sequence[Any]] = None,
    *,
    fetch: Optional[str] = None,
) -> Optional[Iterable[Any]]:
    """
    Wygodna funkcja do wykonywania zapytań.

    :param query: zapytanie SQL
    :param params: sekwencja parametrów
    :param fetch: None / "one" / "all"
    """
    with cursor() as cur:
        cur.execute(query, params, prepare=False)
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
    return None


def reset_pool() -> None:
    """
    Czyści pulę – przydatne w testach.
    """
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

