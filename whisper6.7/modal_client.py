"""
Helper do wywoływania funkcji diarizacji w Modal z backendu FastAPI.

Zakłada, że aplikacja została zdeployowana (`modal deploy modal_diarization.py`)
i dostępna jest funkcja `run_diarization`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import config

DEFAULT_APP_NAME = config.MODAL_DIARIZATION_APP
DEFAULT_FUNCTION_NAME = config.MODAL_DIARIZATION_FUNCTION


class ModalClientError(RuntimeError):
    """Podstawowy wyjątek klienta Modal."""


class ModalNotAvailableError(ModalClientError):
    """Sygnalizuje brak pakietu Modal lub niewłaściwą konfigurację."""


@lru_cache(maxsize=1)
def _get_modal_function():
    """
    Zwraca uchwyt do funkcji Modal; cache'owany, aby uniknąć wielokrotnych lookupów.
    """
    try:
        from modal import App, Function  # Lokalny import, aby nie wymagać pakietu podczas lintowania
        try:
            from modal.experimental import get_app_objects
        except ImportError:
            get_app_objects = None
    except ImportError as exc:  # pragma: no cover - brak w środowisku testowym
        raise ModalNotAvailableError(
            "Pakiet 'modal' nie jest zainstalowany. Zainstaluj go lub ustaw "
            "USE_MODAL_DIARIZATION=0."
        ) from exc

    app_name = os.getenv("MODAL_DIARIZATION_APP", DEFAULT_APP_NAME)
    function_name = os.getenv("MODAL_DIARIZATION_FUNCTION", DEFAULT_FUNCTION_NAME)
    environment = os.getenv("MODAL_ENV", config.MODAL_ENV)

    logging.debug(
        "Lookup funkcji Modal: %s.%s (env=%s)",
        app_name,
        function_name,
        environment,
    )

    # Najpierw spróbuj użyć Function.from_name (obsługiwane w nowszych SDK).
    from_name = getattr(Function, "from_name", None)
    if callable(from_name):
        try:
            from_name_kwargs: Dict[str, Any] = {}
            if environment:
                from_name_kwargs["environment_name"] = environment
            handle = from_name(app_name, function_name, **from_name_kwargs)
            if handle:
                logging.debug("Function.from_name znalazło uchwyt do '%s'.", function_name)
                return handle
        except Exception as from_name_error:
            logging.debug("Function.from_name nie powiodło się: %s", from_name_error)

    # Następnie spróbuj API Function.lookup (jeżeli istnieje w tej wersji).
    function_lookup = getattr(Function, "lookup", None)
    if callable(function_lookup):
        try:
            lookup_kwargs: Dict[str, Any] = {}
            if environment:
                lookup_kwargs["environment_name"] = environment
            return function_lookup(app_name, function_name, **lookup_kwargs)
        except TypeError as type_error:
            logging.debug(
                "Function.lookup nie obsługuje parametru 'environment_name' (%s). "
                "Próbuję bez dodatkowych argumentów.",
                type_error,
            )
            try:
                return function_lookup(app_name, function_name)
            except Exception as lookup_error:
                logging.debug(
                    "Function.lookup(%s, %s) bez env również nie powiodło się (%s). "
                    "Próbuję kolejnych metod.",
                    app_name,
                    function_name,
                    lookup_error,
                )
        except Exception as lookup_error:
            logging.debug(
                "Function.lookup(%s, %s, env=%s) nie powiodło się (%s). Próbuję kolejnych metod.",
                app_name,
                function_name,
                environment,
                lookup_error,
            )
    else:
        logging.debug("Function.lookup nie jest dostępne w tej wersji SDK.")

    # Spróbuj uzyskać funkcje przez modal.experimental.get_app_objects (jeśli dostępne).
    if 'get_app_objects' in locals() and callable(get_app_objects):
        try:
            experimental_kwargs: Dict[str, Any] = {}
            if environment:
                experimental_kwargs["environment_name"] = environment
            objects = get_app_objects(app_name, **experimental_kwargs)
            candidate = objects.get(function_name) if isinstance(objects, dict) else None
            if candidate and callable(getattr(candidate, "call", None)):
                logging.debug("modal.experimental.get_app_objects zwróciło uchwyt do '%s'.", function_name)
                return candidate
        except Exception as experimental_error:
            logging.debug("modal.experimental.get_app_objects nie powiodło się: %s", experimental_error)

    try:
        app_lookup_kwargs: Dict[str, Any] = {}
        if environment:
            app_lookup_kwargs["environment_name"] = environment
        client = App.lookup(app_name, **app_lookup_kwargs)
    except TypeError as type_error:
        logging.debug(
            "App.lookup nie obsługuje parametru 'environment_name' (%s). "
            "Próbuję bez dodatkowych argumentów.",
            type_error,
        )
        try:
            client = App.lookup(app_name)
        except Exception as app_lookup_error:
            raise ModalClientError(
                f"Nie udało się odnaleźć aplikacji Modal '{app_name}' "
                f"(env={environment}) bez argumentu environment: {app_lookup_error}"
            ) from app_lookup_error
    except Exception as app_lookup_error:
        raise ModalClientError(
            f"Nie udało się odnaleźć aplikacji Modal '{app_name}' (env={environment}): "
            f"{app_lookup_error}"
        ) from app_lookup_error

    available_names: list[str] = []

    # Jeśli SDK udostępnia metodę get_function, użyj jej w pierwszej kolejności
    get_fn = getattr(client, "get_function", None)
    if callable(get_fn):
        try:
            func = get_fn(function_name)
            if func:
                return func
        except KeyError:
            pass
        except Exception as get_fn_error:
            logging.debug("client.get_function('%s') call failed: %s", function_name, get_fn_error)
        try:
            maybe_mapping = get_fn()
            if isinstance(maybe_mapping, dict):
                available_names.extend(maybe_mapping.keys())
        except Exception as mapping_error:
            logging.debug("client.get_function() call failed: %s", mapping_error)

    func = getattr(client, function_name, None)
    if func:
        return func

    functions_attr = getattr(client, "functions", None)
    if isinstance(functions_attr, dict):
        available_names.extend(functions_attr.keys())
        if function_name in functions_attr:
            return functions_attr[function_name]
    elif callable(functions_attr):
        try:
            result = functions_attr()
            if isinstance(result, dict):
                available_names.extend(result.keys())
                if function_name in result:
                    return result[function_name]
        except Exception as call_error:
            logging.debug("client.functions() call failed: %s", call_error)

    available_names = sorted({name for name in available_names if isinstance(name, str)})
    raise ModalClientError(
        f"Aplikacja Modal '{app_name}' (env={environment}) nie ma funkcji '{function_name}'. "
        f"Dostępne nazwy według lookupu: {available_names or 'brak danych'}."
    )


def _call_modal_sync(
    audio_bytes: bytes,
    file_name: str,
    username: Optional[str],
) -> Dict[str, Any]:
    """
    Blokujący helper, który synchronizuje wywołanie funkcji Modal.
    """
    function_handle = _get_modal_function()
    if hasattr(function_handle, "call"):
        return function_handle.call(
            file_data=audio_bytes,
            file_name=file_name,
            username=username,
        )
    if hasattr(function_handle, "remote"):
        return function_handle.remote(
            file_data=audio_bytes,
            file_name=file_name,
            username=username,
        )
    if hasattr(function_handle, "spawn"):
        function_call = function_handle.spawn(
            file_data=audio_bytes,
            file_name=file_name,
            username=username,
        )
        if hasattr(function_call, "get"):
            return function_call.get()
    raise ModalClientError(
        "Uchwyt funkcji Modal nie udostępnia metod call/remote/spawn potrzebnych do wywołania."
    )


async def run_modal_diarization(
    audio_path: Path,
    username: Optional[str],
) -> Dict[str, Any]:
    """
    Wysyła plik audio do funkcji Modal i zwraca wynik diarizacji.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Plik audio nie istnieje: {audio_path}")

    logging.info("Wysyłanie pliku %s do diarizacji Modal...", audio_path.name)
    audio_bytes = await asyncio.to_thread(audio_path.read_bytes)

    result = await asyncio.to_thread(
        _call_modal_sync,
        audio_bytes,
        audio_path.name,
        username,
    )

    if not isinstance(result, dict):
        raise ModalClientError(
            f"Odpowiedź Modal nie jest słownikiem: {type(result)}"
        )

    return result

