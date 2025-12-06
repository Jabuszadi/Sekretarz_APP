# modal_inspect.py
from typing import Any, Dict

from modal import App, Function
from modal.experimental import get_app_objects

APP_NAME = "sekretarz-diarization"
ENV_NAME = "main"  # jeśli używasz innego środowiska, podmień tutaj

def _lookup_app() -> App:
    if not ENV_NAME:
        return App.lookup(APP_NAME)

    for env_param in ("environment", "environment_name"):
        try:
            return App.lookup(APP_NAME, **{env_param: ENV_NAME})
        except TypeError:
            continue

    print(
        "App.lookup nie przyjmuje parametru environment/environment_name – próbuję bez niego."
    )
    return App.lookup(APP_NAME)


def _collect_function_handles(app: App) -> Dict[str, Any]:
    handles: Dict[str, Any] = {}
    for attr_name in sorted(dir(app)):
        if attr_name.startswith("_"):
            continue
        try:
            attr_value = getattr(app, attr_name)
        except AttributeError:
            # Niektóre atrybuty (np. image) wymagają lokalnego stanu – pomijamy je.
            continue
        except Exception:
            # Jeżeli inny atrybut zrzuca wyjątkek podczas pobierania, również go omijamy.
            continue
        call_method = getattr(attr_value, "call", None)
        if callable(call_method):
            handles[attr_name] = attr_value
    return handles


def main():
    try:
        app = _lookup_app()
        print(f"OK: App.lookup zwrócił uchwyt typu {type(app).__name__}.")
    except Exception as err:
        print(f"Błąd App.lookup('{APP_NAME}', env='{ENV_NAME}'): {err}")
        return

    handles = _collect_function_handles(app)

    try:
        experimental_objects = get_app_objects(APP_NAME, environment_name=ENV_NAME or None)
    except Exception as err:
        print(f"modal.experimental.get_app_objects nie powiodło się: {err}")
        experimental_objects = {}

    if experimental_objects:
        print(
            "Funkcje według modal.experimental.get_app_objects:",
            ", ".join(sorted(experimental_objects.keys())),
        )

    for name, obj in experimental_objects.items():
        if callable(getattr(obj, "call", None)):
            handles.setdefault(name, obj)

    try:
        function_from_name = Function.from_name(APP_NAME, "run_diarization", environment_name=ENV_NAME or None)
        handles["run_diarization"] = function_from_name
        print("Function.from_name znalazło run_diarization.")
    except Exception as err:
        print(f"Function.from_name('run_diarization') się nie powiodło: {err}")

    if handles:
        print("Dostępne funkcje:", ", ".join(sorted(handles.keys())))
    else:
        print("Nie znaleziono funkcji (uchwytów z metodą call).")

    run_handle = handles.get("run_diarization") or getattr(app, "run_diarization", None)
    if run_handle:
        print("OK: znaleziono uchwyt run_diarization.")
        available_methods = [
            method_name for method_name in ("call", "spawn", "remote") if hasattr(run_handle, method_name)
        ]
        print(f"Metody uchwytu: {available_methods}")
    else:
        print("Brak funkcji run_diarization w tym deploymentcie.")



if __name__ == "__main__":
    main()