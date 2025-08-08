import threading
import uvicorn
import time
import httpx
from pathlib import Path # Potrzebne dla ścieżek
import os # Dla zmiennych środowiskowych
import logging # Upewnij się, że to jest na górze pliku, jeśli go brakuje

# Funkcja sprawdzająca dostępność API
def wait_for_api(api_url, timeout=120): # Zwiększony timeout
    print("Czekam na uruchomienie API...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(api_url, timeout=5) # Mały timeout dla samego sprawdzania
            if r.status_code == 200:
                print("API jest dostępne!")
                return True
        except httpx.RequestError:
            pass
        except Exception as e:
            # print(f"Błąd podczas sprawdzania API: {e}") # Zbyt szczegółowe logowanie podczas czekania
            pass
        time.sleep(1)
    print("Nie udało się połączyć z API w zadanym czasie. Kończę.")
    return False

def run_api_thread():
    # reload=False jest krytyczne, gdy uruchamiamy w wątku
    uvicorn.run(
        "api_app:app", # Uruchamiamy aplikację z api_app.py
        host="0.0.0.0",
        port=7777,
        reload=False, # Ważne: False dla uruchamiania w wątku!
        log_level="info"
    )

def run_daemon_thread():
    from agent_daemon import main as daemon_main
    daemon_main()

if __name__ == "__main__":
    # Upewnij się, że logowanie jest skonfigurowane na DEBUG
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING) # Opcjonalnie: wycisz logi httpx, jeśli są zbyt głośne
    logging.getLogger('qdrant_client').setLevel(logging.WARNING) # Opcjonalnie: wycisz logi qdrant_client
    logging.getLogger('uvicorn').setLevel(logging.WARNING) # Wycisz logi uvicorn
    logging.getLogger('fastmcp').setLevel(logging.WARNING) # Wycisz logi fastmcp

    # --- Ustaw zmienną środowiskową dla PyTorch (jeśli potrzebne) ---
    # To pomoże z problemami z pamięcią CUDA
    # Przenieś to tutaj, aby było ustawione przed załadowaniem PyTorch/modeli
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Ustawiono PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # 1. Uruchamiamy API w osobnym wątku
    api_thread = threading.Thread(target=run_api_thread, daemon=True)
    api_thread.start()

    # 2. Główny wątek czeka na dostępność API
    if not wait_for_api("http://localhost:8000/health"): # Zmieniono port na 8000 i endpoint
        print("Nie udało się uruchomić API. Kończę działanie.")
        exit(1)

    # 3. Jeśli API jest gotowe, uruchamiamy daemona
    daemon_thread = threading.Thread(target=run_daemon_thread, daemon=True)
    daemon_thread.start()

    print("API i Daemon działają. Naciśnij CTRL+C, aby zakończyć.")

    try:
        while True:
            time.sleep(1) # Główny wątek śpi, pozwalając daemonowi i API działać
    except KeyboardInterrupt:
        print("\nZakończono działanie przez CTRL+C.")
    finally:
        # Możesz poczekać na zakończenie wątków, jeśli nie są daemon=True
        # Ale z daemon=True, zakończą się po zakończeniu głównego wątku.
        # Wypisane komunikaty z finally w daemon.main() powinny się pokazać.
        print("Zamykanie procesów...")