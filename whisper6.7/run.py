import threading
import uvicorn
import time
import httpx
from pathlib import Path # Potrzebne dla ścieżek
import os # Dla zmiennych środowiskowych
import logging # Upewnij się, że to jest na górze pliku, jeśli go brakuje
import requests # Dodane dla wait_for_api_and_daemon
import subprocess # Dodane dla uruchamiania API i Daemon
import sys # Dodane dla sys.exit
import platform # Dodane dla platform.system()

# Funkcja sprawdzająca dostępność API
def wait_for_api(api_url, timeout=999): # Zwiększony timeout
    logging.info("Czekam na uruchomienie API...") # Zmieniono na logging
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(api_url, timeout=5) # Mały timeout dla samego sprawdzania
            if r.status_code == 200:
                logging.info("API jest dostępne!") # Zmieniono na logging
                return True
        except httpx.RequestError:
            pass
        except Exception as e:
            pass
        time.sleep(1)
    logging.error("Nie udało się połączyć z API w zadanym czasie. Kończę.") # Zmieniono na logging
    return False

# Usunięto funkcje run_api_thread i run_daemon_thread - nie będą używane z Popen.

def wait_for_api_and_daemon(api_url, daemon_url, timeout=60):
    start_time = time.time()
    logging.info("Czekam na uruchomienie API...")
    # print("Czekam na uruchomienie API...") # USUNIĘTO
    while time.time() - start_time < timeout:
        try:
            # Sprawdź dostępność API
            api_response = requests.get(f"{api_url}/health")
            if api_response.status_code == 200:
                logging.info("API jest dostępne!")
                # print("API jest dostępne!") # USUNIĘTO
                return True
        except requests.exceptions.ConnectionError as e:
            # logging.debug(f"Błąd podczas sprawdzania API: {e}") # Zbyt szczegółowe logowanie podczas czekania
            # print(f"Błąd podczas sprawdzania API: {e}") # Zbyt szczegółowe logowanie podczas czekania # USUNIĘTO
            pass # Ignoruj błędy połączenia podczas oczekiwania
        time.sleep(1)
    logging.error("Nie udało się połączyć z API w zadanym czasie. Kończę.")
    # print("Nie udało się połączyć z API w zadanym czasie. Kończę.") # USUNIĘTO
    return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Przywrócono INFO
    logging.getLogger('httpx').setLevel(logging.WARNING) 
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO) # Zmieniono na INFO
    logging.getLogger('fastmcp').setLevel(logging.WARNING) 

    if platform.system() == "Windows":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logging.info("Ustawiono PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    API_URL = "http://localhost:7777"
    DAEMON_SCRIPT_PATH = Path(__file__).parent / "agent_daemon.py"

    api_process = None # Inicjalizacja
    daemon_process = None # Inicjalizacja

    try:
        logging.info("Uruchamiam API... (api_app.py)")
        # Uruchamiamy API jako subprocess
        api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "7777", "--log-level", "info"], env=os.environ.copy())
        
        if not wait_for_api(API_URL + "/health"): # Czekaj na endpoint /health
            raise Exception("API nie uruchomiło się poprawnie.")

        logging.info("Uruchamiam Daemona... (agent_daemon.py)")
        # Uruchamiamy daemona jako subprocess
        daemon_process = subprocess.Popen([sys.executable, str(DAEMON_SCRIPT_PATH)], env=os.environ.copy())
        
        # Nie ma bezpośredniego /health dla daemona, więc czekamy na jego start logiem
        # Możesz tutaj dodać bardziej złożoną logikę sprawdzania daemona, jeśli potrzebujesz
        logging.info("Daemon został uruchomiony.")

        logging.info("API i Daemon działają. Naciśnij CTRL+C, aby zakończyć.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\nZakończono działanie przez CTRL+C.")
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas uruchamiania procesów: {e}", exc_info=True)
    finally:
        logging.info("Zamykanie procesów...")
        if api_process and api_process.poll() is None:
            api_process.terminate()
            api_process.wait(timeout=5) # Czekaj do 5 sekund na zakończenie
        if daemon_process and daemon_process.poll() is None:
            daemon_process.terminate()
            daemon_process.wait(timeout=5) # Czekaj do 5 sekund na zakończenie
        logging.info("Wszystkie procesy zakończone.")