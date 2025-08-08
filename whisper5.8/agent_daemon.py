# agent_daemon.py
import time
import httpx
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import threading
import queue
import os
from agent_db import (
    init_db,
    add_file,
    file_already_processed,
    update_file_status,
    get_file_status,
    get_files_with_error,
    compute_file_hash
)
import logging

API_URL = "http://localhost:8000/process_file/"
WATCH_DIR = "input_watch"  # Folder do monitorowania

file_queue = queue.Queue()

def wait_for_api(api_url, timeout=60):
    print("Czekam na uruchomienie API...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(api_url)
            if r.status_code == 200:
                print("API jest dostępne!")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Nie udało się połączyć z API w zadanym czasie.")
    return False

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in {".wav", ".mp3", ".mkv", ".mp4", ".flac", ".ogg", ".m4a"}:
            filehash = None
            # Dodaj logikę ponawiania prób obliczenia hashu w przypadku PermissionError
            for i in range(5): # Spróbuj 5 razy z opóźnieniem
                try:
                    filehash = compute_file_hash(file_path)
                    break # Hash obliczony, wychodzimy z pętli
                except PermissionError:
                    print(f"Błąd dostępu do pliku {file_path.name}, ponawiam próbę w 1 sekundę... (Próba {i+1}/5)")
                    time.sleep(1)
                except Exception as e:
                    print(f"Nieoczekiwany błąd podczas obliczania hashu dla {file_path.name}: {e}")
                    return # Inny błąd, przerywamy

            if filehash is None:
                print(f"Nie udało się obliczyć hashu dla {file_path.name} po wielu próbach, pomijam.")
                return

            if get_file_status(filehash) == "success": # Sprawdź, czy już był sukces
                print(f"Plik {file_path.name} już był przetwarzany poprawnie, pomijam dodawanie do kolejki.")
                return

            print(f"Nowy plik wykryty: {file_path}, dodaję do kolejki.")
            file_queue.put((file_path, False)) # Dodaj do kolejki, is_retry=False
            # Jeśli pliku nie ma w bazie, dodaj go jako 'pending'
            if get_file_status(filehash) is None:
                add_file(file_path.name, str(file_path), filehash, "pending", "")

    def process_file(self, file_path: Path, is_retry: bool = False):
        filehash = compute_file_hash(file_path)
        current_status = get_file_status(filehash)

        if current_status == "success":
            print(f"Plik {file_path.name} już był przetwarzany poprawnie, pomijam.")
            return

        # Jeśli plik jest już w bazie z błędem i to nie jest próba retry, pomijamy na razie
        if current_status == "error" and not is_retry:
            print(f"Plik {file_path.name} miał błąd (status 'error'), będzie retry później.")
            return

        # Jeśli pliku nie ma w bazie, dodaj go ze statusem "pending"
        if current_status is None:
            add_file(file_path.name, str(file_path), filehash, "pending", "")

        # ... (logika oczekiwania na rozmiar pliku) ...

        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                # Ustaw timeout dla żądania HTTP
                # Zwiększ timeout dla bardzo długich operacji, np. do 30 minut (1800 sekund) lub więcej
                response = httpx.post(API_URL, files=files, timeout=1800) 

            if response.status_code == 200:
                update_file_status(filehash, "success", response.text)
                logging.info(f"Plik {file_path.name} przetworzony OK. Zaktualizowano status w bazie.")
                # Opcjonalnie: usuń plik z folderu input_watch po sukcesie
                try:
                    os.remove(file_path)
                    logging.info(f"Usunięto przetworzony plik z input_watch: {file_path}")
                except OSError as e:
                    logging.error(f"Błąd podczas usuwania pliku {file_path}: {e}")
            else:
                # This block will now be hit if minutes_service.py failed to save
                update_file_status(filehash, "error", f"API Error: {response.status_code} {response.text}")
                logging.error(f"Plik {file_path.name} błąd API: {response.status_code} {response.text}")
        except httpx.RequestError as e: # Catch specific httpx errors
            update_file_status(filehash, "error", f"Connection Error: {e}")
            logging.error(f"Błąd połączenia podczas wysyłania pliku {file_path}: {e}")
        except Exception as e:
            update_file_status(filehash, "error", f"Unexpected Error: {str(e)}")
            logging.error(f"Nieoczekiwany błąd podczas wysyłania pliku {file_path}: {e}")

def process_existing_files(watch_dir):
    print(f"Przetwarzam istniejące pliki w {watch_dir}...")
    for file_path in Path(watch_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {".wav", ".mp3", ".mkv", ".mp4", ".flac", ".ogg", ".m4a"}:
            filehash = compute_file_hash(file_path) # Oblicz hash
            current_status = get_file_status(filehash)

            if current_status == "success":
                print(f"Plik {file_path.name} już był przetwarzany poprawnie, pomijam.")
                continue
            if current_status == "error":
                print(f"Plik {file_path.name} miał błąd (status 'error'), będzie retry później.")
                continue # Nie dodawaj go teraz, zostanie dodany przez retry_errors

            # Jeśli pliku nie ma w bazie lub ma inny status (np. 'pending' z poprzedniej sesji)
            if current_status is None:
                add_file(file_path.name, str(file_path), filehash, "pending", "")
            print(f"Znaleziono istniejący plik: {file_path}, dodaję do kolejki.")
            file_queue.put((file_path, False)) # Dodaj do kolejki (Path, is_retry=False)

def retry_errors(): # Handler już nie jest potrzebny, kolejka jest globalna
    print("Sprawdzam pliki z błędem do ponownego przetworzenia...")
    error_files_db_records = get_files_with_error() # Zwraca (filename, filepath)
    if not error_files_db_records:
        print("Brak plików z błędem do ponownego przetworzenia.")
        return
    print(f"Retry dla {len(error_files_db_records)} plików z błędem...")
    for filename, filepath_str in error_files_db_records:
        file_path = Path(filepath_str)
        if file_path.exists():
            # Sprawdź, czy plik nie jest już w kolejce jako 'pending' lub 'error'
            filehash = compute_file_hash(file_path)
            current_status = get_file_status(filehash)
            if current_status == "error": # Upewnij się, że to nadal jest błąd i że plik istnieje
                print(f"Ponowna próba przetworzenia: {file_path}, dodaję do kolejki.")
                file_queue.put((file_path, True)) # Dodaj do kolejki (Path, is_retry=True)
            else:
                print(f"Plik {file_path} ma status '{current_status}', nie wymaga retry.")
        else:
            print(f"Plik {file_path} z błędem już nie istnieje na dysku – pomijam.")
            # Opcjonalnie: usunąć go z bazy, jeśli nie istnieje

def worker_process_file():
    while True:
        file_path_item = file_queue.get()
        if file_path_item is None:
            break

        file_path, is_retry = file_path_item
        print(f"\nWorker przetwarza plik: {file_path} (Retry: {is_retry})")

        # --- Logika oczekiwania na stabilność pliku (przeniesiona z NewFileHandler) ---
        # TO JEST BARDZO WAŻNE, ŻEBY TO BYŁO TUTAJ, PRZED PRÓBĄ OTWARCIA
        last_size = -1
        stable_count = 0
        timeout_wait_file = 30 # Czas oczekiwania na stabilność pliku
        start_wait = time.time()
        while stable_count < 3 and (time.time() - start_wait < timeout_wait_file):
            try:
                current_size = file_path.stat().st_size
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = current_size
            except FileNotFoundError:
                print(f"Plik {file_path} zniknął podczas oczekiwania na stabilność.")
                file_queue.task_done()
                return
            except Exception as e:
                print(f"Błąd podczas sprawdzania rozmiaru pliku {file_path}: {e}")
                time.sleep(1)
                continue
            time.sleep(1)

        if stable_count < 3:
            print(f"Plik {file_path} nie osiągnął stabilnego rozmiaru, pomijam tę próbę.")
            update_file_status(compute_file_hash(file_path), "error", "File not stable within timeout")
            file_queue.task_done()
            continue

        # --- Dalej logika wysyłania do API (wewnątrz workera) ---
        filehash = compute_file_hash(file_path)
        current_status = get_file_status(filehash)

        # Użyj metody process_file z NewFileHandler
        handler = NewFileHandler() # Tworzymy instancję NewFileHandler w workerze
        handler.process_file(file_path, is_retry)
        
        # Usunięto warunkowe usuwanie pliku z tego miejsca.
        # Logika usuwania pliku jest teraz zawarta TYLKO w NewFileHandler.process_file
        # i zależy od response.status_code == 200.
        
        file_queue.task_done()

def main():
    init_db() # Inicjalizacja bazy

    # Start workera przetwarzającego kolejkę
    worker_thread = threading.Thread(target=worker_process_file, daemon=True)
    worker_thread.start()

    # Utwórz folder do monitorowania, jeśli nie istnieje
    Path(WATCH_DIR).mkdir(exist_ok=True)

    # Dodaj istniejące pliki do kolejki
    process_existing_files(WATCH_DIR)

    # Dodaj pliki z błędem do kolejki (po przetworzeniu istniejących)
    retry_errors()

    # Start monitorowania nowych plików
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Monitoring folderu: {WATCH_DIR}")

    try:
        while True:
            time.sleep(1) # Główny wątek śpi
    except KeyboardInterrupt:
        print("\nDaemon zakończony przez użytkownika.")
    finally:
        observer.stop()
        observer.join()
        file_queue.put(None) # Sygnał do workera, żeby się zakończył
        worker_thread.join()
        print("Wszystkie procesy daemona zakończone.")

if __name__ == "__main__":
    # Najpierw czekaj na API
    if not wait_for_api("http://localhost:8000/"):
        print("Daemon kończy pracę – brak API.")
        exit(1)
    # Dalej normalny start daemona:
    main()