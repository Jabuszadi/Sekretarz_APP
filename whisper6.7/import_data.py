import sqlite3
import csv
from pathlib import Path
import logging

DB_PATH = "agent_files.db"
CSV_PATH = "processed_files.csv"

def import_processed_files():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Upewnij się, że tabela istnieje
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT,
                filehash TEXT,
                status TEXT,
                api_response TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Sprawdź, czy tabela jest już pusta
        cursor.execute("SELECT COUNT(*) FROM processed_files")
        if cursor.fetchone()[0] > 0:
            logging.info("Tabela processed_files nie jest pusta. Usuwam istniejące dane...")
            cursor.execute("DELETE FROM processed_files")
            conn.commit()

        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # SQLite automatycznie obsługuje wartości NULL dla kolumn nieujętych w INSERT
                cursor.execute("""
                    INSERT INTO processed_files (filename, filepath, filehash, status, api_response, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row['filename'],
                    row['filepath'],
                    row['filehash'],
                    row['status'],
                    row['api_response'] if 'api_response' in row else None, # Uwzględnij, jeśli kolumna istnieje
                    row['processed_at'] # Upewnij się, że format daty jest zgodny z TIMESTAMP
                ))
            conn.commit()
        logging.info(f"Pomyślnie zaimportowano dane z {CSV_PATH} do {DB_PATH}")

    except sqlite3.Error as e:
        logging.error(f"Błąd SQLite: {e}")
    except FileNotFoundError:
        logging.error(f"Błąd: Plik {CSV_PATH} nie został znaleziony.")
    except Exception as e:
        logging.error(f"Wystąpił nieoczekiwany błąd: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    import_processed_files()
