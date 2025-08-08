import os
import sys
import logging

# Dodaj katalog nadrzędny (główny katalog projektu) do sys.path
# To pozwala na importowanie modułów z głównego katalogu, takich jak agent_db
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.insert(0, parent_dir)

import agent_db # Teraz to powinno działać
# import qdrant_handler # Potrzebne, jeśli testujesz semantyczne wyszukiwanie Qdrant

# Konfiguracja logowania, aby widzieć komunikaty debugowania z agent_db
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test_search():
    logging.info("Starting test for search_transcripts_semantic...")

    # Upewnij się, że baza danych jest zainicjalizowana
    try:
        agent_db.init_db()
        logging.info("Database initialized successfully for test.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        return

    # Przykład użycia search_transcripts_semantic z zapytaniem, o którym wiemy, że powinno znaleźć dane
    query_to_test = "cross-validation"
    logging.info(f"Attempting to search for: '{query_to_test}'")

    try:
        results = agent_db.search_transcripts_semantic(query_to_test, limit=5)
        
        if results:
            logging.info(f"Search for '{query_to_test}' returned {len(results)} results:")
            for i, res in enumerate(results):
                logging.info(f"--- Result {i+1} ---")
                # Sprawdź, czy klucze istnieją przed próbą ich użycia
                logging.info(f"Transcript ID: {res.get('transcript_id', 'N/A')}")
                # logging.info(f"Content (snippet): {res.get('content', 'N/A')[:200]}...") # Zakomentowane, bo search_transcripts_semantic może nie zwracać całego contentu
                logging.info(f"Summary Text: {res.get('summary_text', 'N/A')[:200]}...") # Dodano
                logging.info(f"Similarity Score: {res.get('similarity_score', 'N/A')}")
                logging.info(f"Occurrences: {res.get('occurrences', 'N/A')}")
        else:
            logging.info(f"Search for '{query_to_test}' returned no results.")
            # Dodajmy zapytania debugujące, jeśli nie ma wyników
            logging.info("--- Debugging database content for search ---")
            

            total_transcripts = agent_db.execute_read_query("SELECT COUNT(*) FROM transcripts")[0]['COUNT(*)'] # ZMIENIONO: execute_sql_query na execute_read_query
            logging.info(f"Total transcripts in DB: {total_transcripts}")
            
            total_segments = agent_db.execute_read_query("SELECT COUNT(*) FROM transcript_segments")[0]['COUNT(*)'] # ZMIENIONO: execute_sql_query na execute_read_query
            logging.info(f"Total transcript segments in DB: {total_segments}")
            
            # Spróbuj pobrać jakiś content ręcznie
            sample_content = agent_db.execute_read_query("SELECT content FROM transcripts LIMIT 1") # ZMIENIONO: execute_sql_query na execute_read_query

            if sample_content:
                logging.info(f"Sample transcript content: {sample_content[0].get('content', 'N/A')[:200]}...")
            
    except Exception as e:
        logging.error(f"An error occurred during search_transcripts_semantic test: {e}", exc_info=True)

if __name__ == "__main__":
    # Ustaw KMP_DUPLICATE_LIB_OK=TRUE również dla tego skryptu, aby uniknąć błędów OMP
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    run_test_search()