import asyncio
import logging
import sys
import os

# Dodaj katalog główny projektu do ścieżki systemowej
# Zakładamy, że ten skrypt znajduje się w podkatalogu 'scripts'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

import config # Teraz import config powinien działać

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import CollectionInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def initialize_qdrant_client():
    """Initializes and returns an AsyncQdrantClient instance."""
    logging.info("Initializing Qdrant client...")
    try:
        qdrant_client = AsyncQdrantClient(
            url=config.QDRANT_ENDPOINT,
            api_key=config.QDRANT_API_KEY,
            timeout=30
        )
        await qdrant_client.get_collections() # Test connection
        logging.info("✅ Successfully initialized and connected to Qdrant client.")
        return qdrant_client
    except Exception as e:
        logging.error(f"❌ Failed to initialize Qdrant client: {e}")
        return None

async def delete_all_qdrant_collections():
    """
    Connects to Qdrant, lists all collections, and offers to delete them after confirmation.
    """
    qdrant_client = await initialize_qdrant_client()
    if not qdrant_client:
        logging.error("Failed to get Qdrant client. Exiting.")
        return

    try:
        collections_response = await qdrant_client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if not collection_names:
            logging.info("No Qdrant collections found to delete.")
            return

        logging.info(f"Found {len(collection_names)} Qdrant collections:")
        for name in collection_names:
            print(f"- {name}")

        confirmation = input("Are you sure you want to delete ALL of these collections? (yes/no): ").lower()
        if confirmation == 'yes':
            for collection_name in collection_names:
                logging.info(f"Attempting to delete collection: '{collection_name}'...")
                try:
                    await qdrant_client.delete_collection(collection_name=collection_name)
                    logging.info(f"✅ Collection '{collection_name}' deleted successfully.")
                except Exception as e:
                    logging.error(f"❌ Failed to delete collection '{collection_name}': {e}")
            logging.info("Finished attempting to delete all specified Qdrant collections.")
        else:
            logging.info("Collection deletion cancelled by user.")

    except Exception as e:
        logging.error(f"❌ Error listing or deleting Qdrant collections: {e}")
    finally:
        # Close the client connection if necessary (AsyncQdrantClient manages connections)
        pass # Client handles its own lifecycle in this context

if __name__ == "__main__":
    asyncio.run(delete_all_qdrant_collections())