# qdrant_handler.py
import logging
from datetime import datetime
from qdrant_client import QdrantClient, AsyncQdrantClient
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import Optional, List
import config
from qdrant_client.http.models import CollectionInfo
from pydantic import BaseModel, Field # Dodano import BaseModel i Field

# Import configuration variables needed here
from config import QDRANT_ENDPOINT, QDRANT_API_KEY, EMBEDDING_MODEL_NAME, DEVICE

qdrant_client: Optional[AsyncQdrantClient] = None
embedding_model: Optional[SentenceTransformerEmbeddings] = None

async def initialize_qdrant_resources():
    global qdrant_client, embedding_model

    if qdrant_client is None:
        logging.info("Initializing Qdrant client...")
        try:
            qdrant_client = AsyncQdrantClient(
                url=config.QDRANT_ENDPOINT,
                api_key=config.QDRANT_API_KEY,
                timeout=30
            )
            await qdrant_client.get_collections()
            logging.info("✅ Successfully initialized and connected to Qdrant client.")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Qdrant client: {e}")
            qdrant_client = None
            raise

    if embedding_model is None:
        logging.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        try:
            embedding_model = SentenceTransformerEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": config.DEVICE}
            )
            _ = embedding_model.embed_query("test query")
            logging.info(f"✅ Successfully loaded embedding model: {config.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model {config.EMBEDDING_MODEL_NAME}: {e}")
            embedding_model = None
            raise

def get_qdrant_client() -> AsyncQdrantClient:
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized. Call initialize_qdrant_resources first.")
    return qdrant_client

def get_embedding_model() -> SentenceTransformerEmbeddings:
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized. Call initialize_qdrant_resources first.")
    return embedding_model

# --- Document Search Function (Original, more specific) ---
def search_documents(query: str, collection_name: str, target_date: str) -> List[str]:
    """Searches for documents in a specific Qdrant collection matching the target date."""
    try:
        datetime.strptime(target_date, "%d-%m-%Y")
    except ValueError:
        logging.warning(f"⚠️ Invalid date format: {target_date}. Please use DD-MM-YYYY. Skipping search for collection '{collection_name}'.")
        return []

    try:
        query_vector = embedding_model.embed_query(query)

        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=50,
            with_payload=True
        )

        documents = []
        for point in results.points:
            if hasattr(point, "payload") and point.payload is not None:
                payload = point.payload
                content = payload.get("page_content") or payload.get("content")
                if content:
                    point_date = payload.get("date")
                    if point_date == target_date:
                        documents.append(content)
                        if len(documents) >= 10:
                            break
                else:
                    logging.warning(f"⚠️ Payload found, but no 'page_content' or 'content' key in payload for point ID {point.id} in collection {collection_name}")
            else:
                logging.warning(f"⚠️ Unexpected point structure or missing payload in collection {collection_name}: {point}")

        return documents

    except Exception as e:
        logging.error(f"❌ Error searching documents in collection '{collection_name}' for date '{target_date}': {e}")
        return []

# NOWE: Model Pydantic dla wyników wyszukiwania Qdrant
class QdrantSearchResult(BaseModel):
    content: str = Field(..., description="Treść znalezionego fragmentu dokumentu.")
    file_path: str = Field(..., description="Ścieżka do oryginalnego pliku, z którego pochodzi fragment.")
    # Możesz dodać inne pola, jeśli są w payloadzie Qdrant i są potrzebne, np. 'speaker', 'start_time'

# NOWA FUNKCJA: Wyszukiwanie we wszystkich kolekcjach (dodana dla agenta czatowego)
async def search_all_collections(query: str, limit_per_collection: int = 5, total_limit: int = 20, target_collection: Optional[str] = None) -> List[QdrantSearchResult]: # Zmieniono typ zwracany
    """
    Searches for documents across all available Qdrant collections or a specific one.

    Args:
        query (str): The search query.
        limit_per_collection (int): Maximum number of results to retrieve from each collection.
        total_limit (int): Maximum total number of documents to return.
        target_collection (Optional[str]): If provided, search only this specific collection.

    Returns:
        List[QdrantSearchResult]: A list of relevant document contents with file paths.
    """
    if target_collection:
        logging.info(f"Searching in specific Qdrant collection '{target_collection}' for query: '{query}'")
    else:
        logging.info(f"Searching across all Qdrant collections for query: '{query}'")

    if qdrant_client is None or embedding_model is None:
        logging.error("Qdrant client or embedding model not initialized. Cannot perform search.")
        return []

    try:
        query_vector = embedding_model.embed_query(query)
        all_relevant_documents: List[QdrantSearchResult] = [] # Zmieniono na listę obiektów QdrantSearchResult

        if target_collection:
            collection_names = [target_collection]
        else:
            collections_response = await qdrant_client.get_collections()
            collection_names = [c.name for c in collections_response.collections]

        logging.info(f"Found {len(collection_names)} Qdrant collections to search: {collection_names}")

        for collection_name in collection_names:
            if len(all_relevant_documents) >= total_limit:
                break

            try:
                results = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit_per_collection,
                    with_payload=True
                )

                for point in results.points:
                    if hasattr(point, "payload") and point.payload is not None:
                        payload = point.payload
                        logging.debug(f"DEBUG: Qdrant point payload: {payload}") # NOWE: Loguj pełny payload
                        content = payload.get("page_content") or payload.get("content")
                        file_path = payload.get("file_path") # Pobierz file_path z payloadu
                        if content and file_path: # Upewnij się, że oba są dostępne
                            all_relevant_documents.append(QdrantSearchResult(content=content, file_path=file_path))
                            if len(all_relevant_documents) >= total_limit:
                                break
                        else:
                            logging.warning(f"⚠️ Payload found, but missing 'page_content'/'content' or 'file_path' in payload for point ID {point.id} in collection {collection_name}. Full Payload: {payload}") # ZMIENIONO: Dodano pełny payload do ostrzeżenia
                    else:
                        logging.warning(f"⚠️ Unexpected point structure or missing payload in collection {collection_name}: {point}")
            except Exception as e_col:
                logging.error(f"❌ Error searching in collection '{collection_name}': {e_col}")

        logging.info(f"Found {len(all_relevant_documents)} relevant documents across collections.")
        return all_relevant_documents # Zwracamy listę obiektów QdrantSearchResult

    except Exception as e:
        logging.error(f"❌ General error during search across collections: {e}")
        return []

# NOWA FUNKCJA: Pobieranie nazw wszystkich kolekcji Qdrant
async def get_all_collection_names(limit: Optional[int] = None) -> List[str]: # Dodano 'limit'
    """
    Retrieves the names of all collections in Qdrant, optionally limited by count.

    Args:
        limit (Optional[int]): The maximum number of collection names to return.

    Returns:
        List[str]: A list of collection names.
    """
    if qdrant_client is None:
        logging.error("Qdrant client not initialized. Cannot retrieve collection names.")
        return []
    try:
        collections_response = await qdrant_client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if limit is not None and limit > 0:
            collection_names = collection_names[:limit] # Ograniczenie liczby kolekcji
            logging.info(f"Retrieved {len(collection_names)} Qdrant collection names (limited to {limit}).")
        else:
            logging.info(f"Retrieved {len(collection_names)} Qdrant collection names.")

        return collection_names
    except Exception as e:
        logging.error(f"❌ Error retrieving Qdrant collection names: {e}")
        return []
