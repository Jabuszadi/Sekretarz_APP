# qdrant_uploader.py
import logging
from typing import List
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from qdrant_client.http.models import Distance, VectorParams

# Import config variables
from config import QDRANT_ENDPOINT, QDRANT_API_KEY
# Import getter functions for qdrant_client and embedding_model
from qdrant_handler import get_qdrant_client, get_embedding_model

# --- Initialize Embeddings ---
# Removed local initialization of embedding_model as it's now obtained via getter

# --- Qdrant Upload Function ---

async def upload_documents_to_qdrant(collection_name: str, documents: List[Document], batch_job_id: str):
    """Uploads a list of LangChain Documents to a specified Qdrant collection."""

    # Get the initialized embedding model
    current_embedding_model = get_embedding_model()
    current_qdrant_client = get_qdrant_client()

    if current_embedding_model is None:
         logging.error(f"❌ Cannot upload to Qdrant: Embedding model is None.")
         return False

    if not documents:
        logging.warning(f"⚠️ No documents provided for collection '{collection_name}'. Skipping upload.")
        return True

    if not QDRANT_ENDPOINT:
         logging.error(f"❌ Cannot upload to Qdrant: QDRANT_ENDPOINT is not set.")
         return False

    logging.info(f"⬆️ Uploading {len(documents)} documents to Qdrant collection: '{collection_name}' at {QDRANT_ENDPOINT}...")

    try:
        # Qdrant.from_documents handles collection creation if it doesn't exist
        qdrant_instance = Qdrant.from_documents(
            documents=documents,
            embedding=current_embedding_model,
            url=QDRANT_ENDPOINT,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
            prefer_grpc=True
        )
        logging.info(f"✅ Successfully uploaded documents to collection '{collection_name}'.")

        # Sprawdź, czy kolekcja istnieje, a jeśli nie, utwórz ją
        if not await current_qdrant_client.collection_exists(collection_name):
            await current_qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=current_embedding_model.embed_query("").shape[0], distance=Distance.COSINE),
            )
            print(f"Created new Qdrant collection: {collection_name}")

            # TUTAJ TWORZYMY INDEKS DLA POLA "date"
            try:
                await current_qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="date",
                    field_schema="keyword"
                )
                print(f"Created keyword index for 'date' field in collection {collection_name}")
            except Exception as e:
                print(f"Warning: Could not create index for 'date' field: {e}. It might already exist or there's another issue.")

        return collection_name

    except Exception as e:
        logging.error(f"❌ Failed to upload documents to Qdrant collection '{collection_name}': {e}")
        return False
