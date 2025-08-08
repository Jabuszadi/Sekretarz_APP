from datetime import datetime
from typing import List, Optional
import pandas as pd
from chunker import chunk_by_duration
from qdrant_uploader import upload_documents_to_qdrant
from langchain_core.documents import Document
from models import TranscriptionSegment
from utils import get_relevant_date_for_file
from pathlib import Path
from qdrant_handler import get_embedding_model
import logging
import agent_db

# Usunięto funkcję create_documents_from_chunks, ponieważ chunk_by_duration zwraca już Documenty.
# def create_documents_from_chunks(chunks_by_file: dict, variant_name: str) -> list[Document]:
#     """Create LangChain documents from chunks."""
#     all_documents = []
#     for original_file_path, chunks in chunks_by_file.items():
#         for chunk_data in chunks:
#             page_content = chunk_data.get("content", "")
#             metadata = {
#                 "chunk_id": chunk_data.get("chunk_id"),
#                 "file_path": chunk_data.get("file_path"),
#                 "date": chunk_data.get("date"),
#                 "speaker": chunk_data.get("speaker", "UNKNOWN"),
#                 "from_time": chunk_data.get("from_time"),
#                 "to_time": chunk_data.get("to_time"),
#                 "from_time_srt": chunk_data.get("from_time_srt"),
#                 "to_time_srt": chunk_data.get("to_time_srt"),
#                 "chunk_type": chunk_data.get("chunk_type", variant_name),
#                 "prechunk_id": chunk_data.get("prechunk_id"),
#                 "postchunk_id": chunk_data.get("postchunk_id"),
#                 "word_count": chunk_data.get("word_count", 0),
#             }
#             # Remove None values from metadata
#             metadata = {k: v for k, v in metadata.items() if v is not None}
#             all_documents.append(Document(page_content=page_content, metadata=metadata))
#     return all_documents


async def ingest_transcription(
    transcription_segments: List[TranscriptionSegment], # 1
    original_source_path: Path,                         # 2
    job_id: str,                                        # 3
    transcript_id: int,                                 # 4
    chunk_duration: int,                                # 5
    chunk_overlap: int                                  # 6
) -> str:
    """Process segments and upload to Qdrant."""
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "file_path": str(original_source_path),
            "date": get_relevant_date_for_file(original_source_path),
            "from_time": s.start,
            "to_time": s.end,
            "speaker": s.speaker,
            "content": s.text
        }
        for s in transcription_segments
    ])
    
    logging.debug(f"DataFrame date column values before chunking: {df['date'].tolist()}")
    
    # Chunk data
    # chunk_by_duration już zwraca List[Document], więc używamy tego bezpośrednio.
    documents = chunk_by_duration(df, duration=chunk_duration, overlap=chunk_overlap) # Użyj przekazanych duration i overlap

    # Zapis chunków do bazy
    if transcript_id is not None: # Upewnij się, że transcript_id nie jest None
        for doc in documents:
            meta = doc.metadata
            agent_db.add_transcript_chunk(
                transcript_id,
                meta.get("chunk_id"),
                meta.get("from_time"),
                meta.get("to_time"),
                meta.get("speaker"),
                meta.get("word_count"),
                meta.get("chunk_type"),
                meta.get("file_path"),
                meta.get("date")
            )
        logging.info(f"Added {len(documents)} chunks to transcript_chunks for transcript_id={transcript_id}") # Logowanie dodania chunków
    else:
        logging.warning("transcript_id is None, skipping chunk database insertion.")


    # Dodaj job_id do metadanych każdego istniejącego obiektu Document
    for doc in documents:
        doc.metadata["job_id"] = job_id
        logging.debug(f"Document metadata before upload - date: {doc.metadata.get('date')}, job_id: {doc.metadata.get('job_id')}")
    
    # Upload to Qdrant
    collection_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_source_path.stem}"

    current_embedding_model = get_embedding_model()

    try:
        collection_name_from_upload = await upload_documents_to_qdrant(collection_name, documents, job_id)
        if not collection_name_from_upload:
            logging.error(f"Failed to upload documents for job {job_id}.")
            return False
        return collection_name_from_upload
    except Exception as e:
        logging.error(f"❌ Failed to upload documents to Qdrant collection '{collection_name}': {e}")
        return False