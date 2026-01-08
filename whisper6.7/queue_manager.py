"""
Moduł do zarządzania kolejką wiadomości.
Zapewnia zapisywanie, przetwarzanie i retry dla wiadomości z API.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Callable

import agent_db

logger = logging.getLogger(__name__)


class MessageQueueManager:
    """Zarządca kolejki wiadomości."""
    
    def __init__(self, process_message_callback: Callable):
        """
        Inicjalizuje zarządcę kolejki.
        
        Args:
            process_message_callback: Funkcja async do przetwarzania wiadomości.
                                     Powinna przyjmować (query: str, collection_name: Optional[str], username: str)
                                     i zwracać str (odpowiedź) lub rzucać wyjątek.
        """
        self.process_message_callback = process_message_callback
        self.is_running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def add_message(
        self,
        username: str,
        query: str,
        collection_name: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 3,
    ) -> str:
        """
        Dodaje wiadomość do kolejki.
        
        Args:
            username: Nazwa użytkownika
            query: Treść zapytania
            collection_name: Opcjonalna nazwa kolekcji Qdrant
            priority: Priorytet wiadomości (wyższy = pierwszy)
            max_retries: Maksymalna liczba prób przetworzenia
            
        Returns:
            str: ID wiadomości
        """
        message_id = str(uuid.uuid4())
        agent_db.add_message_to_queue(
            message_id=message_id,
            username=username,
            query=query,
            collection_name=collection_name,
            priority=priority,
            max_retries=max_retries,
        )
        logger.info(f"Dodano wiadomość do kolejki: {message_id} (użytkownik: {username})")
        return message_id
    
    async def process_message(self, message: Dict[str, Any]) -> bool:
        """
        Przetwarza pojedynczą wiadomość.
        
        Args:
            message: Słownik z danymi wiadomości z bazy danych
            
        Returns:
            bool: True jeśli przetworzono pomyślnie, False w przeciwnym razie
        """
        message_id = message["message_id"]
        query = message["query"]
        collection_name = message.get("collection_name")
        username = message["username"]
        retry_count = message.get("retry_count", 0)
        max_retries = message.get("max_retries", 3)
        
        logger.info(f"Przetwarzanie wiadomości {message_id} (próba {retry_count + 1}/{max_retries})")
        
        # Oznacz jako przetwarzane
        agent_db.update_message_status(message_id, "processing")
        
        try:
            # Przetwórz wiadomość
            response = await self.process_message_callback(query, collection_name, username)
            
            # Oznacz jako zakończone
            agent_db.update_message_status(
                message_id,
                "completed",
                response=response,
            )
            logger.info(f"Wiadomość {message_id} przetworzona pomyślnie")
            return True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Błąd podczas przetwarzania wiadomości {message_id}: {error_message}", exc_info=True)
            
            # Zwiększ licznik prób
            if retry_count < max_retries - 1:
                # Oblicz opóźnienie przed następną próbą (exponential backoff)
                delay_seconds = min(60 * (2 ** retry_count), 3600)  # Max 1 godzina
                agent_db.increment_retry_count(message_id, delay_seconds)
                logger.info(
                    f"Wiadomość {message_id} zostanie ponowiona za {delay_seconds} sekund "
                    f"(próba {retry_count + 1}/{max_retries})"
                )
            else:
                # Przekroczono maksymalną liczbę prób
                agent_db.update_message_status(
                    message_id,
                    "failed",
                    error_message=error_message,
                )
                logger.error(f"Wiadomość {message_id} nieudana po {max_retries} próbach")
            
            return False
    
    async def worker_loop(self, batch_size: int = 5, poll_interval: float = 2.0):
        """
        Główna pętla workera przetwarzającego kolejkę.
        
        Args:
            batch_size: Liczba wiadomości do przetworzenia jednocześnie
            poll_interval: Czas oczekiwania między sprawdzaniem kolejki (w sekundach)
        """
        logger.info("Worker kolejki wiadomości uruchomiony")
        self.is_running = True
        
        while self.is_running:
            try:
                # Pobierz wiadomości oczekujące
                pending_messages = agent_db.get_pending_messages(limit=batch_size)
                
                if pending_messages:
                    logger.info(f"Znaleziono {len(pending_messages)} wiadomości do przetworzenia")
                    
                    # Przetwórz wiadomości równolegle
                    tasks = [self.process_message(msg) for msg in pending_messages]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Brak wiadomości, poczekaj przed następnym sprawdzeniem
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Błąd w worker loop: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)
        
        logger.info("Worker kolejki wiadomości zatrzymany")
    
    def start_worker(self, batch_size: int = 5, poll_interval: float = 2.0):
        """
        Uruchamia worker w tle.
        
        Args:
            batch_size: Liczba wiadomości do przetworzenia jednocześnie
            poll_interval: Czas oczekiwania między sprawdzaniem kolejki (w sekundach)
        """
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self.worker_loop(batch_size, poll_interval)
            )
            logger.info("Worker kolejki wiadomości uruchomiony w tle")
    
    def stop_worker(self):
        """Zatrzymuje worker."""
        self.is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            logger.info("Worker kolejki wiadomości zatrzymany")
    
    async def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Pobiera status wiadomości.
        
        Args:
            message_id: ID wiadomości
            
        Returns:
            Dict z danymi wiadomości lub None jeśli nie znaleziono
        """
        return agent_db.get_message_by_id(message_id)
    
    def cleanup_old_messages(self, days: int = 7) -> int:
        """
        Usuwa stare zakończone wiadomości.
        
        Args:
            days: Liczba dni po których wiadomości są usuwane
            
        Returns:
            Liczba usuniętych wiadomości
        """
        deleted_count = agent_db.delete_old_completed_messages(days)
        logger.info(f"Usunięto {deleted_count} starych wiadomości")
        return deleted_count

