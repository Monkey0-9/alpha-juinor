# infra/bus.py
import queue
import threading
import logging
from typing import Callable, Dict, List, Any
import time

logger = logging.getLogger(__name__)

class MessageBus:
    """
    Internal Event Bus for decoupling components.
    Simulates an institutional async architecture.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type}")

    def publish(self, event_type: str, data: Any):
        self._queue.put((event_type, data))

    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                event_type, data = self._queue.get(timeout=0.1)
                if event_type in self._subscribers:
                    for callback in self._subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in subscriber for {event_type}: {e}")
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop_event.set()
        self._thread.join()
