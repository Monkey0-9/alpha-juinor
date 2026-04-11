
import redis
import json
import logging
import uuid
import os
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger("TASK_QUEUE")

class TaskQueue:
    """
    Distributed Task Queue using Redis.
    Enables horizontal scaling of signal processing and execution.
    """
    def __init__(self, queue_name: str = "quant_tasks"):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.r = redis.from_url(self.redis_url)
        self.queue_name = queue_name
        logger.info(f"TaskQueue initialized on {self.redis_url}")

    def enqueue(self, task_type: str, data: Dict[str, Any]):
        """Push a task to the queue."""
        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "data": data,
            "timestamp": os.getpid() # Simulated PID for tracking
        }
        self.r.lpush(self.queue_name, json.dumps(task))
        logger.debug(f"Enqueued task {task['task_id']} ({task_type})")

    def dequeue(self, timeout: int = 5) -> Optional[Dict]:
        """Pull a task from the queue."""
        result = self.r.brpop(self.queue_name, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

def get_task_queue() -> TaskQueue:
    return TaskQueue()
