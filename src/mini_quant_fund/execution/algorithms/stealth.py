import random
from datetime import datetime, timedelta

class StealthAlgorithm:
    """Randomized slicing and venue switching to avoid detection"""
    
    def generate_randomized_schedule(self, total_qty: int):
        slices = []
        remaining = total_qty
        now = datetime.utcnow()
        
        while remaining > 0:
            slice_size = min(remaining, random.randint(10, 100))
            jitter = random.randint(5, 30)
            slices.append({
                "qty": slice_size,
                "time": now + timedelta(seconds=jitter)
            })
            remaining -= slice_size
            now += timedelta(seconds=jitter)
            
        return slices
