
from datetime import datetime

class GannTimeFilter:
    """
    Gann Time Cycle Filter (SAFE/SIMPLIFIED).
    
    Rule:
    - Blocks trading during specific 'Turn Times' or high-volatility windows 
      where 'Square of Time' effects might cause reversal chop.
    - Example: First 5 mins and Last 5 mins of hourly/session blocks.
    """
    
    def can_trade(self, timestamp: datetime) -> bool:
        try:
            # Simple Time Window Filter logic
            # E.g. Avoid minute 0-2 and 58-59 of the hour (Gann 'change of trend' times)
            minute = timestamp.minute
            
            # Block the 'Turn of the Hour' (often volatile/noisy)
            if minute < 2 or minute > 58:
                return False
                
            return True
        except Exception:
            return True
