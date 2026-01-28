# TODO - Pylance Error Fixes

## Issues to Fix
- [x] main.py:510 - `generate_signals` not known attribute of None
- [x] main.py:519 - `allocate` not known attribute of None
- [x] portfolio/allocator.py - OrderList missing `__len__` method
- [x] pyproject.toml - Add cSpell settings for unknown words

## Completed Actions

### Fix 1: Add None check for self.strategy.generate_signals()
- File: main.py
- Fix: Added defensive None check before calling generate_signals

### Fix 2: Add None check for self.allocator.allocate()
- File: main.py
- Fix: Added defensive None check before calling allocate

### Fix 3: Add __len__ method to OrderList
- File: portfolio/allocator.py
- Fix: Added `__len__(self)` method that returns `len(self.orders)`

### Fix 4: Add cSpell settings for unknown words
- File: pyproject.toml
- Words added: dotenv, PRECHECK, signum, OHLCV, lookback, Pylance, Praveen

