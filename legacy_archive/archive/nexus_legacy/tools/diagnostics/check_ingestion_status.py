import sqlite3
import json

conn = sqlite3.connect('runtime/institutional_trading.db')

# Check total symbols
c = conn.execute('SELECT count(DISTINCT symbol) FROM price_history')
total_symbols = c.fetchone()[0]
print(f'Total symbols with data: {total_symbols}')

# Check symbols with 1260+ rows
c2 = conn.execute('SELECT symbol, count(*) as cnt FROM price_history GROUP BY symbol HAVING cnt >= 1260 ORDER BY cnt DESC')
valid = c2.fetchall()
print(f'Symbols with 1260+ rows (governance compliant): {len(valid)}')

# Check symbols with less than 1260
c3 = conn.execute('SELECT symbol, count(*) as cnt FROM price_history GROUP BY symbol HAVING cnt < 1260 ORDER BY cnt DESC')
invalid = c3.fetchall()
print(f'Symbols with <1260 rows: {len(invalid)}')

# Show ingestion audit summary
c4 = conn.execute('SELECT status, count(*) FROM ingestion_audit GROUP BY status')
audit = c4.fetchall()
print('\nIngestion Audit Summary:')
for status, cnt in audit:
    print(f'  {status}: {cnt}')

# Check latest ingestion run
c5 = conn.execute('SELECT run_id, total_symbols, successful, rejected, failed FROM ingestion_audit_runs ORDER BY run_id DESC LIMIT 1')
row = c5.fetchone()
if row:
    print(f'\nLatest Ingestion Run: {row[0]}')
    print(f'  Total: {row[1]}, Success: {row[2]}, Rejected: {row[3]}, Failed: {row[4]}')

conn.close()
