import sqlite3

conn = sqlite3.connect('runtime/institutional_trading.db')
c = conn.execute('SELECT symbol, status, reason_code, error_message FROM ingestion_audit ORDER BY finished_at DESC LIMIT 10')
rows = c.fetchall()

print('Recent Ingestion Audit Records:')
for r in rows:
    err_msg = r[3][:80] if r[3] else 'None'
    print(f'{r[0]}: {r[1]} | {r[2]} | {err_msg}')

conn.close()
