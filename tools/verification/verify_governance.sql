-- SQL Verification Script for Institutional Market Data Governance

-- 1. Symbols with insufficient rows (< 1260)
SELECT symbol, COUNT(*) AS rows
FROM price_history
WHERE date >= date('now','-5 years')
GROUP BY symbol
HAVING COUNT(*) < 1260;

-- 2. Data quality summary (last 24h)
SELECT avg(quality_score) FROM data_quality
WHERE recorded_at >= date('now','-1 day');

-- 3. Providers that returned entitlement errors
SELECT provider, COUNT(*) FROM ingestion_audit
WHERE reason_code='ENTITLEMENT_FAILURE'
GROUP BY provider;

-- 4. Successful Ingestion Summary
SELECT status, count(*) FROM ingestion_audit
WHERE finished_at >= date('now', '-1 day')
GROUP BY status;
