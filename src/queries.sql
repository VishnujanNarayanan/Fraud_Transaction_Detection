-- ============================================================================
-- Fraud analytics over the PaySim transactions table.
--
-- These are the aggregations the notebook performed in pandas, expressed as SQL
-- against the SQLite database built by src/db.py. Keeping them here rather than
-- inline in a notebook cell means the analysis is readable by anyone who knows
-- SQL but not pandas, and can be run against a warehouse table unchanged.
--
-- Each query is delimited by a "-- name: <key>" header, which src/db.py splits on.
-- ============================================================================


-- name: row_count
-- Table size and the raw fraud base rate. Every other number is read against this.
SELECT
    COUNT(*)                                        AS transactions,
    SUM(isFraud)                                    AS fraud_rows,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)       AS fraud_percent
FROM transactions;


-- name: fraud_by_type
-- Fraud volume and rate per transaction channel. This is the single most important
-- cut in the dataset: fraud is confined to two of the five channels, which is what
-- makes type_CASH_OUT and type_TRANSFER dominate the fitted coefficients.
SELECT
    type,
    COUNT(*)                                        AS transactions,
    SUM(isFraud)                                    AS fraud_rows,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)       AS fraud_percent,
    ROUND(AVG(amount), 2)                           AS mean_amount
FROM transactions
GROUP BY type
ORDER BY fraud_rows DESC;


-- name: contingency_type_fraud
-- The type-by-fraud contingency table the chi-square test consumes. Emitting it
-- from SQL means the test is fed a result anyone can reproduce with a query.
SELECT
    type,
    SUM(CASE WHEN isFraud = 0 THEN 1 ELSE 0 END)    AS not_fraud,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END)    AS fraud
FROM transactions
GROUP BY type
ORDER BY type;


-- name: amount_summary_by_class
-- Amount distribution split by class. Fraudulent transfers are an order of
-- magnitude larger on average, which is the difference the Mann-Whitney U test
-- confirms is not chance.
SELECT
    isFraud,
    COUNT(*)                                        AS transactions,
    ROUND(MIN(amount), 2)                           AS min_amount,
    ROUND(AVG(amount), 2)                           AS mean_amount,
    ROUND(MAX(amount), 2)                           AS max_amount
FROM transactions
GROUP BY isFraud;


-- name: amount_percentiles
-- Percentiles by class, computed with a window function rather than pulling
-- 6.36M rows into memory to call .quantile(). NTILE puts each row in a
-- hundredth, and the boundary row of each bucket of interest is the percentile.
WITH ranked AS (
    SELECT
        isFraud,
        amount,
        NTILE(100) OVER (PARTITION BY isFraud ORDER BY amount) AS pct
    FROM transactions
)
SELECT
    isFraud,
    MAX(CASE WHEN pct = 25 THEN amount END)         AS p25,
    MAX(CASE WHEN pct = 50 THEN amount END)         AS p50,
    MAX(CASE WHEN pct = 75 THEN amount END)         AS p75,
    MAX(CASE WHEN pct = 95 THEN amount END)         AS p95,
    MAX(CASE WHEN pct = 99 THEN amount END)         AS p99
FROM ranked
GROUP BY isFraud;


-- name: fraud_by_hour
-- Fraud rate by hour of day. `step` is an hour counter over a 744-hour simulation,
-- so step % 24 recovers the clock hour -- the same derivation the preprocessor
-- encodes cyclically as hour_sin/hour_cos.
SELECT
    step % 24                                       AS hour_of_day,
    COUNT(*)                                        AS transactions,
    SUM(isFraud)                                    AS fraud_rows,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)       AS fraud_percent
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- name: balance_anomaly_rate
-- The engineered suspicious_flag, expressed in SQL: money left the sender but the
-- recipient's balance never moved. This is the strongest engineered feature in the
-- model, and this query is how you would justify it to a risk team without
-- showing them a coefficient.
SELECT
    CASE
        WHEN amount > 0 AND newbalanceDest = oldbalanceDest THEN 1 ELSE 0
    END                                             AS suspicious_flag,
    COUNT(*)                                        AS transactions,
    SUM(isFraud)                                    AS fraud_rows,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)       AS fraud_percent
FROM transactions
GROUP BY suspicious_flag
ORDER BY suspicious_flag;


-- name: flagged_vs_actual
-- The dataset's own rule ("flag transfers over 200,000") against the truth. It
-- catches a negligible share of real fraud, which is the argument for a model and
-- the reason isFlaggedFraud is dropped as leakage rather than used as a feature.
SELECT
    isFlaggedFraud,
    COUNT(*)                                        AS transactions,
    SUM(isFraud)                                    AS fraud_rows
FROM transactions
GROUP BY isFlaggedFraud
ORDER BY isFlaggedFraud;


-- name: top_fraud_destinations
-- Recipient accounts receiving the most fraudulent value. A fraud desk works a
-- list like this directly; it is the plainest example of what the data is FOR.
SELECT
    nameDest                                        AS destination_account,
    COUNT(*)                                        AS fraud_transfers,
    ROUND(SUM(amount), 2)                           AS total_amount
FROM transactions
WHERE isFraud = 1
GROUP BY nameDest
ORDER BY total_amount DESC
LIMIT 20;
