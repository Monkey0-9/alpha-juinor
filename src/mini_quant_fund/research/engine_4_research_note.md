# The Post-Decay Playbook: Engine #4 Candidates

## Premise

"Network Alpha" (Insider Clusters) will eventually decay as markets become efficient or regulations tighten. We must prepare **Engine #4** now.

## Candidate A: Options Order Flow Imbalance (OFI)

- **Concept**: Track net buying pressure of OTM Calls vs Puts for individual equities.
- **Edge**: Significant OFI in short-dated options often precedes stock price jumps (gamma squeezes or informed speculation).
- **Data Source**: CBOE or OPRA feed (expensive but high signal).
- **Implementation**: `data/alternative/options_flow.py`.

## Candidate B: Supply Chain Disruption Index

- **Concept**: Use NLP to scrape shipping manifests and trade news for "delay", "shortage", "port strike".
- **Edge**: Supply shocks hit gross margins 1-2 quarters out. Early detection allows shorting affected sectors (e.g., Retail, Auto).
- **Data Source**: ImportGenius API or financial news scrapers.

## Recommendation

**Proceed with Candidate A (Options OFI)**.

- **Reason**: Higher frequency signal (matches our RL timeframe) and stronger academic backing.
- **Action**: Begin collecting data for `SPY` and top 10 tickers.
