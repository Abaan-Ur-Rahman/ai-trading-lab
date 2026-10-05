# AI Trading Lab

> A professional quantitative research platform for financial market analysis using machine learning, explainable AI, and production-quality software engineering.

> **Status:** 🚧 Active Development (Version 0.1.0-alpha)

---

## Project Overview

AI Trading Lab is a long-term software engineering and machine learning project focused on building an extensible platform for financial market analysis.

Rather than attempting to predict future prices with certainty, the platform is designed to estimate the probability of different market outcomes while providing transparent explanations for every recommendation.

The project is being developed with the same engineering principles used in professional software systems, emphasizing modular architecture, testing, documentation, reproducibility, and maintainability.

---

## Vision

The long-term vision is to create a modular, asset-agnostic platform capable of supporting multiple financial markets, including:

- Precious Metals
- Cryptocurrencies
- Forex
- US Equities
- Exchange Traded Funds (ETFs)

The architecture is intentionally designed so new asset classes can be added without redesigning the system.

---

## Current Status

**Built and working:**

- **Data layer** — a Twelve Data API provider, a paginated historical-fetch service (for pulling years of history past a single request's limit), and CSV caching.
- **Feature engineering pipeline** — technical indicators (EMA, RSI, ATR, MACD), engineered features (log returns at multiple horizons, EMA gap, RSI, ATR%, MACD histogram%, a recent-range position indicator), and a leakage-safe dataset builder (chronological warm-up/horizon trimming, no future information in any feature).
- **ML layer** — a model-agnostic wrapper interface with Logistic Regression and Random Forest implementations, chronological train/validation/test splitting with purge gaps against label-horizon leakage, feature scaling fit only on training data, classification and trading-signal evaluation metrics, majority-class and rule-based baselines, model/scaler/metadata persistence, single-point live inference, and a local JSONL experiment log.
- **Validated methodology** — trained and evaluated end-to-end against ~20,000 hours of real XAUUSD 1h data, with model selection done strictly on a validation set and the test set touched exactly once per chosen configuration, to avoid the subtle test-set leakage that comes from picking a "winner" based on repeated test-set peeking.
- **Feature-set comparison** — ran a controlled comparison of 4 feature-set configurations (the original 5 indicators; +multi-horizon momentum; +recent-range position; all 9 combined) across both models, 8 candidates total, selected strictly on validation macro-F1. Momentum features gave Random Forest a real improvement (macro-F1 0.296 → 0.352); recent-range position showed little benefit and slightly hurt Random Forest when combined with momentum. The winning configuration's one-time test-set score (0.325) came in a bit below the previous, simpler configuration's test score (0.340) — a useful reminder that a validation-set winner doesn't automatically generalize best, which is exactly why the test set stays untouched until the final check rather than being used to pick favorites.
- **Hyperparameter tuning** — added chronological, `TimeSeriesSplit`-based tuning (3-fold CV, 18 hyperparameter configurations) for each of the 8 (feature set, model) candidates individually, rather than tuning only the previously-identified best configuration. This changed the winner: untuned, `plus_momentum` was Random Forest's best feature set (macro-F1 0.352, vs. 0.296 for `base`); tuned, `base` (0.406) overtook `plus_momentum` (0.399). That's a concrete justification for tuning all 8 candidates instead of just the prior winner — doing the cheaper thing would have picked the wrong feature set. Caveat: the top few tuned RF configs on `base` are within ~0.003 macro-F1 of each other across folds, so this is "tuning changed which candidate wins," not "base is conclusively the better feature set" — the margin is narrow enough that it could reorder again with different hyperparameter grids or more folds.
- **Validation-vs-test gap widened after tuning** — the tuned winner's validation macro-F1 (0.406) dropped to 0.340 on the one-time test check, a larger drop than the untuned comparison saw (0.352 → 0.325). This is an expected consequence of stacking hyperparameter selection (18 configs × 3-fold CV per candidate) underneath candidate selection (8 candidates): validation macro-F1 gets more chances to be optimistic even though no individual step touches the test set improperly — it's selection-on-selection, not leakage. Validation macro-F1 should be read as the *selection criterion* used to pick a configuration, not as an unbiased estimate of real-world performance; the test score is the honest estimate. Here that honest estimate is weak — the test-set trading report shows cumulative return −0.829 and hit rate 0.304, both worse than the untuned configuration's test numbers — so this model is not trading-ready as-is.
- **Pagination robustness fix** — `TwelveDataProvider.get_candles` previously treated any validation-skipped candle (a malformed OHLC record, observed in practice on the newest, still-forming candle in a live feed) as "end of history" and stopped paginating early, which could silently truncate a 20,000-candle fetch by thousands of rows. It now tells apart a real end-of-history (the API itself returned fewer raw records than asked) from a validation skip (the API returned a full page but one or more records failed OHLC validation), and only in the latter case makes a small, bounded number of follow-up requests further back in time to top up the shortfall. Verified end-to-end: a full 20,000-candle fetch now reliably returns 20,000/20,000 candles.
- **Cross-asset features, tested and currently not adopted** — added an optional second-instrument feature set (`features/cross_asset.py`: `secondary_log_return`, `ratio_log_return`, `rolling_correlation` against EUR/USD as a dollar-strength proxy, plumbed through `features/pipeline.py` and compared as its own `plus_eurusd` candidate in `scripts/train_xauusd_model.py`). Run against ~20,000 hours of real XAUUSD/EURUSD 1h data (train/val/test spanning 2023-11 to 2026-10), `plus_eurusd` (RandomForest, val macro-F1 0.3824) beat the other engineered feature sets (`plus_momentum` 0.3650, `plus_range` 0.3746, `plus_momentum_range` 0.3777) but still lost to plain `base` (0.4112), so `base` was selected again and `plus_eurusd` never reached the one-time test check — per the methodology above, only the validation winner gets evaluated on test, specifically to avoid reporting a cherry-picked number for a losing candidate. **Correction:** an earlier note in this README (and in a prior session's handoff) claimed `plus_eurusd` *won* validation (0.4080) and substantially improved test macro-F1 (0.3400 → 0.3752). That result could not be reproduced — the run that produced it left no corresponding model artifact, cached secondary-instrument data, or `experiments.jsonl` entry — and a fresh, from-scratch rerun on real data (included here) shows `base` winning instead. Treat the earlier claim as unverified and superseded by this entry. Cross-asset information is not obviously useless (it beat every other engineered feature set), but it has not yet beaten the simple baseline on any run that's actually reproducible on disk.

**Not started yet:**

- Ensemble learning and explainable AI
- A proper capital-constrained backtesting engine (the current trading evaluation is a per-row signal-quality check, explicitly *not* a portfolio backtest)
- Risk management, paper trading, dashboard, deployment, and multi-asset expansion

---

## Known Issues

- **Experiment log entries recorded before the model-selection fix** have a `notes` field that incorrectly states Random Forest was chosen based on a validation comparison. Earlier versions of `scripts/train_xauusd_model.py` hardcoded Random Forest as the winner regardless of its actual validation score; a run against ~20,000 candles of XAUUSD data later showed Random Forest losing to Logistic Regression on validation macro-F1 (0.296 vs 0.336), a result the log never reflected at the time. `experiments/experiments.jsonl` is append-only by design, so those entries were left unedited rather than rewritten — this note is the correction, not the fix. The underlying bug is fixed: model selection is now a real, validation-based comparison between candidates, logged accurately (see the `Candidate`/`evaluate_candidate` logic in `scripts/train_xauusd_model.py`).
- **A previously reported cross-asset win (`plus_eurusd` beating `base` on both validation and test) was unverified and does not reproduce.** See the "Cross-asset features" entry under Current Status for the corrected, reproducible result. Treat any result in chat history, notes, or prior session handoffs as provisional until it has a matching `experiments.jsonl` entry and persisted model artifact on disk — those are the only record that survives between sessions.

---

## Planned Capabilities

- Automated market data collection
- Feature engineering pipeline
- Technical indicator library
- Market structure analysis
- Machine learning model comparison
- Ensemble learning
- Explainable AI
- Probability-based Buy / Sell / Hold estimation
- Walk-forward backtesting
- Risk management
- Interactive dashboard
- Performance reporting

---

## Engineering Goals

This project is also intended to demonstrate professional software engineering practices, including:

- Clean Architecture
- Modular Design
- Git-based Development Workflow
- Automated Testing
- Comprehensive Documentation
- Architecture Decision Records (ADRs)
- Continuous Refactoring
- Reproducible Machine Learning Experiments

---

## Current Version

0.1.0-alpha

---

## License

MIT License — see [LICENSE](LICENSE).