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
- **Cross-asset dollar features — confirmed on a pre-registered, one-time test check.** `features/cross_asset.py` adds three features per second instrument (`secondary_log_return`, `ratio_log_return`, `rolling_correlation`). Single tuned runs of this comparison flipped between runs (an earlier claimed EUR/USD win did not reproduce), so the question was settled with three steps instead of one run:
  1. **Reproducibility fix.** Rolling correlation over windows where EUR/USD did not move (forward-filled through its own data gaps) is undefined; pandas left ~1e-11 of floating-point residue there, so the same CSVs gave NaN on Linux but arbitrary values on Windows, i.e. different datasets (19,884 vs 19,926 rows). Flat windows are now set to 0.0; every machine builds the same 19,962 rows.
  2. **Noise-aware comparison on development data only.** `scripts/seed_sweep.py` (fixed RF hyperparameters, many seeds) showed seed noise of ~0.004 macro-F1 and that tuning noise, not seed noise, had been flipping earlier results. `scripts/walk_forward_sweep.py` then scored every feature set on the same rows across 5 expanding-window periods (Apr 2025–May 2026). EUR/USD alone and USD/JPY alone each beat `base` in 4/5 periods (+0.018 / +0.019); both together (`plus_combined`, 11 features) beat it in 5/5 (+0.022). An equal-weighted synthetic dollar index from both pairs did no better than either alone (+0.016, 4/5), so the two pairs carry partly different information. Momentum features were inconsistent (2/5) and are not adopted.
  3. **One-time test check, rule fixed in git before running.** `scripts/final_test_check.py` compared exactly `base` vs `plus_combined` on the untouched test partition (2026-05-30 to 2026-10-01, 2,990 rows), trained on all development data, 10 seeds, with the decision rule committed beforehand: confirmed only if `plus_combined` wins on mean macro-F1 *and* on at least 8/10 seeds. Result: **0.3831 vs 0.3678 (+0.015), 10/10 seeds — confirmed.** Recorded in `reports/final_test_check.json`; the script refuses to run again, so that test partition is now used up for this question.
  **Caveats:** the gain is small and regime-dependent — on the older Jan–May 2026 window, trained only on data up to January, the two configurations tied (−0.0008, 4/10 seeds), which suggests the dollar relationship needs recent training data. Most of the gain is in HOLD (F1 0.655 → 0.712): the dollar features mainly help the model recognise when to stay out (195 trades vs 625), not which direction to take (SELL F1 fell slightly, 0.226 → 0.207). Signal-quality cumulative return is less negative (−0.40 vs −0.96) but still negative, so this is a better model, not a profitable one.
- **Production model** — `scripts/train_production_model.py` fits the validated `plus_combined` configuration (fixed RF hyperparameters, no re-tuning) on all available data and saves it to `models/XAUUSD_1h_rf_balanced_plus_combined_production_v1`. Its `metadata.json` records which currency pairs it needs (`secondary_symbols`) and embeds the walk-forward and one-time test evidence, since a model trained on everything has no held-out score of its own; any future evaluation must use data after its training window (ends 2026-10-01). `load_model` now accepts any feature columns the pipeline can rebuild for the recorded instruments, and `predict_from_ohlcv` takes the matching `secondary_ohlcv` and refuses to predict if a currency feed is more than 24 hours behind gold. In the saved model, the USD/JPY and EUR/USD rolling correlations rank 4th and 5th of 11 features by importance, behind only ATR%, MACD histogram% and EMA gap. The original model-selection experiment, `scripts/train_xauusd_model.py`, is unchanged.
- **Capital-constrained backtest — unprofitable after costs.** `src/backtest/engine.py` simulates one account: signal at a bar's close, entry at the next bar's open, 5-bar hold, one position at a time, 100% of equity per trade (no leverage), longs and shorts, 0.05% cost per side; rules fixed before any result was seen. `scripts/backtest_walk_forward.py` feeds it genuinely out-of-sample signals (retrained monthly on past data only, 2025-04-28 to 2026-10-01, 5 seeds). Result: `base` −52% (Sharpe −2.44), `plus_combined` −34% (Sharpe −1.62), buy-and-hold gold +26% (Sharpe +0.76). The dollar features lose less mainly by trading 45% less. The model does have a small gross edge (about +0.04% per trade before costs, seed 0), but the 0.10% round-trip cost is more than double it; break-even is roughly 0.02% per side, and even at zero cost (+21%) it trails buy-and-hold. That cost breakdown is descriptive only: no rule was changed in response, and any change (longer holding period, higher confidence threshold, cheaper venue) must be validated on data after 2026-10-01. Results in `reports/backtest_walk_forward.json`.

**Not started yet:**

- Ensemble learning and explainable AI
- A proper capital-constrained backtesting engine (the current trading evaluation is a per-row signal-quality check, explicitly *not* a portfolio backtest)
- Risk management, paper trading, dashboard, deployment, and multi-asset expansion

---

## Known Issues

- **Experiment log entries recorded before the model-selection fix** have a `notes` field that incorrectly states Random Forest was chosen based on a validation comparison. Earlier versions of `scripts/train_xauusd_model.py` hardcoded Random Forest as the winner regardless of its actual validation score; a run against ~20,000 candles of XAUUSD data later showed Random Forest losing to Logistic Regression on validation macro-F1 (0.296 vs 0.336), a result the log never reflected at the time. `experiments/experiments.jsonl` is append-only by design, so those entries were left unedited rather than rewritten — this note is the correction, not the fix. The underlying bug is fixed: model selection is now a real, validation-based comparison between candidates, logged accurately (see the `Candidate`/`evaluate_candidate` logic in `scripts/train_xauusd_model.py`).
- **A previously reported cross-asset win (`plus_eurusd` beating `base` on both validation and test) was unverified and did not reproduce.** That specific claim stays retracted. The cross-asset question was later re-run with seed averaging, walk-forward validation and a pre-registered one-time test check (see Current Status), which is the result to cite. Treat any result in chat history, notes, or prior session handoffs as provisional until it has a matching record on disk (`experiments.jsonl`, a persisted model, or a file under `reports/`).

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