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

**In progress:**

- Evaluating whether additional engineered features (multi-horizon momentum, recent-range position) actually improve on the original feature set, rather than assuming more features help.

**Not started yet:**

- Formal hyperparameter tuning (`TimeSeriesSplit`-based)
- Ensemble learning and explainable AI
- A proper capital-constrained backtesting engine (the current trading evaluation is a per-row signal-quality check, explicitly *not* a portfolio backtest)
- Risk management, paper trading, dashboard, deployment, and multi-asset expansion

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