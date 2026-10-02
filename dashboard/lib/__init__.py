"""Dashboard library — modular building blocks for the static dashboard generator.

The dashboard generator is split into small focused modules:

* :mod:`registry`      – the single source of truth for tickers and models.
* :mod:`config_loader` – loads and validates ``dashboard/config.json``.
* :mod:`data_fetcher`  – yfinance wrapper with simple in-memory cache.
* :mod:`indicators`    – technical indicator computation.
* :mod:`signals`       – per-model signal generation (technical + ML proxy).
* :mod:`ml_loader`     – loads ``stock_ml/`` artifacts when available.
* :mod:`portfolio`     – portfolio simulator with full audit trail.
* :mod:`metrics`       – classification + portfolio performance metrics.
* :mod:`exporters`     – JSON / CSV writers with the new ``docs/data/`` layout.
* :mod:`audit`         – generation metadata, git hash, timestamps.

Each module is intentionally small enough to describe in an engineering thesis.
"""
