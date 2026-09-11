"""Generation metadata / audit trail helpers."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_hash(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def build_run_metadata(repo_root: str, config: Dict[str, Any], artifact_status: Dict[str, bool]) -> Dict[str, Any]:
    ml_cfg = config.get("ml_signals", {}) or {}
    label_cfg = config.get("label", {}) or {}
    return {
        "generated_at_utc": utc_now_iso(),
        "git_commit": git_commit_hash(repo_root),
        "schema_version": config.get("schema_version", "2.0"),
        "data_version": config.get("data_version", "unknown"),
        "strategy_version": config.get("strategy_version", "unknown"),
        "data_period": config.get("data_period", "unknown"),
        "data_interval": config.get("data_interval", "unknown"),
        # Label semantics — surfaced so the dashboard can explain HOLD
        # without the user opening README.
        "label": {
            "mode": label_cfg.get("mode"),                # "binary" | "multiclass"
            "threshold": label_cfg.get("threshold"),
            "version": label_cfg.get("version"),
            "is_binary": label_cfg.get("mode") == "binary",
        },
        "ml_thresholds": {
            "buy_threshold": ml_cfg.get("probability_threshold_buy"),
            "sell_threshold": ml_cfg.get("probability_threshold_sell"),
            "dead_band": [
                ml_cfg.get("probability_threshold_sell"),
                ml_cfg.get("probability_threshold_buy"),
            ],
        },
        "benchmark": config.get("benchmark", "SPY"),
        "benchmark_tickers": config.get("benchmark_tickers", [config.get("benchmark", "SPY")]),
        "sector_palette": config.get("sector_palette", {}),
        "n_tickers": len(config.get("tickers", [])),
        "n_portfolios": len(config.get("portfolios", [])),
        "ml_artifacts_present": artifact_status,
        "ml_artifacts_any_real": any(artifact_status.values()) if artifact_status else False,
        "source": "dashboard/generate.py",
    }


def write_json(path: str, data: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default)


def _json_default(obj):
    # numpy and pandas types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Not JSON serializable: {type(obj)}")
