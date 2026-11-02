"""Độ tin cậy của mô hình: nạp artifacts/trust_metrics.json (do project01/train/eval_trust.py tạo) và phục vụ qua API."""
from __future__ import annotations

import json
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "trust_metrics.json")
_CACHE: dict = {}


def load() -> dict:
    if not _CACHE:
        with open(PATH, encoding="utf-8") as f:
            _CACHE.update(json.load(f))
    return _CACHE


def _trust() -> dict:
    try:
        return load()
    except FileNotFoundError:
        return dict(error="Chưa có trust_metrics.json. Hãy chạy project01/train/eval_trust.py.")


def _fairness() -> dict:
    d = _trust()
    if "error" in d:
        return d
    return dict(groups=d["groups"], fairness=d["fairness"])


def routes(clinic, speaker):
    return {"/api/trust": _trust, "/api/trust/fairness": _fairness}, {}
