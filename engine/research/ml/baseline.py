"""Deterministic, dependency-free ML baseline for shadow research.

This is a logistic scorer trained only on chronological train rows. It has no
fit-on-test path, no trading side effects, and returns a model fingerprint for
reproducible experiments. It is a comparator, not a strategy or risk manager.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray
    test: np.ndarray
    purge: int
    embargo: int


def chronological_split(n: int, *, train_fraction: float = 0.7, purge: int = 0, embargo: int = 0) -> TimeSplit:
    if n < 4 or not 0.0 < train_fraction < 1.0 or purge < 0 or embargo < 0:
        raise ValueError("need n>=4, train_fraction in (0,1), purge/embargo >= 0")
    cut = int(n * train_fraction)
    train_end = max(0, cut - purge)
    test_start = min(n, cut + embargo)
    train, test = np.arange(train_end, dtype=np.int64), np.arange(test_start, n, dtype=np.int64)
    if not len(train) or not len(test):
        raise ValueError("purge/embargo removed all train or test rows")
    return TimeSplit(train, test, cut - train_end, test_start - cut)


@dataclass(frozen=True)
class BaselineModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    train_end: int
    feature_names: tuple[str, ...]
    model_hash: str

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = _standardize(features, self.mean, self.scale)
        z = np.clip(x @ self.weights + self.bias, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def to_manifest(self) -> dict[str, Any]:
        return {"schema": "arqvio.ml.baseline", "version": 1, "model_hash": self.model_hash,
                "train_end": self.train_end, "feature_names": list(self.feature_names),
                "scope": "shadow research only; no orders or risk decisions"}


def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != mean.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a finite 2-D array matching the training width")
    return (x - mean) / scale


def fit_baseline(features: np.ndarray, labels: np.ndarray, *, feature_names: tuple[str, ...], split: TimeSplit,
                 epochs: int = 200, learning_rate: float = 0.05, l2: float = 1e-3) -> BaselineModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(feature_names) != x.shape[1]:
        raise ValueError("features, labels and feature_names shapes must agree")
    if set(split.train) & set(split.test) or max(split.train) >= min(split.test):
        raise ValueError("train/test must be chronological and disjoint")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("labels must be binary 0/1")
    train_x, train_y = x[split.train], y[split.train]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = _standardize(train_x, mean, scale)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(int(epochs)):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ weights + bias, -60.0, 60.0)))
        error = p - train_y
        weights -= learning_rate * (z.T @ error / len(z) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    payload = json.dumps({"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
                          "bias": bias, "train_end": int(max(split.train) + 1),
                          "feature_names": feature_names}, sort_keys=True, separators=(",", ":"))
    return BaselineModel(mean, scale, weights, bias, int(max(split.train) + 1), feature_names,
                         hashlib.sha256(payload.encode()).hexdigest())
