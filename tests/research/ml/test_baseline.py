import numpy as np
import pytest

from engine.research.ml.baseline import chronological_split, fit_baseline


def test_split_is_chronological_purged_and_embargoed():
    split = chronological_split(100, train_fraction=0.7, purge=5, embargo=3)
    assert split.train[-1] == 64 and split.test[0] == 73
    assert set(split.train).isdisjoint(split.test)
    assert split.purge == 5 and split.embargo == 3


def test_baseline_fits_train_only_and_reproduces_hash():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 2))
    y = (x[:, 0] > 0).astype(float)
    split = chronological_split(100, purge=5, embargo=5)
    a = fit_baseline(x, y, feature_names=("signal", "atr"), split=split)
    b = fit_baseline(x, y, feature_names=("signal", "atr"), split=split)
    assert a.model_hash == b.model_hash
    assert len(a.predict_proba(x[split.test])) == len(split.test)
    assert a.train_end == 65
    assert np.all((a.predict_proba(x[split.test]) >= 0) & (a.predict_proba(x[split.test]) <= 1))


def test_test_only_information_cannot_change_fit():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 2))
    y = (x[:, 0] > 0).astype(float)
    split = chronological_split(80, purge=2, embargo=2)
    a = fit_baseline(x, y, feature_names=("a", "b"), split=split)
    x2, y2 = x.copy(), y.copy()
    x2[split.test, 0] += 1_000_000
    y2[split.test] = 1.0 - y2[split.test]
    b = fit_baseline(x2, y2, feature_names=("a", "b"), split=split)
    assert a.model_hash == b.model_hash


def test_invalid_or_leaky_inputs_refuse():
    with pytest.raises(ValueError):
        chronological_split(20, purge=20)
    with pytest.raises(ValueError):
        fit_baseline(np.ones((4, 2)), np.array([0, 1, 2, 1]), feature_names=("a", "b"),
                     split=chronological_split(4))
