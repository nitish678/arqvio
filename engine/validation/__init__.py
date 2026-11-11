"""Statistical validation core for ARQVIO.

This package is intentionally standalone: every module here operates on plain
NumPy arrays of returns. It does NOT import the data layer, the backtester,
MT5, or any I/O. If you find yourself wanting to add such a dependency, the
math is wrong here, or the caller is in the wrong place.

What lives here:

    stats           Per-observation Sharpe, moments, annualisation, equity, MaxDD
    dsr             Deflated Sharpe Ratio
    psr             Probabilistic Sharpe Ratio
    pbo_cscv        Probability of Backtest Overfitting via CSCV
    purged_cv       Purged + embargoed cross-validation splitter
    minbtl          Minimum Backtest Length
    vs_random       Vs-random benchmark (p(real > random), max-vs-max)
    trial_registry  Durable count of every trial (rejects included)

Conventions for every function in this package:

    * Input is a 1-D ``numpy.ndarray`` of returns (per-period arithmetic), or
      a sequence of per-trial Sharpe ratios where noted.
    * Output is a frozen dataclass (or a plain float) — never a mutable dict.
    * No I/O except the trial_registry (SQLite WAL, mandatory provenance).
    * Functions are pure; ``inspect.signature`` shows all inputs.

References:

    * Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
      for Selection Bias, Non-Normal Returns, and Correlation in the Sharpe
      Ratio of a Trading Strategy."
    * Bailey, Borwein, López de Prado, Zhu (2014), "Pseudo-Mathematics and
      Financial Charlatanism."
    * López de Prado (2018), "Advances in Financial Machine Learning", ch. 7.

Cross-checked against `arqvio v1` reference (Sep 2026). The vs_random module
specifically is implemented to avoid the documented max-vs-single trap: it
bootstraps the distribution of the maximum of a random search of the same
size, not the distribution of individual random trials. `p_single` is still
reported as a diagnostic so the trap remains visible in reports.
"""

__all__ = [
    "stats",
    "dsr",
    "psr",
    "pbo_cscv",
    "purged_cv",
    "minbtl",
    "vs_random",
    "trial_registry",
]
