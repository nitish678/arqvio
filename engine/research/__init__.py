"""Research engine.

S2 (data + costs for one symbol, one timeframe: USDCADm H2) lives here.
Subsequent slices extend the catalog to all broker symbols and timeframes
on the same interface; the per-symbol implementation never changes.

Layout (S2 slice):

    research/
    ├── data/   Symbol spec, dataset identity, bar source, time offset, spread matrix, warehouse
    └── costs/  Spread, commission, slippage, swap, weekend gap
"""
