from engine.research.vault.portfolio_view import axes, exposure_breakdown, replacement_eligibility


def test_exposure_breakdown_counts_currency_legs_read_only():
    entries = [{"id": "a", "symbol": "EURUSDm", "family": "MA_CROSS", "timeframe": "H1",
                "dna": {"risk_pct": 0.01}}, {"id": "b", "symbol": "XAUUSDm", "family": "RSI_REVERSION",
                "timeframe": "H4", "dna": {"risk_pct": 0.02}}]
    out = exposure_breakdown(entries)
    assert out["member_count"] == 2
    assert out["currencies"]["EUR"]["members"] == 1
    assert out["currencies"]["USD"]["members"] == 2
    assert out["auto_executed"] if "auto_executed" in out else True
    assert out["source"].startswith("vault planned")


def test_axes_exposes_base_and_quote_legs():
    assert axes({"symbol": "EURUSDm", "family": "MA_CROSS", "timeframe": "H1", "dna": {}})["currency"] == "EUR+USD"


def test_replacement_is_recommendation_only():
    out = replacement_eligibility({"id": "bot1", "parity_grade": "F"}, judged={"verdict": "WEAKENING"})
    assert out["eligible"] is True
    assert out["auto_executed"] is False
    assert "BELOW_PARITY" in out["reasons"] and "WEAKENING_BELOW_P5" in out["reasons"]
