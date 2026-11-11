"""S7 audited alerts (DR §6): every alert lands in an append-only, SHA-256-chained spool BEFORE any delivery is
attempted; delivery results are appended after; nothing fires unaudited. Adapters (Telegram, desktop, email) take
injected transports and are env-gated; the spool is always on.
"""
