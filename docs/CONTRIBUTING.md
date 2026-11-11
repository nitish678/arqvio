# Contributing

Welcome. ARQVIO is at an early stage, and this public snapshot deliberately excludes private datasets, credentials and competitor research. Contributions should respect those boundaries.

## Ways to help

- Setup reproducibility: fresh Windows / RDP installs, dependency locks, offline tests.
- Documentation: clearer setup, risk warnings and method boundaries.
- Correctness: regression tests for launch accounting, backtest parity fallbacks and portfolio math (using fixtures).
- Accessibility/UI: keyboard navigation, error/empty states and clear unsupported-feature notices.

## Financial-safety rules

- Pull requests involving deployment, MT5 attachment, account credentials or risk limits need an explicit action label, not a silent runtime default.
- Do not publish locker/vault exports, market histories, license keys or OS keyring material in commits, diffs or screenshots.
- Use only an authorized demo/test broker and an isolated environment for any MT5-dependent change.

## Pull requests

1. Describe what ran, not what was intended. Include pass/fail counts, fixture names and any skips.
2. State whether tests use pinned private fixtures.
3. Keep commits atomic; prefer small correct fixes over speculative layers.
4. Reviewers check persisted formats, replay race conditions and silent defaults before merging.
5. Report missing artifacts or impossible fixture dependencies rather than weakening tests.

## Releases and branding

Maintainers publish installers/artifacts separately. Do not build and publish "ARQVIO" executables from this source under this name without a separate maintainer release decision. Forked projects should use their own branding.
