# LEADER-X India

LEADER-X India is an explainable Indian stock-discovery research engine being
built for the OmniRush Frontier Buildathon 2026. This checkout currently
contains Phases 1–6: the reproducible foundation, validated synthetic data,
Business Momentum Score, Expectations Gap Proxy, Streamlit research dashboard
and an offline historical backtesting engine.

## Data safety and scope

The included demonstration dataset is deterministic, fictional and labelled
`synthetic` in every record. Its symbols use the `SYN_` prefix and are not NSE
tickers. Synthetic results validate data plumbing and software behavior only;
they do not represent real securities and do not establish trading
profitability.

The older local RSI screener remains in `server.py` and `screener.py`. It is
not part of the LEADER-X foundation work and was not modified.

## Requirements

- Python 3.12 (the project records this in `.python-version`)
- uv 0.12.10 or compatible
- Docker 29+ for the container check (optional)

No credentials or network access are required to generate or inspect the
synthetic data.

## Local setup

```powershell
uv sync
uv run python -m leader_x.synthetic
```

The generator writes:

- `data/synthetic/companies.parquet`
- `data/synthetic/quarterly_financials.parquet`
- `data/synthetic/daily_prices.parquet`
- `data/synthetic/corporate_announcements.parquet`
- `data/leader_x.duckdb`

All paths are project-relative. The DuckDB database is built from the Parquet
files and can be queried offline.

## Run the Streamlit research dashboard

Launch the offline research dashboard with:

```powershell
uv run streamlit run app.py
```

The dashboard has three pages:

- **Stock Discovery** — ranks fictional companies using the Expectations Gap
  Proxy, with filters for minimum gap score, signal status and data quality.
- **Company Research** — shows Business Momentum Score, Expectations Gap Proxy,
  component weights, confidence, coverage, financial trends, prices and the
  corporate-announcement timeline for a selected company.
- **Evidence & Risk** — explains the evidence boundary, proxy limitations,
  missing inputs and synthetic-data restrictions.

The dashboard makes confidence, coverage, missing-data warnings and
`synthetic` data-quality labels visible rather than hiding them behind a rank.
The Business Momentum Score and Expectations Gap Proxy are research signals;
the current fictional dataset and these scores do not establish investment
returns, predictive power or trading profitability.

## Historical Backtesting Engine

Phase 6 provides a deterministic, synthetic-data-only backtesting engine in
`src/leader_x/backtesting.py`. It reuses the existing Business Momentum Score
and Expectations Gap Proxy without changing their formulas. It compares the
combined LEADER-X signal with a simple trailing-price momentum baseline under
identical execution and portfolio-accounting rules.

The engine's default assumptions are explicit and configurable:

- Long-only, unleveraged portfolio.
- Monthly-style rebalancing using a configurable trading-session interval.
- Equal-weight top-K holdings, with cash retained when fewer than K valid
  candidates exist.
- Signals use only financials and announcements available by the historical
  `as_of` timestamp, and prices through that session only.
- Both LEADER-X and the momentum baseline use the same point-in-time eligible
  research universe, including financial-data sufficiency, price history,
  tradability and next-session execution availability.
- A signal formed after session T closes executes at the next eligible
  trading-session open; the same-session close is never used as an entry.
- Daily portfolio values use consistent open-to-open accounting. The return
  into a rebalance session is earned before the new target portfolio is traded.
- Slippage, commission and other costs are configurable basis-point charges on
  actual traded notional, including entries, exits and rebalances.
- Terminal holdings are valued at the final available open. They are not
  treated as closed positions for win-rate calculations.
- If a held company has no later price, its last available open is carried
  forward for marking only. This is not an invented liquidation price and does
  not imply that a delisted holding was tradable.

The result reports trade count, gross return before direct simulated costs, net
return, annualized net return using a 365-day year and the actual observed
portfolio-value period, closed-position win rate, maximum drawdown from a
portfolio-value series that includes initial capital, transaction costs,
exclusions and comparison against the momentum baseline. Intermediate
rebalance costs are included in the lifecycle accounting of closed positions.
If there is not enough history, not enough closed positions or an inadequate
observed period, the engine returns `insufficient_data` and suppresses headline
gross, net and annualized returns rather than implying that a strategy is
effective.

All initial outputs are labelled `synthetic`. They validate signal timing and
accounting behavior only; they do not establish predictive power, investment
returns or real-world profitability.

## Checks

Run the canonical test command:

```powershell
uv run pytest
```

Run lint checks for the new foundation modules:

```powershell
uv run ruff check .
```

The tests cover model validation, missing metrics, duplicate announcement
identities, timezone-aware publication/effective timestamps, OHLC validation,
synthetic labels, generated dataset shape, Business Momentum scoring and
Expectations Gap Proxy timestamp/missing-data safeguards.

## Docker

```powershell
docker build -t leader-x-india .
docker run --rm -p 8501:8501 leader-x-india
```

Start Docker Desktop or the Docker daemon before running these commands. The
image uses Python 3.12.10, installs from the committed `uv.lock`, creates the
synthetic dataset offline during the image build, and starts the Streamlit
dashboard at `http://localhost:8501`.

## Invoke the backtesting engine

The backtesting engine is currently invoked through its Python API; it does not
have a dedicated CLI. The following example uses the existing synthetic
dataset loader and the implemented `BacktestConfig` and
`compare_strategies` arguments:

```python
from datetime import date
from pathlib import Path

from leader_x.backtesting import BacktestConfig, compare_strategies
from leader_x.dashboard import load_dataset

dataset = load_dataset(Path("data/synthetic"))
config = BacktestConfig(
    start_date=date(2024, 1, 2),
    end_date=date(2025, 12, 31),
    initial_capital=100_000,
)
comparison = compare_strategies(
    dataset.companies,
    dataset.financials,
    dataset.prices,
    dataset.announcements,
    config,
)
print(comparison.status)
```

This uses synthetic data only. The dataset may correctly return
`insufficient_data`; the example does not claim to establish a valid
performance comparison, predictive power or profitability.

## Project layout

```text
src/leader_x/models.py       Validated domain models
src/leader_x/storage.py      Parquet and DuckDB persistence
src/leader_x/synthetic.py    Deterministic synthetic dataset generator
src/leader_x/validation.py   Collection-level duplicate checks
src/leader_x/dashboard.py    Dashboard data loading and view models
src/leader_x/backtesting.py   Historical signal and portfolio accounting
app.py                       Streamlit research dashboard
scripts/                     Local command wrappers
tests/                       Foundation, scoring, dashboard and backtesting tests
```

## Business Momentum Score

Phase 3 now provides a deterministic score from 0 to 100 using only fields in
`QuarterlyFinancial`:

| Component | Weight |
|---|---:|
| Revenue trend | 35% |
| Profitability trend | 25% |
| Cash-flow trend | 20% |
| Balance-sheet trend | 10% |
| Operating consistency | 10% |

Components use the latest consecutive comparable quarters available at the
requested `as_of` timestamp. Percentage-point changes use a fixed 20-point
normalisation scale; level changes use a denominator of `max(abs(previous), 1)`;
all changes are clamped before mapping to 0–100 so outliers cannot produce
unbounded scores. Debt and net debt/EBITDA improve when they fall. Missing
metrics do not become zero: the component is unavailable, the remaining
weights are reweighted transparently, and confidence is reduced by the
coverage ratio.

Financial reports are eligible only when `effective_from <= as_of`. The default
freshness limit is 450 days, and a score requires at least two consecutive
quarters. Results can be `valid`, `insufficient_data` or `stale_data` and carry
structured explanations with periods, missing inputs, component weights,
metric direction and data quality.

This score is a research signal and is not a claim that synthetic data or the
score predicts profitable trades.

## Expectations Gap Proxy

The Expectations Gap Proxy is a research proxy, not analyst consensus. It uses:

| Component | Original weight |
|---|---:|
| Fundamental improvement | 50% |
| Price recognition, subtracted through non-recognition | 30% |
| Announcement surprise | 20% |

Fundamental improvement compares the current Business Momentum Score with a
prior point-in-time score. Price recognition uses available 1-month and
3-month historical returns, and uses benchmark-relative returns only when a
benchmark is supplied. Without a benchmark, the explanation explicitly states
that absolute returns were used.

Existing announcements do not contain a consensus or guidance baseline, so
announcement surprise is unavailable by default. It becomes available only
when verified observed-versus-baseline evidence is explicitly supplied. The
remaining components are then reweighted, rather than treating missing
surprise as zero.

Financials are filtered by `effective_from`, announcements by both publication
and effective timestamps, and prices by `trade_date <= as_of`. Future reports,
prices and announcement evidence cannot affect earlier results. Synthetic
outputs are labelled and do not establish profitable trading outcomes.

## Future work

Live market-data ingestion, verified analyst-consensus inputs and production
execution integrations remain future work. The current Expectations Gap is a
proxy, not a true analyst-consensus gap, and the current backtesting data is
synthetic only.
