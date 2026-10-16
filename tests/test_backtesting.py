from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from leader_x.backtesting import (
    BacktestConfig,
    _apply_rebalance,
    _holdings_value,
    _price_map,
    _Signal,
    compare_strategies,
    run_backtest,
)
from leader_x.models import Company, DailyPrice, QuarterlyFinancial

IST = timezone(timedelta(hours=5, minutes=30))


def company(company_id: str = "alpha") -> Company:
    return Company(
        company_id=company_id,
        ticker=f"SYN_{company_id.upper()}",
        company_name=f"Synthetic {company_id.title()}",
        sector="Synthetic",
        listing_date=date(2023, 1, 1),
        source="synthetic-test",
        data_quality_status="synthetic",
    )


def financial(quarter: int, revenue_yoy: str = "10", effective_from: datetime | None = None):
    period_end = {1: date(2024, 3, 31), 2: date(2024, 6, 30), 3: date(2024, 9, 30)}[quarter]
    reported = datetime(2024, 2, 28, tzinfo=IST) if effective_from is None else effective_from - timedelta(hours=1)
    effective = effective_from or datetime(2024, 3, 1, tzinfo=IST)
    return QuarterlyFinancial(
        financial_id=f"alpha-2024q{quarter}",
        company_id="alpha",
        fiscal_year=2024,
        fiscal_quarter=f"Q{quarter}",
        period_end_date=period_end,
        reported_at=min(reported, effective - timedelta(hours=1)),
        effective_from=effective,
        revenue=Decimal("100"),
        revenue_yoy_pct=Decimal(revenue_yoy),
        ebitda=Decimal("15"),
        ebitda_margin_pct=Decimal("15"),
        profit_after_tax=Decimal("8"),
        pat_yoy_pct=Decimal("10"),
        eps=Decimal("0.8"),
        operating_cash_flow=Decimal("20"),
        free_cash_flow=Decimal("15"),
        debt=Decimal("100"),
        net_debt_to_ebitda=Decimal("3"),
        source="synthetic-test",
        data_quality_status="synthetic",
    )


def prices(company_id: str = "alpha", closes: list[float] | None = None) -> list[DailyPrice]:
    closes = closes or [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    rows = []
    trade_date = date(2024, 1, 2)
    for close in closes:
        while trade_date.weekday() >= 5:
            trade_date += timedelta(days=1)
        value = Decimal(str(close))
        rows.append(
            DailyPrice(
                company_id=company_id,
                trade_date=trade_date,
                open=value,
                high=value + 1,
                low=value - Decimal("1"),
                close=value,
                adjusted_close=value,
                volume=1000,
                source="synthetic-test",
                data_quality_status="synthetic",
            )
        )
        trade_date += timedelta(days=1)
    return rows


def config(**overrides) -> BacktestConfig:
    values = {
        "start_date": date(2024, 6, 3),
        "end_date": date(2024, 7, 31),
        "initial_capital": 1000,
        "top_k": 1,
        "rebalance_days": 3,
        "momentum_lookback_days": 2,
        "minimum_closed_positions": 0,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def test_next_session_open_execution_and_open_to_open_valuation():
    rows = prices(closes=[100 + index for index in range(140)])
    result = run_backtest([company()], [financial(1), financial(2)], rows, [], config(), "momentum_baseline")
    first_trade = result.trades[0]
    assert first_trade.execution_date > first_trade.signal_as_of.date()
    expected_price = next(row.open for row in rows if row.trade_date == first_trade.execution_date)
    assert first_trade.execution_price == float(expected_price)
    assert any(point.date == first_trade.execution_date for point in result.portfolio_values)


def test_transaction_costs_reduce_net_return_and_are_charged_once():
    rows = prices(closes=[100 + index for index in range(140)])
    no_cost = run_backtest([company()], [financial(1), financial(2)], rows, [], config(), "momentum_baseline")
    with_cost = run_backtest(
        [company()], [financial(1), financial(2)], rows, [], config(slippage_bps=100), "momentum_baseline"
    )
    assert with_cost.total_transaction_costs > 0
    assert with_cost.net_return < no_cost.net_return
    assert with_cost.trade_count >= no_cost.trade_count


def test_future_financial_report_cannot_change_earlier_backtest_signal():
    rows = prices(closes=[100 + index for index in range(140)])
    base = [financial(1, revenue_yoy="5"), financial(2, revenue_yoy="10")]
    future = financial(
        3,
        revenue_yoy="999",
        effective_from=datetime(2025, 1, 1, tzinfo=IST),
    )
    first = run_backtest([company()], base, rows, [], config(), "leader_x")
    second = run_backtest([company()], base + [future], rows, [], config(), "leader_x")
    assert [trade.execution_date for trade in first.trades] == [trade.execution_date for trade in second.trades]
    assert [trade.traded_notional for trade in first.trades] == [trade.traded_notional for trade in second.trades]


def test_missing_history_is_explicit_and_not_a_fabricated_result():
    result = run_backtest(
        [company()],
        [financial(1)],
        prices(closes=[100 + index for index in range(140)]),
        [],
        config(momentum_lookback_days=20, minimum_closed_positions=1),
        "leader_x",
    )
    assert result.status == "insufficient_data"
    assert result.net_return is None
    assert result.excluded_signals


def test_results_are_deterministic_and_include_initial_value_in_drawdown():
    inputs = (
        [company()],
        [financial(1), financial(2)],
        prices(closes=[100] * 140 + [100, 90, 110, 80, 100]),
        [],
        config(),
    )
    first = run_backtest(*inputs, strategy="momentum_baseline")
    second = run_backtest(*inputs, strategy="momentum_baseline")
    assert first.model_dump() == second.model_dump()
    assert first.portfolio_values[0].portfolio_value == first.initial_capital
    assert first.maximum_drawdown is not None
    assert first.maximum_drawdown > 0


def test_baseline_and_leader_x_share_accounting_and_are_labeled_synthetic():
    result = compare_strategies(
        [company()],
        [financial(1), financial(2), financial(3)],
        prices(closes=[100 + index for index in range(140)]),
        [],
        config(minimum_closed_positions=0),
    )
    assert result.data_quality_status == "synthetic"
    assert result.leader_x.data_quality_status == "synthetic"
    assert result.momentum_baseline.data_quality_status == "synthetic"
    assert result.leader_x.explanation == result.momentum_baseline.explanation


def test_intermediate_rebalance_cost_is_included_in_closed_position_return():
    rows = prices(closes=[100, 101, 102, 103])
    price_map = _price_map(rows)
    settings = config(slippage_bps=100)
    holdings = {}
    open_positions = {}
    cash, _, _, _ = _apply_rebalance(
        "momentum_baseline",
        _Signal(datetime(2024, 1, 2, 23, 59, tzinfo=IST), rows[1].trade_date, {"alpha": 1.0}, []),
        1000.0,
        holdings,
        open_positions,
        price_map,
        1000.0,
        settings,
        "synthetic",
    )
    cash, _, _, _ = _apply_rebalance(
        "momentum_baseline",
        _Signal(datetime(2024, 1, 3, 23, 59, tzinfo=IST), rows[2].trade_date, {"alpha": 0.5}, []),
        cash,
        holdings,
        open_positions,
        price_map,
        990.0,
        settings,
        "synthetic",
    )
    assert open_positions["alpha"].entry_cost > 10.0
    _, _, _, closed = _apply_rebalance(
        "momentum_baseline",
        _Signal(datetime(2024, 1, 4, 23, 59, tzinfo=IST), rows[3].trade_date, {}, []),
        cash,
        holdings,
        open_positions,
        price_map,
        990.0,
        settings,
        "synthetic",
    )
    assert closed
    assert open_positions == {}
    assert closed[0].net_return < closed[0].gross_return


def test_missing_price_is_carried_for_marking_not_treated_as_liquidation():
    rows = prices(closes=[100, 110, 120])
    value = _holdings_value({"alpha": 2.0}, rows[-1].trade_date + timedelta(days=5), _price_map(rows))
    assert value == 240.0


def test_reported_period_and_annualization_use_observed_portfolio_dates():
    result = run_backtest(
        [company()],
        [financial(1), financial(2)],
        prices(closes=[100 + index for index in range(145)]),
        [],
        config(minimum_closed_positions=0),
        "momentum_baseline",
    )
    assert result.evaluation_start == result.portfolio_values[0].date
    assert result.evaluation_end == result.portfolio_values[-1].date
    elapsed = (result.evaluation_end - result.evaluation_start).days
    expected = (1 + result.net_return) ** (365 / elapsed) - 1
    assert result.annualized_net_return == pytest.approx(expected, abs=2e-6)


def test_insufficient_results_suppress_all_headline_returns():
    result = run_backtest(
        [company()],
        [financial(1)],
        prices(closes=[100 + index for index in range(145)]),
        [],
        config(minimum_closed_positions=1),
        "leader_x",
    )
    assert result.status == "insufficient_data"
    assert result.gross_return is None
    assert result.net_return is None
    assert result.annualized_net_return is None


def test_baseline_cannot_select_company_outside_shared_leader_x_universe():
    companies = [company("alpha"), company("beta")]
    alpha_prices = prices("alpha", closes=[100 + index for index in range(145)])
    beta_prices = prices("beta", closes=[1000 - index for index in range(145)])
    result = run_backtest(
        companies,
        [financial(1), financial(2)],
        alpha_prices + beta_prices,
        [],
        config(minimum_closed_positions=0),
        "momentum_baseline",
    )
    assert result.trades
    assert {trade.company_id for trade in result.trades} == {"alpha"}
