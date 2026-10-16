"""Deterministic, offline historical backtesting for LEADER-X signals.

Signals are formed after a historical session closes and execute at the next
eligible session's open. Portfolio values are marked consistently at daily
opens, so the return from one open to the next is earned before any next
rebalance trade is applied. Costs are charged once on the actual notional
traded at each execution.

The implementation is deliberately conservative and small: long-only,
equal-weight, monthly rebalancing, no leverage and fractional shares. The
initial dataset is synthetic, so results validate pipeline and accounting
behavior only; they do not establish predictive power or profitability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .expectations import calculate_expectations_gap
from .models import Company, CorporateAnnouncement, DailyPrice, QuarterlyFinancial

IST = timezone(timedelta(hours=5, minutes=30))
StrategyName = Literal["leader_x", "momentum_baseline"]
RunStatus = Literal["valid", "insufficient_data"]


class BacktestConfig(BaseModel):
    """Explicit assumptions shared by both strategies."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date
    initial_capital: float = Field(gt=0)
    top_k: int = Field(default=3, ge=1)
    rebalance_days: int = Field(default=21, ge=1)
    momentum_lookback_days: int = Field(default=63, ge=2)
    slippage_bps: float = Field(default=10.0, ge=0)
    commission_bps: float = Field(default=0.0, ge=0)
    other_cost_bps: float = Field(default=0.0, ge=0)
    minimum_closed_positions: int = Field(default=1, ge=0)

    @property
    def total_cost_bps(self) -> float:
        return self.slippage_bps + self.commission_bps + self.other_cost_bps


class BacktestTrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: StrategyName
    company_id: str
    signal_as_of: datetime
    execution_date: date
    execution_price: float = Field(gt=0)
    quantity_delta: float
    traded_notional: float = Field(ge=0)
    transaction_cost: float = Field(ge=0)
    reason: Literal["entry", "exit", "rebalance"]
    data_quality_status: str


class ClosedPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: StrategyName
    company_id: str
    entry_date: date
    exit_date: date
    entry_value: float = Field(gt=0)
    exit_value: float = Field(gt=0)
    gross_return: float
    net_return: float
    data_quality_status: str


class PortfolioPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    portfolio_value: float = Field(gt=0)
    cash: float
    holdings_value: float = Field(ge=0)


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: StrategyName
    status: RunStatus
    data_quality_status: str
    evaluation_start: date | None
    evaluation_end: date | None
    annualization_days: int
    initial_capital: float
    terminal_value: float
    gross_return: float | None
    net_return: float | None
    annualized_net_return: float | None
    trade_count: int
    closed_position_count: int
    win_rate: float | None
    maximum_drawdown: float | None
    total_transaction_costs: float
    portfolio_values: list[PortfolioPoint]
    trades: list[BacktestTrade]
    closed_positions: list[ClosedPosition]
    excluded_signals: list[str]
    explanation: str


class BacktestComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    data_quality_status: str
    leader_x: BacktestResult
    momentum_baseline: BacktestResult
    net_return_difference: float | None
    explanation: str


@dataclass(frozen=True)
class _Signal:
    as_of: datetime
    execution_date: date
    targets: dict[str, float]
    excluded: list[str]


def _valuation_price(
    company_id: str,
    valuation_date: date,
    price_by_company: dict[str, dict[date, DailyPrice]],
) -> float | None:
    """Use the last known open for marking, never as an implied trade price."""
    history = price_by_company.get(company_id, {})
    available_dates = [item for item in history if item <= valuation_date]
    if not available_dates:
        return None
    return float(history[max(available_dates)].open)


def _holdings_value(
    holdings: dict[str, float],
    valuation_date: date,
    price_by_company: dict[str, dict[date, DailyPrice]],
) -> float:
    return sum(
        shares * price
        for company_id, shares in holdings.items()
        if (price := _valuation_price(company_id, valuation_date, price_by_company)) is not None
    )


@dataclass
class _OpenPosition:
    shares: float
    entry_date: date
    entry_value: float
    entry_cost: float


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return value


def _price_map(prices: list[DailyPrice]) -> dict[str, dict[date, DailyPrice]]:
    result: dict[str, dict[date, DailyPrice]] = {}
    for price in prices:
        result.setdefault(price.company_id, {})[price.trade_date] = price
    return result


def _all_dates(price_by_company: dict[str, dict[date, DailyPrice]]) -> list[date]:
    return sorted({trade_date for prices in price_by_company.values() for trade_date in prices})


def _next_date(dates: list[date], current: date) -> date | None:
    return next((item for item in dates if item > current), None)


def _signal_dates(dates: list[date], config: BacktestConfig) -> list[date]:
    eligible = [item for item in dates if config.start_date <= item <= config.end_date]
    return eligible[:: config.rebalance_days]


def _as_of(trade_date: date) -> datetime:
    return datetime.combine(trade_date, datetime.max.time(), IST)


def _company_quality(companies: list[Company], financials: list[QuarterlyFinancial], prices: list[DailyPrice]) -> str:
    statuses = {item.data_quality_status for item in [*companies, *financials, *prices]}
    return "synthetic" if statuses == {"synthetic"} else ("partial" if len(statuses) > 1 else "unverified")


def _eligible_universe(
    companies: list[Company],
    financials_by_company: dict[str, list[QuarterlyFinancial]],
    prices_by_company: dict[str, list[DailyPrice]],
    announcements_by_company: dict[str, list[CorporateAnnouncement]],
    as_of: datetime,
    execution_date: date,
    momentum_lookback_days: int,
) -> tuple[set[str], list[str]]:
    """Build the common point-in-time universe used by both strategies."""
    eligible: set[str] = set()
    excluded: list[str] = []
    for company in sorted(companies, key=lambda item: item.company_id):
        company_prices = sorted(
            (row for row in prices_by_company.get(company.company_id, []) if row.trade_date <= as_of.date()),
            key=lambda item: item.trade_date,
        )
        all_company_prices = prices_by_company.get(company.company_id, [])
        if execution_date not in {row.trade_date for row in all_company_prices}:
            excluded.append(f"{company.company_id}: no next-session execution price")
            continue
        if len(company_prices) <= momentum_lookback_days:
            excluded.append(f"{company.company_id}: insufficient point-in-time price history")
            continue
        company_financials = financials_by_company.get(company.company_id, [])
        if not company_financials:
            excluded.append(f"{company.company_id}: missing point-in-time financial history")
            continue
        try:
            research_result = calculate_expectations_gap(
                company_financials,
                company_prices,
                as_of,
                announcements=announcements_by_company.get(company.company_id, []),
            )
        except ValueError as exc:
            excluded.append(f"{company.company_id}: {exc}")
            continue
        if research_result.signal_status != "valid":
            excluded.append(f"{company.company_id}: {research_result.signal_status}")
            continue
        eligible.add(company.company_id)
    return eligible, excluded


def _leader_targets(
    companies: list[Company],
    financials_by_company: dict[str, list[QuarterlyFinancial]],
    prices_by_company: dict[str, list[DailyPrice]],
    announcements_by_company: dict[str, list[CorporateAnnouncement]],
    as_of: datetime,
    execution_date: date,
    top_k: int,
    eligible_company_ids: set[str],
) -> tuple[dict[str, float], list[str]]:
    candidates: list[tuple[float, str]] = []
    excluded: list[str] = []
    for company in sorted(companies, key=lambda item: item.company_id):
        if company.company_id not in eligible_company_ids:
            continue
        company_prices = [row for row in prices_by_company.get(company.company_id, []) if row.trade_date <= as_of.date()]
        if execution_date not in {row.trade_date for row in prices_by_company.get(company.company_id, [])}:
            excluded.append(f"{company.company_id}: no next-session execution price")
            continue
        company_financials = financials_by_company.get(company.company_id, [])
        if not company_financials or not company_prices:
            excluded.append(f"{company.company_id}: missing historical inputs")
            continue
        try:
            result = calculate_expectations_gap(
                company_financials,
                company_prices,
                as_of,
                announcements=announcements_by_company.get(company.company_id, []),
            )
        except ValueError as exc:
            excluded.append(f"{company.company_id}: {exc}")
            continue
        if result.signal_status != "valid" or result.expectations_gap_score is None:
            excluded.append(f"{company.company_id}: {result.signal_status}")
            continue
        candidates.append((result.expectations_gap_score, company.company_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [company_id for _, company_id in candidates[:top_k]]
    weight = 1.0 / len(selected) if selected else 0.0
    return {company_id: weight for company_id in selected}, excluded


def _momentum_targets(
    companies: list[Company],
    prices_by_company: dict[str, list[DailyPrice]],
    as_of: date,
    execution_date: date,
    top_k: int,
    lookback_days: int,
    eligible_company_ids: set[str],
) -> tuple[dict[str, float], list[str]]:
    candidates: list[tuple[float, str]] = []
    excluded: list[str] = []
    for company in sorted(companies, key=lambda item: item.company_id):
        if company.company_id not in eligible_company_ids:
            continue
        rows = sorted(prices_by_company.get(company.company_id, []), key=lambda item: item.trade_date)
        rows = [row for row in rows if row.trade_date <= as_of]
        if execution_date not in {row.trade_date for row in prices_by_company.get(company.company_id, [])}:
            excluded.append(f"{company.company_id}: no next-session execution price")
            continue
        if len(rows) <= lookback_days:
            excluded.append(f"{company.company_id}: insufficient price history for baseline")
            continue
        previous = rows[-lookback_days - 1].adjusted_close or rows[-lookback_days - 1].close
        latest = rows[-1].adjusted_close or rows[-1].close
        if previous <= 0:
            excluded.append(f"{company.company_id}: invalid baseline denominator")
            continue
        candidates.append((float(latest / previous - 1), company.company_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [company_id for _, company_id in candidates[:top_k]]
    weight = 1.0 / len(selected) if selected else 0.0
    return {company_id: weight for company_id in selected}, excluded


def _apply_rebalance(
    strategy: StrategyName,
    signal: _Signal,
    cash: float,
    holdings: dict[str, float],
    open_positions: dict[str, _OpenPosition],
    price_by_company: dict[str, dict[date, DailyPrice]],
    portfolio_value: float,
    config: BacktestConfig,
    data_quality_status: str,
) -> tuple[float, float, list[BacktestTrade], list[ClosedPosition]]:
    execution_date = signal.execution_date
    target_values = {company_id: portfolio_value * weight for company_id, weight in signal.targets.items()}
    current_values = {
        company_id: shares * float(price_by_company[company_id][execution_date].open)
        for company_id, shares in holdings.items()
        if company_id in price_by_company and execution_date in price_by_company[company_id]
    }
    all_companies = sorted(set(holdings) | set(target_values))
    trades: list[BacktestTrade] = []
    closed: list[ClosedPosition] = []
    total_cost = 0.0
    for company_id in all_companies:
        price = price_by_company.get(company_id, {}).get(execution_date)
        if price is None:
            continue
        execution_price = float(price.open)
        target_value = target_values.get(company_id, 0.0)
        current_value = current_values.get(company_id, 0.0)
        delta_value = target_value - current_value
        delta_shares = delta_value / execution_price
        traded_notional = abs(delta_value)
        if traded_notional <= 1e-9:
            continue
        cost = traded_notional * config.total_cost_bps / 10_000
        total_cost += cost
        new_shares = holdings.get(company_id, 0.0) + delta_shares
        old_position = open_positions.get(company_id)
        target_is_zero = target_value <= 1e-9
        if old_position and target_is_zero:
            entry_value = old_position.entry_value
            exit_value = current_value
            closed.append(
                ClosedPosition(
                    strategy=strategy,
                    company_id=company_id,
                    entry_date=old_position.entry_date,
                    exit_date=execution_date,
                    entry_value=entry_value,
                    exit_value=exit_value,
                    gross_return=(exit_value - entry_value) / entry_value,
                    net_return=(exit_value - entry_value - old_position.entry_cost - cost) / entry_value,
                    data_quality_status=data_quality_status,
                )
            )
        elif not old_position and new_shares > 0:
            open_positions[company_id] = _OpenPosition(
                shares=new_shares,
                entry_date=execution_date,
                entry_value=target_value,
                entry_cost=cost,
            )
        elif old_position:
            old_position.shares = new_shares
            old_position.entry_cost += cost
        if target_is_zero:
            holdings.pop(company_id, None)
            open_positions.pop(company_id, None)
            reason: Literal["entry", "exit", "rebalance"] = "exit"
        else:
            holdings[company_id] = new_shares
            reason = "entry" if old_position is None else "rebalance"
        cash -= delta_value + cost
        trades.append(
            BacktestTrade(
                strategy=strategy,
                company_id=company_id,
                signal_as_of=signal.as_of,
                execution_date=execution_date,
                execution_price=execution_price,
                quantity_delta=delta_shares,
                traded_notional=traded_notional,
                transaction_cost=cost,
                reason=reason,
                data_quality_status=data_quality_status,
            )
        )
    return cash, total_cost, trades, closed


def _drawdown(values: list[PortfolioPoint]) -> float | None:
    if not values:
        return None
    peak = values[0].portfolio_value
    maximum = 0.0
    for point in values:
        peak = max(peak, point.portfolio_value)
        maximum = max(maximum, (peak - point.portfolio_value) / peak)
    return round(maximum, 6)


def _result(
    strategy: StrategyName,
    status: RunStatus,
    config: BacktestConfig,
    data_quality_status: str,
    values: list[PortfolioPoint],
    trades: list[BacktestTrade],
    closed: list[ClosedPosition],
    excluded: list[str],
    total_costs: float,
) -> BacktestResult:
    terminal = values[-1].portfolio_value if values else config.initial_capital
    observed_start = values[0].date if values else None
    observed_end = values[-1].date if values else None
    elapsed_days = (observed_end - observed_start).days if observed_start and observed_end else 0
    net_return = terminal / config.initial_capital - 1 if values else None
    gross_terminal = terminal + total_costs
    gross_return = gross_terminal / config.initial_capital - 1 if values else None
    annualized = (
        (1 + net_return) ** (365 / elapsed_days) - 1
        if net_return is not None and elapsed_days > 0 and 1 + net_return > 0
        else None
    )
    wins = sum(position.net_return > 0 for position in closed)
    win_rate = wins / len(closed) if closed else None
    inadequate_period = elapsed_days <= 0
    if len(closed) < config.minimum_closed_positions or inadequate_period:
        status = "insufficient_data"
        gross_return = None
        net_return = None
        annualized = None
    explanation = (
        "Synthetic data only: results validate deterministic signal, execution and accounting behavior; "
        "they do not establish predictive power or real-world profitability. "
        "Terminal holdings are valued using the last available open for marking only and are not treated as "
        "closed positions for win rate; a missing later price is not an implied liquidation price."
    )
    if status == "insufficient_data":
        explanation += " Headline performance returns are suppressed because the evaluation is insufficient."
    return BacktestResult(
        strategy=strategy,
        status=status,
        data_quality_status=data_quality_status,
        evaluation_start=observed_start,
        evaluation_end=observed_end,
        annualization_days=365,
        initial_capital=config.initial_capital,
        terminal_value=round(terminal, 6),
        gross_return=round(gross_return, 6) if gross_return is not None else None,
        net_return=round(net_return, 6) if net_return is not None else None,
        annualized_net_return=round(annualized, 6) if annualized is not None else None,
        trade_count=len(trades),
        closed_position_count=len(closed),
        win_rate=round(win_rate, 6) if win_rate is not None else None,
        maximum_drawdown=_drawdown(values),
        total_transaction_costs=round(total_costs, 6),
        portfolio_values=values,
        trades=trades,
        closed_positions=closed,
        excluded_signals=excluded,
        explanation=explanation,
    )


def run_backtest(
    companies: list[Company],
    financials: list[QuarterlyFinancial],
    prices: list[DailyPrice],
    announcements: list[CorporateAnnouncement],
    config: BacktestConfig,
    strategy: StrategyName,
) -> BacktestResult:
    """Run one strategy using only information available at each signal date."""
    if config.end_date <= config.start_date:
        raise ValueError("end_date must be after start_date")
    if not prices:
        return _result(strategy, "insufficient_data", config, "unverified", [], [], [], ["no prices"], 0.0)
    price_by_company = _price_map(prices)
    dates = _all_dates(price_by_company)
    evaluation_dates = [item for item in dates if config.start_date <= item <= config.end_date]
    if len(evaluation_dates) < 2:
        return _result(strategy, "insufficient_data", config, "synthetic", [], [], [], ["fewer than two evaluation sessions"], 0.0)
    financials_by_company: dict[str, list[QuarterlyFinancial]] = {}
    prices_by_company: dict[str, list[DailyPrice]] = {}
    announcements_by_company: dict[str, list[CorporateAnnouncement]] = {}
    for row in financials:
        financials_by_company.setdefault(row.company_id, []).append(row)
    for row in prices:
        prices_by_company.setdefault(row.company_id, []).append(row)
    for row in announcements:
        announcements_by_company.setdefault(row.company_id, []).append(row)
    quality = _company_quality(companies, financials, prices)
    signal_dates = _signal_dates(evaluation_dates, config)
    signals: list[_Signal] = []
    excluded: list[str] = []
    for signal_date in signal_dates:
        execution_date = _next_date(dates, signal_date)
        if execution_date is None or execution_date > config.end_date:
            excluded.append(f"{signal_date}: no eligible next trading session")
            continue
        signal_as_of = _as_of(signal_date)
        eligible_company_ids, universe_reasons = _eligible_universe(
            companies,
            financials_by_company,
            prices_by_company,
            announcements_by_company,
            signal_as_of,
            execution_date,
            config.momentum_lookback_days,
        )
        excluded.extend(f"{signal_date}: {reason}" for reason in universe_reasons)
        if strategy == "leader_x":
            targets, reasons = _leader_targets(
                companies,
                financials_by_company,
                prices_by_company,
                announcements_by_company,
                signal_as_of,
                execution_date,
                config.top_k,
                eligible_company_ids,
            )
        else:
            targets, reasons = _momentum_targets(
                companies,
                prices_by_company,
                signal_date,
                execution_date,
                config.top_k,
                config.momentum_lookback_days,
                eligible_company_ids,
            )
        excluded.extend(f"{signal_date}: {reason}" for reason in reasons)
        signals.append(_Signal(signal_as_of, execution_date, targets, reasons))

    cash = config.initial_capital
    holdings: dict[str, float] = {}
    open_positions: dict[str, _OpenPosition] = {}
    trades: list[BacktestTrade] = []
    closed: list[ClosedPosition] = []
    values = [
        PortfolioPoint(
            date=evaluation_dates[0],
            portfolio_value=config.initial_capital,
            cash=config.initial_capital,
            holdings_value=0.0,
        )
    ]
    total_costs = 0.0
    signals_by_execution = {signal.execution_date: signal for signal in signals}
    for current_date in evaluation_dates[1:]:
        holdings_value = _holdings_value(holdings, current_date, price_by_company)
        portfolio_value = cash + holdings_value
        signal = signals_by_execution.get(current_date)
        if signal is not None:
            cash, costs, new_trades, new_closed = _apply_rebalance(
                strategy,
                signal,
                cash,
                holdings,
                open_positions,
                price_by_company,
                portfolio_value,
                config,
                quality,
            )
            total_costs += costs
            trades.extend(new_trades)
            closed.extend(new_closed)
            holdings_value = _holdings_value(holdings, current_date, price_by_company)
            portfolio_value = cash + holdings_value
        if portfolio_value > 0 and isfinite(portfolio_value):
            values.append(
                PortfolioPoint(
                    date=current_date,
                    portfolio_value=portfolio_value,
                    cash=cash,
                    holdings_value=holdings_value,
                )
            )
    return _result(strategy, "valid", config, quality, values, trades, closed, excluded, total_costs)


def compare_strategies(
    companies: list[Company],
    financials: list[QuarterlyFinancial],
    prices: list[DailyPrice],
    announcements: list[CorporateAnnouncement],
    config: BacktestConfig,
) -> BacktestComparison:
    """Compare LEADER-X and momentum using identical execution/accounting rules."""
    leader_x = run_backtest(companies, financials, prices, announcements, config, "leader_x")
    baseline = run_backtest(companies, financials, prices, announcements, config, "momentum_baseline")
    comparable = leader_x.status == "valid" and baseline.status == "valid"
    difference = (
        leader_x.net_return - baseline.net_return
        if comparable and leader_x.net_return is not None and baseline.net_return is not None
        else None
    )
    status: RunStatus = "valid" if comparable else "insufficient_data"
    return BacktestComparison(
        status=status,
        data_quality_status="synthetic" if leader_x.data_quality_status == baseline.data_quality_status == "synthetic" else "partial",
        leader_x=leader_x,
        momentum_baseline=baseline,
        net_return_difference=round(difference, 6) if difference is not None else None,
        explanation=(
            "Both strategies use identical signal dates, next-session open execution, position sizing and costs. "
            "Synthetic results do not establish real-world profitability."
            if comparable
            else "Comparison is insufficient because one or both strategies lack enough closed positions or valid history."
        ),
    )
