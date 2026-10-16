"""Deterministic, explainable Business Momentum scoring.

The score intentionally uses only fields present in ``QuarterlyFinancial``.
Each component is scored from 0 to 100 and is computed from the latest
consecutive, comparable quarters that are available as of the requested
timestamp. Missing values remain unavailable; they are never converted to
zero. Available component weights are reweighted to sum to 100.

Normalisation rules:

* Percentage-point changes use a fixed 20-point scale and are clamped to
  ``[-1, 1]`` before mapping to ``[0, 100]``. This limits outliers.
* Positive level changes use ``(latest - previous) / max(abs(previous), 1)``.
  A zero denominator therefore remains safe and deterministic.
* Balance-sheet metrics are improvements when debt or net debt/EBITDA falls.
* Consistency is the fraction of available adjacent comparisons that improve,
  mapped to a 0-100 score. A flat comparison counts as 50.

Freshness policy: the latest eligible financial report must have an
``effective_from`` no more than 450 days before ``as_of``. Older data receives
``stale_data`` status and no score. A valid component requires two consecutive
fiscal quarters with the relevant metric available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import QuarterlyFinancial

COMPONENT_WEIGHTS: dict[str, float] = {
    "revenue_trend": 35.0,
    "profitability_trend": 25.0,
    "cash_flow_trend": 20.0,
    "balance_sheet_trend": 10.0,
    "operating_consistency": 10.0,
}
FRESHNESS_LIMIT = timedelta(days=450)
SignalStatus = Literal["valid", "insufficient_data", "stale_data"]


class ComponentScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float | None = Field(default=None, ge=0, le=100)
    original_weight: float
    used_weight: float
    available: bool
    metrics_used: list[str]
    periods_compared: list[str]
    improved: list[str]
    deteriorated: list[str]
    missing_inputs: list[str]
    explanation: str


class ScoreExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics_improved: list[str]
    metrics_deteriorated: list[str]
    periods_compared: list[str]
    missing_inputs: list[str]
    component_weights_used: dict[str, float]
    why: str
    data_quality_status: str


class BusinessMomentumResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str
    as_of: datetime
    business_momentum_score: float | None = Field(default=None, ge=0, le=100)
    component_scores: dict[str, ComponentScore]
    coverage_ratio: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    signal_status: SignalStatus
    latest_effective_from: datetime | None
    data_quality_status: str
    explanation: ScoreExplanation


@dataclass(frozen=True)
class _Pair:
    previous: QuarterlyFinancial
    latest: QuarterlyFinancial


def _ensure_as_of(as_of: datetime) -> datetime:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include timezone information")
    return as_of


def _quarter_number(value: QuarterlyFinancial) -> int:
    return value.fiscal_year * 4 + int(value.fiscal_quarter[1:]) - 1


def _latest_pair(rows: list[QuarterlyFinancial], metric_names: tuple[str, ...]) -> _Pair | None:
    eligible = [row for row in rows if all(getattr(row, name) is not None for name in metric_names)]
    for latest, previous in zip(reversed(eligible), reversed(eligible[:-1])):
        if _quarter_number(latest) - _quarter_number(previous) == 1:
            return _Pair(previous=previous, latest=latest)
    return None


def _periods(pair: _Pair | None) -> list[str]:
    if pair is None:
        return []
    return [
        f"{pair.previous.fiscal_year}-{pair.previous.fiscal_quarter}",
        f"{pair.latest.fiscal_year}-{pair.latest.fiscal_quarter}",
    ]


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _map_change(change: float, scale: float = 1.0) -> float:
    if not isfinite(change):
        raise ValueError("non-finite metric change")
    return round(50.0 + 50.0 * _clamp(change / scale), 6)


def _percentage_point_change(pair: _Pair, field: str) -> tuple[float, str]:
    previous = float(getattr(pair.previous, field))
    latest = float(getattr(pair.latest, field))
    return _map_change(latest - previous, scale=20.0), field


def _level_change(pair: _Pair, field: str, lower_is_better: bool = False) -> tuple[float, str]:
    previous = float(getattr(pair.previous, field))
    latest = float(getattr(pair.latest, field))
    denominator = max(abs(previous), 1.0)
    change = (latest - previous) / denominator
    if lower_is_better:
        change = -change
    return _map_change(change), field


def _revenue_change(pair: _Pair) -> tuple[float, str]:
    if pair.previous.revenue_yoy_pct is not None and pair.latest.revenue_yoy_pct is not None:
        return _percentage_point_change(pair, "revenue_yoy_pct")
    previous = float(pair.previous.revenue or 0)
    latest = float(pair.latest.revenue or 0)
    if previous <= 0:
        raise ValueError("revenue denominator is not positive")
    return _map_change((latest - previous) / previous), "revenue"


def _component(
    name: str,
    pair: _Pair | None,
    metrics: list[tuple[float, str]],
    missing: list[str],
    original_weight: float,
    direction_words: tuple[str, str] = ("improved", "deteriorated"),
) -> ComponentScore:
    periods = _periods(pair)
    if pair is None or not metrics:
        return ComponentScore(
            name=name,
            score=None,
            original_weight=original_weight,
            used_weight=0.0,
            available=False,
            metrics_used=[],
            periods_compared=periods,
            improved=[],
            deteriorated=[],
            missing_inputs=missing,
            explanation=f"{name} is unavailable because comparable inputs are missing.",
        )
    score = round(sum(value for value, _ in metrics) / len(metrics), 6)
    improved = [metric for value, metric in metrics if value > 50.0]
    deteriorated = [metric for value, metric in metrics if value < 50.0]
    direction = f"{direction_words[0]}: {', '.join(improved)}" if improved else "no improving metric"
    if deteriorated:
        direction += f"; {direction_words[1]}: {', '.join(deteriorated)}"
    return ComponentScore(
        name=name,
        score=score,
        original_weight=original_weight,
        used_weight=original_weight,
        available=True,
        metrics_used=[metric for _, metric in metrics],
        periods_compared=periods,
        improved=improved,
        deteriorated=deteriorated,
        missing_inputs=missing,
        explanation=f"{direction}. Normalised component score: {score:.2f}.",
    )


def _consistency(rows: list[QuarterlyFinancial], weight: float) -> ComponentScore:
    comparisons: list[float] = []
    improved: list[str] = []
    deteriorated: list[str] = []
    periods: list[str] = []
    missing: list[str] = []
    for previous, latest in zip(rows, rows[1:]):
        if _quarter_number(latest) - _quarter_number(previous) != 1:
            continue
        period = f"{previous.fiscal_year}-{previous.fiscal_quarter}→{latest.fiscal_year}-{latest.fiscal_quarter}"
        period_values: list[float] = []
        for field in ("revenue_yoy_pct", "ebitda_margin_pct", "pat_yoy_pct", "operating_cash_flow"):
            if getattr(previous, field) is None or getattr(latest, field) is None:
                continue
            if field == "operating_cash_flow":
                result = _level_change(_Pair(previous, latest), field)[0]
            else:
                result = _percentage_point_change(_Pair(previous, latest), field)[0]
            period_values.append(result)
        if period_values:
            period_score = sum(period_values) / len(period_values)
            comparisons.append(period_score)
            periods.append(period)
            if period_score > 50:
                improved.append(period)
            elif period_score < 50:
                deteriorated.append(period)
    if not comparisons:
        return ComponentScore(
            name="operating_consistency",
            score=None,
            original_weight=weight,
            used_weight=0.0,
            available=False,
            metrics_used=[],
            periods_compared=[],
            improved=[],
            deteriorated=[],
            missing_inputs=["at least two consecutive quarters with an operating metric"],
            explanation="Operating consistency is unavailable because no comparable operating metrics exist.",
        )
    score = round(sum(comparisons) / len(comparisons), 6)
    return ComponentScore(
        name="operating_consistency",
        score=score,
        original_weight=weight,
        used_weight=weight,
        available=True,
        metrics_used=["revenue_yoy_pct", "ebitda_margin_pct", "pat_yoy_pct", "operating_cash_flow"],
        periods_compared=periods,
        improved=improved,
        deteriorated=deteriorated,
        missing_inputs=missing,
        explanation=f"Compared {len(comparisons)} consecutive quarter transitions; score: {score:.2f}.",
    )


def _quality(rows: list[QuarterlyFinancial]) -> str:
    statuses = {row.data_quality_status for row in rows}
    if statuses == {"synthetic"}:
        return "synthetic"
    if len(statuses) > 1 or "partial" in statuses:
        return "partial"
    return next(iter(statuses), "unverified")


def _confidence(coverage_ratio: float, age_days: int, freshness_limit: timedelta) -> float:
    limit_days = freshness_limit.days
    if age_days > limit_days:
        return 0.0
    freshness_factor = 1.0 - 0.5 * (age_days / limit_days)
    return round(max(0.0, min(1.0, coverage_ratio * freshness_factor)), 6)


def calculate_business_momentum(
    financials: list[QuarterlyFinancial],
    as_of: datetime,
    *,
    freshness_limit: timedelta = FRESHNESS_LIMIT,
) -> BusinessMomentumResult:
    """Calculate a point-in-time Business Momentum Score for one company."""
    as_of = _ensure_as_of(as_of)
    if not financials:
        raise ValueError("at least one financial record is required")
    company_id = financials[0].company_id
    if any(row.company_id != company_id for row in financials):
        raise ValueError("financial records must belong to one company")
    eligible = sorted((row for row in financials if row.effective_from <= as_of), key=lambda row: row.period_end_date)
    latest = eligible[-1] if eligible else None
    quality = _quality(eligible or financials)
    if latest is None:
        status: SignalStatus = "insufficient_data"
        latest_effective = None
        age_days = freshness_limit.days + 1
    else:
        latest_effective = latest.effective_from
        age_days = max(0, (as_of - latest_effective).days)
        status = "stale_data" if age_days > freshness_limit.days else "valid"
    if len(eligible) < 2:
        status = "insufficient_data"

    revenue_pair = _latest_pair(eligible, ("revenue_yoy_pct",)) or _latest_pair(eligible, ("revenue",))
    profitability_pairs = []
    for field in ("ebitda_margin_pct", "pat_yoy_pct"):
        pair = _latest_pair(eligible, (field,))
        if pair is not None:
            profitability_pairs.append((field, pair))
    profitability_pair = profitability_pairs[0][1] if profitability_pairs else None
    cash_pairs = [pair for field in ("operating_cash_flow", "free_cash_flow") if (pair := _latest_pair(eligible, (field,))) is not None]
    balance_pairs = [pair for field in ("debt", "net_debt_to_ebitda") if (pair := _latest_pair(eligible, (field,))) is not None]

    revenue_metrics: list[tuple[float, str]] = []
    revenue_missing: list[str] = []
    if revenue_pair is None:
        revenue_missing.append("revenue_yoy_pct or revenue across two consecutive quarters")
    else:
        try:
            revenue_metrics.append(_revenue_change(revenue_pair))
        except ValueError:
            revenue_missing.append("positive prior revenue denominator")
    components: dict[str, ComponentScore] = {}
    components["revenue_trend"] = _component(
        "revenue_trend", revenue_pair, revenue_metrics, revenue_missing, COMPONENT_WEIGHTS["revenue_trend"]
    )

    profitability_metrics: list[tuple[float, str]] = []
    profitability_missing: list[str] = []
    for field, pair in profitability_pairs:
        profitability_metrics.append(_percentage_point_change(pair, field))
    if not profitability_metrics:
        profitability_missing.append("ebitda_margin_pct or pat_yoy_pct across two consecutive quarters")
    components["profitability_trend"] = _component(
        "profitability_trend",
        profitability_pair,
        profitability_metrics,
        profitability_missing,
        COMPONENT_WEIGHTS["profitability_trend"],
    )

    cash_metrics: list[tuple[float, str]] = []
    cash_missing: list[str] = []
    for field in ("operating_cash_flow", "free_cash_flow"):
        pair = _latest_pair(eligible, (field,))
        if pair:
            cash_metrics.append(_level_change(pair, field))
    if not cash_metrics:
        cash_missing.append("operating_cash_flow or free_cash_flow across two consecutive quarters")
    components["cash_flow_trend"] = _component(
        "cash_flow_trend",
        cash_pairs[0] if cash_pairs else None,
        cash_metrics,
        cash_missing,
        COMPONENT_WEIGHTS["cash_flow_trend"],
    )

    balance_metrics: list[tuple[float, str]] = []
    balance_missing: list[str] = []
    for field in ("debt", "net_debt_to_ebitda"):
        pair = _latest_pair(eligible, (field,))
        if pair:
            balance_metrics.append(_level_change(pair, field, lower_is_better=True))
    if not balance_metrics:
        balance_missing.append("debt or net_debt_to_ebitda across two consecutive quarters")
    components["balance_sheet_trend"] = _component(
        "balance_sheet_trend",
        balance_pairs[0] if balance_pairs else None,
        balance_metrics,
        balance_missing,
        COMPONENT_WEIGHTS["balance_sheet_trend"],
    )
    components["operating_consistency"] = _consistency(eligible, COMPONENT_WEIGHTS["operating_consistency"])

    available_weight = sum(component.original_weight for component in components.values() if component.available)
    coverage_ratio = round(available_weight / 100.0, 6)
    if status == "valid" and available_weight == 0:
        status = "insufficient_data"
    used_weights = {
        name: round(component.original_weight / available_weight * 100.0, 6)
        for name, component in components.items()
        if component.available and available_weight
    }
    for name, component in components.items():
        component.used_weight = used_weights.get(name, 0.0)

    score: float | None = None
    if status == "valid" and available_weight:
        score = round(
            sum(component.score * component.original_weight for component in components.values() if component.available)
            / available_weight,
            6,
        )
    if status == "valid" and latest is not None:
        confidence = _confidence(coverage_ratio, age_days, freshness_limit)
    else:
        confidence = (
            0.0
            if status == "stale_data"
            else _confidence(coverage_ratio, age_days, freshness_limit)
        )

    improved = [item for component in components.values() for item in component.improved]
    deteriorated = [item for component in components.values() for item in component.deteriorated]
    periods = sorted({period for component in components.values() for period in component.periods_compared})
    missing = sorted({item for component in components.values() for item in component.missing_inputs})
    why = (
        f"Score is {score:.2f} after reweighting {sum(component.available for component in components.values())} "
        f"available components; coverage is {coverage_ratio:.2%} and confidence is {confidence:.2%}."
        if score is not None
        else f"No score assigned because signal status is {status}. Coverage is {coverage_ratio:.2%}."
    )
    if quality == "synthetic":
        why += " Inputs are synthetic and do not establish trading profitability."
    explanation = ScoreExplanation(
        metrics_improved=improved,
        metrics_deteriorated=deteriorated,
        periods_compared=periods,
        missing_inputs=missing,
        component_weights_used=used_weights,
        why=why,
        data_quality_status=quality,
    )
    return BusinessMomentumResult(
        company_id=company_id,
        as_of=as_of,
        business_momentum_score=score,
        component_scores=components,
        coverage_ratio=coverage_ratio,
        confidence_score=confidence,
        signal_status=status,
        latest_effective_from=latest_effective,
        data_quality_status=quality,
        explanation=explanation,
    )
