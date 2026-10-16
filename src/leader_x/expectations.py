"""Expectations Gap Proxy built on the Business Momentum Score.

This module deliberately does not claim to know analyst consensus or market
expectations. Existing corporate announcements contain no structured baseline,
so announcement surprise remains unavailable unless a caller supplies a
verified observed-versus-baseline evidence record.

The proxy uses these contributions:

* Fundamental improvement: 50% of the score.
* Price recognition: 30%, represented as ``100 - recognition`` because high
  recognition reduces the gap.
* Announcement surprise: 20%, only when a valid comparable baseline exists.

The final score is a weighted average of available components. Missing
components are not treated as zero; available component weights are
renormalised and the confidence score is reduced by coverage and source
confidence. Prices are selected at or before the requested observation date.
Financials and evidence are filtered by their effective/publication timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import CorporateAnnouncement, DailyPrice, QuarterlyFinancial
from .scoring import BusinessMomentumResult, calculate_business_momentum

PROXY_WEIGHTS: dict[str, float] = {
    "fundamental_improvement": 50.0,
    "price_recognition": 30.0,
    "announcement_surprise": 20.0,
}
PROXY_FULL_SCALE = 0.50
ProxyStatus = Literal["valid", "insufficient_data", "stale_data"]


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return value


class AnnouncementSurpriseEvidence(BaseModel):
    """Optional verified evidence; not populated by the synthetic generator."""

    model_config = ConfigDict(extra="forbid")

    announcement_id: str = Field(min_length=3)
    company_id: str = Field(min_length=3)
    observed_value: float
    baseline_value: float
    published_at: datetime
    effective_from: datetime
    source_name: str = Field(min_length=1)
    is_verified: bool = False
    data_quality_status: str

    _published_timezone = field_validator("published_at", "effective_from")(_aware)

    @model_validator(mode="after")
    def validate_evidence(self) -> AnnouncementSurpriseEvidence:
        if self.effective_from < self.published_at:
            raise ValueError("effective_from cannot precede published_at")
        if not isfinite(self.observed_value) or not isfinite(self.baseline_value):
            raise ValueError("surprise evidence values must be finite")
        if self.baseline_value == 0:
            raise ValueError("baseline_value cannot be zero")
        return self


class ProxyComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float | None = Field(default=None, ge=0, le=100)
    original_weight: float
    used_weight: float
    available: bool
    metrics_used: list[str]
    observation_windows: list[str]
    periods_compared: list[str]
    missing_inputs: list[str]
    explanation: str


class ExpectationsGapExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_weights: dict[str, float]
    reweighted_component_weights: dict[str, float]
    business_momentum_change: float | None
    momentum_comparison_dates: list[str]
    price_recognition_inputs: list[str]
    price_observation_windows: list[str]
    announcement_surprise_evidence: list[str]
    missing_data: list[str]
    data_coverage: float
    confidence: float
    signal_status: ProxyStatus
    data_quality_status: str
    why: str


class ExpectationsGapResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str
    as_of: datetime
    prior_as_of: datetime
    expectations_gap_score: float | None = Field(default=None, ge=0, le=100)
    fundamental_improvement_score: float | None = Field(default=None, ge=0, le=100)
    price_recognition_score: float | None = Field(default=None, ge=0, le=100)
    announcement_surprise_score: float | None = Field(default=None, ge=0, le=100)
    current_momentum: BusinessMomentumResult
    prior_momentum: BusinessMomentumResult
    components: dict[str, ProxyComponent]
    coverage_ratio: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    signal_status: ProxyStatus
    data_quality_status: str
    explanation: ExpectationsGapExplanation


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _change_score(change: float, scale: float) -> float:
    if not isfinite(change):
        raise ValueError("non-finite change")
    return round(50.0 + 50.0 * _clamp(change / scale), 6)


def _price_at_or_before(prices: list[DailyPrice], target: datetime) -> DailyPrice | None:
    eligible = [price for price in prices if price.trade_date <= target.date()]
    return max(eligible, key=lambda price: price.trade_date) if eligible else None


def _return_for_window(
    prices: list[DailyPrice],
    as_of: datetime,
    days: int,
) -> tuple[float, str] | None:
    end = _price_at_or_before(prices, as_of)
    start = _price_at_or_before(prices, as_of - timedelta(days=days))
    if end is None or start is None or start.close <= 0:
        return None
    return (float(end.close) / float(start.close)) - 1.0, f"{days // 30}-month"


def _price_component(
    prices: list[DailyPrice],
    as_of: datetime,
    benchmark_prices: list[DailyPrice] | None,
) -> tuple[ProxyComponent, float | None, float]:
    observations: list[str] = []
    returns: list[float] = []
    missing: list[str] = []
    benchmark_available = benchmark_prices is not None and bool(benchmark_prices)
    for days, label in ((30, "1-month"), (90, "3-month")):
        company_return = _return_for_window(prices, as_of, days)
        if company_return is None:
            missing.append(f"{label} company price window")
            continue
        value, _ = company_return
        if benchmark_available:
            benchmark_return = _return_for_window(benchmark_prices or [], as_of, days)
            if benchmark_return is None:
                missing.append(f"{label} benchmark price window")
                continue
            value -= benchmark_return[0]
            observations.append(f"{label} relative return: {value:.2%}")
        else:
            observations.append(f"{label} absolute return: {value:.2%}")
        returns.append(value)
    if not returns:
        missing.append("at least one complete 1-month or 3-month price window")
        return (
            ProxyComponent(
                name="price_recognition",
                score=None,
                original_weight=PROXY_WEIGHTS["price_recognition"],
                used_weight=0.0,
                available=False,
                metrics_used=[],
                observation_windows=[],
                periods_compared=[],
                missing_inputs=missing,
                explanation="Price recognition is unavailable because no complete historical window exists.",
            ),
            None,
            0.0,
        )
    score = round(sum(_change_score(value, PROXY_FULL_SCALE) for value in returns) / len(returns), 6)
    method = "relative to the supplied benchmark" if benchmark_available else "using absolute returns; no defensible benchmark was supplied"
    return (
        ProxyComponent(
            name="price_recognition",
            score=score,
            original_weight=PROXY_WEIGHTS["price_recognition"],
            used_weight=PROXY_WEIGHTS["price_recognition"],
            available=True,
            metrics_used=["close", "trade_date"],
            observation_windows=[item.split(":")[0] for item in observations],
            periods_compared=[],
            missing_inputs=missing,
            explanation=f"Price recognition score: {score:.2f}, {method}.",
        ),
        score,
        len(returns) / 2.0,
    )


def _announcement_component(
    company_id: str,
    as_of: datetime,
    announcements: list[CorporateAnnouncement] | None,
    evidence: list[AnnouncementSurpriseEvidence] | None,
) -> tuple[ProxyComponent, float | None]:
    missing: list[str] = []
    if not announcements:
        missing.append("no corporate announcements supplied")
    if not evidence:
        missing.append("no verified observed-versus-baseline surprise evidence")
    candidates = [
        item
        for item in (evidence or [])
        if item.company_id == company_id
        and item.effective_from <= as_of
        and item.published_at <= as_of
        and item.is_verified
        and (not announcements or any(a.announcement_id == item.announcement_id for a in announcements))
    ]
    if not candidates:
        return (
            ProxyComponent(
                name="announcement_surprise",
                score=None,
                original_weight=PROXY_WEIGHTS["announcement_surprise"],
                used_weight=0.0,
                available=False,
                metrics_used=[],
                observation_windows=[],
                periods_compared=[],
                missing_inputs=sorted(set(missing + ["no eligible verified evidence at as_of"])),
                explanation="Announcement surprise is unavailable; the current announcement schema has no defensible baseline.",
            ),
            None,
        )
    latest = max(candidates, key=lambda item: item.effective_from)
    relative_surprise = (latest.observed_value - latest.baseline_value) / abs(latest.baseline_value)
    score = _change_score(relative_surprise, PROXY_FULL_SCALE)
    return (
        ProxyComponent(
            name="announcement_surprise",
            score=score,
            original_weight=PROXY_WEIGHTS["announcement_surprise"],
            used_weight=PROXY_WEIGHTS["announcement_surprise"],
            available=True,
            metrics_used=["observed_value", "baseline_value"],
            observation_windows=[latest.announcement_id],
            periods_compared=[],
            missing_inputs=[],
            explanation=f"Verified observed-versus-baseline surprise: {relative_surprise:.2%}.",
        ),
        score,
    )


def _quality(*statuses: str) -> str:
    values = set(statuses)
    if values == {"synthetic"}:
        return "synthetic"
    if len(values) > 1 or "partial" in values:
        return "partial"
    return next(iter(values), "unverified")


def calculate_expectations_gap(
    financials: list[QuarterlyFinancial],
    prices: list[DailyPrice],
    as_of: datetime,
    *,
    prior_as_of: datetime | None = None,
    benchmark_prices: list[DailyPrice] | None = None,
    announcements: list[CorporateAnnouncement] | None = None,
    surprise_evidence: list[AnnouncementSurpriseEvidence] | None = None,
) -> ExpectationsGapResult:
    """Calculate the point-in-time Expectations Gap Proxy for one company."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include timezone information")
    prior_as_of = prior_as_of or as_of - timedelta(days=90)
    if prior_as_of.tzinfo is None or prior_as_of.utcoffset() is None:
        raise ValueError("prior_as_of must include timezone information")
    if prior_as_of >= as_of:
        raise ValueError("prior_as_of must precede as_of")
    if not financials or not prices:
        raise ValueError("financials and prices are required")
    company_id = financials[0].company_id
    if any(row.company_id != company_id for row in financials):
        raise ValueError("financial records must belong to one company")
    if any(price.company_id != company_id for price in prices):
        raise ValueError("price records must belong to one company")

    current = calculate_business_momentum(financials, as_of)
    prior = calculate_business_momentum(financials, prior_as_of)
    fundamental_score: float | None = None
    fundamental_missing: list[str] = []
    momentum_change: float | None = None
    if current.business_momentum_score is None or prior.business_momentum_score is None:
        fundamental_missing.append("current and prior Business Momentum Scores")
    else:
        momentum_change = round(current.business_momentum_score - prior.business_momentum_score, 6)
        fundamental_score = _change_score(momentum_change, 50.0)
    fundamental_component = ProxyComponent(
        name="fundamental_improvement",
        score=fundamental_score,
        original_weight=PROXY_WEIGHTS["fundamental_improvement"],
        used_weight=PROXY_WEIGHTS["fundamental_improvement"] if fundamental_score is not None else 0.0,
        available=fundamental_score is not None,
        metrics_used=["business_momentum_score"],
        observation_windows=[],
        periods_compared=[prior_as_of.isoformat(), as_of.isoformat()],
        missing_inputs=fundamental_missing,
        explanation=(
            f"Business Momentum changed by {momentum_change:.2f} points; improvement score: {fundamental_score:.2f}."
            if fundamental_score is not None
            else "Fundamental improvement is unavailable because one point-in-time momentum score is unavailable."
        ),
    )
    price_component, price_score, price_coverage = _price_component(prices, as_of, benchmark_prices)
    announcement_component, announcement_score = _announcement_component(
        company_id, as_of, announcements, surprise_evidence
    )
    components = {
        "fundamental_improvement": fundamental_component,
        "price_recognition": price_component,
        "announcement_surprise": announcement_component,
    }
    available_weight = sum(component.original_weight for component in components.values() if component.available)
    coverage_ratio = round(available_weight / 100.0, 6)
    used_weights = {
        name: round(component.original_weight / available_weight * 100.0, 6)
        for name, component in components.items()
        if component.available and available_weight
    }
    for name, component in components.items():
        component.used_weight = used_weights.get(name, 0.0)

    status: ProxyStatus
    if current.signal_status == "stale_data" or prior.signal_status == "stale_data":
        status = "stale_data"
    elif fundamental_score is None or price_score is None:
        status = "insufficient_data"
    else:
        status = "valid"
    score: float | None = None
    if status == "valid" and available_weight:
        contributions = {
            "fundamental_improvement": fundamental_score,
            "price_recognition": 100.0 - price_score,
            "announcement_surprise": announcement_score,
        }
        score = round(
            sum(contributions[name] * component.original_weight for name, component in components.items() if component.available)
            / available_weight,
            6,
        )
    quality = _quality(
        current.data_quality_status,
        prior.data_quality_status,
        *(row.data_quality_status for row in prices),
        *(row.data_quality_status for row in (benchmark_prices or [])),
        *(row.data_quality_status for row in (announcements or [])),
        *(item.data_quality_status for item in (surprise_evidence or [])),
    )
    source_confidence = min(current.confidence_score, prior.confidence_score, price_coverage)
    confidence = round(coverage_ratio * source_confidence, 6)
    missing = sorted(
        set(
            fundamental_missing
            + price_component.missing_inputs
            + announcement_component.missing_inputs
        )
    )
    price_inputs = price_component.observation_windows
    why = (
        f"Proxy score is {score:.2f}: fundamental improvement contributes positively, while price recognition "
        f"is subtracted through its {100.0 - price_score:.2f} non-recognition contribution."
        if score is not None
        else f"No proxy score assigned because signal status is {status}."
    )
    if not benchmark_prices:
        why += " Price recognition uses absolute returns because no defensible benchmark was supplied."
    if quality == "synthetic":
        why += " Inputs are synthetic and do not establish trading profitability."
    explanation = ExpectationsGapExplanation(
        component_weights=PROXY_WEIGHTS,
        reweighted_component_weights=used_weights,
        business_momentum_change=momentum_change,
        momentum_comparison_dates=[prior_as_of.isoformat(), as_of.isoformat()],
        price_recognition_inputs=price_inputs,
        price_observation_windows=price_component.observation_windows,
        announcement_surprise_evidence=announcement_component.observation_windows,
        missing_data=missing,
        data_coverage=coverage_ratio,
        confidence=confidence,
        signal_status=status,
        data_quality_status=quality,
        why=why,
    )
    return ExpectationsGapResult(
        company_id=company_id,
        as_of=as_of,
        prior_as_of=prior_as_of,
        expectations_gap_score=score,
        fundamental_improvement_score=fundamental_score,
        price_recognition_score=price_score,
        announcement_surprise_score=announcement_score,
        current_momentum=current,
        prior_momentum=prior,
        components=components,
        coverage_ratio=coverage_ratio,
        confidence_score=confidence,
        signal_status=status,
        data_quality_status=quality,
        explanation=explanation,
    )
