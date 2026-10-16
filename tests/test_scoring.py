from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from leader_x.models import QuarterlyFinancial
from leader_x.scoring import calculate_business_momentum

IST = timezone(timedelta(hours=5, minutes=30))
AS_OF = datetime(2025, 1, 31, 12, tzinfo=IST)


def financial(
    quarter: int,
    *,
    revenue: str = "100",
    revenue_yoy: str | None = "10",
    margin: str | None = "15",
    pat_yoy: str | None = "10",
    cash_flow: str | None = "20",
    free_cash_flow: str | None = "15",
    debt: str | None = "100",
    net_debt_ebitda: str | None = "3",
    effective_from: datetime | None = None,
    data_quality: str = "synthetic",
) -> QuarterlyFinancial:
    period_end = {
        1: date(2024, 3, 31),
        2: date(2024, 6, 30),
        3: date(2024, 9, 30),
        4: date(2024, 12, 31),
    }[quarter]
    default_reported_at = datetime.combine(period_end + timedelta(days=20), datetime.min.time(), IST)
    effective = effective_from or default_reported_at + timedelta(days=1)
    reported_at = min(default_reported_at, effective - timedelta(hours=1))
    return QuarterlyFinancial(
        financial_id=f"alpha-2024q{quarter}",
        company_id="alpha",
        fiscal_year=2024,
        fiscal_quarter=f"Q{quarter}",
        period_end_date=period_end,
        reported_at=reported_at,
        effective_from=effective,
        revenue=Decimal(revenue),
        revenue_yoy_pct=Decimal(revenue_yoy) if revenue_yoy is not None else None,
        ebitda=Decimal(revenue) * Decimal(margin) / 100 if margin is not None else None,
        ebitda_margin_pct=Decimal(margin) if margin is not None else None,
        profit_after_tax=Decimal("8"),
        pat_yoy_pct=Decimal(pat_yoy) if pat_yoy is not None else None,
        eps=Decimal("0.8"),
        operating_cash_flow=Decimal(cash_flow) if cash_flow is not None else None,
        free_cash_flow=Decimal(free_cash_flow) if free_cash_flow is not None else None,
        debt=Decimal(debt) if debt is not None else None,
        net_debt_to_ebitda=Decimal(net_debt_ebitda) if net_debt_ebitda is not None else None,
        source="synthetic-test",
        data_quality_status=data_quality,
    )


def test_improving_history_scores_above_neutral_and_has_explanation():
    rows = [
        financial(1, revenue_yoy="5", margin="12", pat_yoy="5", cash_flow="10", debt="120"),
        financial(2, revenue_yoy="15", margin="18", pat_yoy="15", cash_flow="25", debt="90"),
    ]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.signal_status == "valid"
    assert result.business_momentum_score is not None
    assert result.business_momentum_score > 50
    assert result.coverage_ratio == 1
    assert result.confidence_score > 0
    assert result.explanation.periods_compared
    assert result.explanation.metrics_improved
    assert result.explanation.component_weights_used["revenue_trend"] == 35
    assert result.data_quality_status == "synthetic"


def test_deteriorating_history_scores_below_neutral():
    rows = [
        financial(1, revenue_yoy="20", margin="20", pat_yoy="20", cash_flow="30", debt="80"),
        financial(2, revenue_yoy="-10", margin="10", pat_yoy="-5", cash_flow="10", debt="140"),
    ]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.business_momentum_score is not None
    assert result.business_momentum_score < 50
    assert result.explanation.metrics_deteriorated


def test_stable_history_is_deterministic_and_near_neutral():
    rows = [financial(1), financial(2)]
    first = calculate_business_momentum(rows, AS_OF)
    second = calculate_business_momentum(rows, AS_OF)
    assert first.model_dump() == second.model_dump()
    assert first.business_momentum_score == pytest.approx(50)


def test_partial_history_reweights_available_components_and_penalizes_confidence():
    rows = [
        financial(1, cash_flow=None, free_cash_flow=None, debt=None, net_debt_ebitda=None),
        financial(2, cash_flow=None, free_cash_flow=None, debt=None, net_debt_ebitda=None),
    ]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.signal_status == "valid"
    assert result.component_scores["cash_flow_trend"].available is False
    assert result.component_scores["balance_sheet_trend"].available is False
    assert result.coverage_ratio == pytest.approx(0.7)
    assert sum(result.explanation.component_weights_used.values()) == pytest.approx(100)
    assert result.confidence_score < result.coverage_ratio
    assert result.explanation.missing_inputs


def test_missing_quarter_does_not_create_a_comparable_pair():
    rows = [financial(1, revenue_yoy="5"), financial(3, revenue_yoy="25")]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.component_scores["revenue_trend"].available is False
    assert result.component_scores["profitability_trend"].available is False
    assert result.business_momentum_score is None
    assert result.signal_status == "insufficient_data"


def test_one_quarter_is_insufficient_data():
    result = calculate_business_momentum([financial(2)], AS_OF)
    assert result.business_momentum_score is None
    assert result.signal_status == "insufficient_data"
    assert result.confidence_score == 0


def test_stale_financials_are_not_scored():
    old = datetime(2023, 1, 1, tzinfo=IST)
    rows = [financial(1, effective_from=old), financial(2, effective_from=old + timedelta(days=1))]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.signal_status == "stale_data"
    assert result.business_momentum_score is None
    assert result.confidence_score == 0


def test_future_report_cannot_influence_earlier_score():
    base = [financial(1, revenue_yoy="5", margin="12"), financial(2, revenue_yoy="10", margin="14")]
    future = financial(
        3,
        revenue_yoy="999",
        margin="99",
        cash_flow="999",
        debt="1",
        effective_from=AS_OF + timedelta(days=1),
    )
    without_future = calculate_business_momentum(base, AS_OF)
    with_future = calculate_business_momentum(base + [future], AS_OF)
    assert with_future.model_dump() == without_future.model_dump()


def test_outlier_and_zero_denominator_are_safe_and_scores_are_bounded():
    rows = [
        financial(1, revenue="0", revenue_yoy=None, margin="1", cash_flow="0", debt="100"),
        financial(2, revenue="1000000", revenue_yoy=None, margin="99", cash_flow="1000000", debt="1"),
    ]
    result = calculate_business_momentum(rows, AS_OF)
    assert result.business_momentum_score is not None
    assert 0 <= result.business_momentum_score <= 100
    for component in result.component_scores.values():
        if component.score is not None:
            assert 0 <= component.score <= 100
