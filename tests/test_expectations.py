from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from leader_x.expectations import AnnouncementSurpriseEvidence, calculate_expectations_gap
from leader_x.models import CorporateAnnouncement, DailyPrice, QuarterlyFinancial

IST = timezone(timedelta(hours=5, minutes=30))
PRIOR_AS_OF = datetime(2024, 8, 1, 12, tzinfo=IST)
AS_OF = datetime(2025, 1, 31, 12, tzinfo=IST)


def financial(
    quarter: int,
    *,
    revenue_yoy: str,
    margin: str,
    pat_yoy: str,
    cash_flow: str,
    debt: str,
    effective_from: datetime | None = None,
) -> QuarterlyFinancial:
    period_end = {
        1: date(2024, 3, 31),
        2: date(2024, 6, 30),
        3: date(2024, 9, 30),
    }[quarter]
    default_reported = datetime.combine(period_end + timedelta(days=20), datetime.min.time(), IST)
    effective = effective_from or default_reported + timedelta(days=1)
    reported = min(default_reported, effective - timedelta(hours=1))
    revenue = Decimal("100") + Decimal(quarter * 10)
    margin_decimal = Decimal(margin)
    ebitda = revenue * margin_decimal / Decimal("100")
    return QuarterlyFinancial(
        financial_id=f"alpha-2024q{quarter}",
        company_id="alpha",
        fiscal_year=2024,
        fiscal_quarter=f"Q{quarter}",
        period_end_date=period_end,
        reported_at=reported,
        effective_from=effective,
        revenue=revenue,
        revenue_yoy_pct=Decimal(revenue_yoy),
        ebitda=ebitda,
        ebitda_margin_pct=margin_decimal,
        profit_after_tax=Decimal("8"),
        pat_yoy_pct=Decimal(pat_yoy),
        eps=Decimal("0.8"),
        operating_cash_flow=Decimal(cash_flow),
        free_cash_flow=Decimal(cash_flow) - Decimal("3"),
        debt=Decimal(debt),
        net_debt_to_ebitda=Decimal("3"),
        source="synthetic-expectations-test",
        data_quality_status="synthetic",
    )


def prices(values: dict[date, str], company_id: str = "alpha") -> list[DailyPrice]:
    result = []
    for trade_date, value in values.items():
        close = Decimal(value)
        result.append(
            DailyPrice(
                company_id=company_id,
                trade_date=trade_date,
                open=close,
                high=close + Decimal("1"),
                low=close - Decimal("1"),
                close=close,
                adjusted_close=close,
                volume=1000,
                source="synthetic-expectations-test",
                data_quality_status="synthetic",
            )
        )
    return result


PRICE_DATES = {
    date(2024, 11, 1): "100",
    date(2024, 12, 31): "100",
    date(2025, 1, 31): "101",
}


def improving_financials() -> list[QuarterlyFinancial]:
    return [
        financial(1, revenue_yoy="5", margin="12", pat_yoy="5", cash_flow="10", debt="120"),
        financial(2, revenue_yoy="10", margin="14", pat_yoy="10", cash_flow="20", debt="100"),
        financial(3, revenue_yoy="20", margin="18", pat_yoy="20", cash_flow="30", debt="80"),
    ]


def weakening_financials() -> list[QuarterlyFinancial]:
    return [
        financial(1, revenue_yoy="20", margin="20", pat_yoy="20", cash_flow="30", debt="80"),
        financial(2, revenue_yoy="10", margin="15", pat_yoy="10", cash_flow="20", debt="100"),
        financial(3, revenue_yoy="-10", margin="10", pat_yoy="-5", cash_flow="10", debt="140"),
    ]


def test_muted_price_recognition_produces_a_high_proxy_score():
    result = calculate_expectations_gap(
        improving_financials(), prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF
    )
    assert result.signal_status == "valid"
    assert result.expectations_gap_score is not None
    assert result.expectations_gap_score > 50
    assert result.fundamental_improvement_score > 50
    assert result.price_recognition_score < 60
    assert result.announcement_surprise_score is None
    assert result.data_quality_status == "synthetic"
    assert "absolute returns" in result.explanation.why


def test_improvement_already_reflected_in_price_reduces_proxy_score():
    reflected_prices = prices(
        {
            date(2024, 11, 1): "100",
            date(2024, 12, 31): "140",
            date(2025, 1, 31): "150",
        }
    )
    muted = calculate_expectations_gap(
        improving_financials(), prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF
    )
    reflected = calculate_expectations_gap(
        improving_financials(), reflected_prices, AS_OF, prior_as_of=PRIOR_AS_OF
    )
    assert reflected.price_recognition_score > muted.price_recognition_score
    assert reflected.expectations_gap_score < muted.expectations_gap_score


def test_weakening_fundamentals_with_temporary_price_spike_is_not_a_positive_gap():
    spike_prices = prices(
        {
            date(2024, 11, 1): "100",
            date(2024, 12, 31): "100",
            date(2025, 1, 31): "180",
        }
    )
    result = calculate_expectations_gap(
        weakening_financials(), spike_prices, AS_OF, prior_as_of=PRIOR_AS_OF
    )
    assert result.signal_status == "valid"
    assert result.fundamental_improvement_score < 50
    assert result.price_recognition_score > 90
    assert result.expectations_gap_score < 50


def test_missing_announcements_are_unavailable_and_weights_reweight():
    result = calculate_expectations_gap(
        improving_financials(), prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF, announcements=[]
    )
    assert result.signal_status == "valid"
    assert result.components["announcement_surprise"].available is False
    assert result.coverage_ratio == pytest.approx(0.8)
    assert sum(result.explanation.reweighted_component_weights.values()) == pytest.approx(100)
    assert result.confidence_score < result.coverage_ratio
    assert result.explanation.missing_data


def test_insufficient_history_is_not_ranked_as_valid():
    result = calculate_expectations_gap(
        improving_financials()[:1], prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF
    )
    assert result.signal_status == "insufficient_data"
    assert result.expectations_gap_score is None
    assert result.fundamental_improvement_score is None


def test_future_financials_and_prices_cannot_change_earlier_proxy():
    base = calculate_expectations_gap(
        improving_financials(), prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF
    )
    future_report = financial(
        3,
        revenue_yoy="999",
        margin="99",
        pat_yoy="999",
        cash_flow="999",
        debt="1",
        effective_from=AS_OF + timedelta(days=1),
    )
    future_price = prices({date(2025, 2, 3): "10000"})
    with_future = calculate_expectations_gap(
        improving_financials() + [future_report],
        prices(PRICE_DATES) + future_price,
        AS_OF,
        prior_as_of=PRIOR_AS_OF,
    )
    assert with_future.model_dump() == base.model_dump()


def test_benchmark_relative_returns_are_used_when_available():
    benchmark = prices(
        {
            date(2024, 11, 1): "100",
            date(2024, 12, 31): "100",
            date(2025, 1, 31): "100",
        },
        company_id="benchmark",
    )
    result = calculate_expectations_gap(
        improving_financials(), prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF, benchmark_prices=benchmark
    )
    assert result.signal_status == "valid"
    assert result.explanation.price_recognition_inputs
    assert "relative to" in result.components["price_recognition"].explanation


def test_verified_announcement_evidence_enables_surprise_component():
    announcement = CorporateAnnouncement(
        announcement_id="event-alpha-1",
        company_id="alpha",
        announcement_type="results",
        headline="Synthetic results baseline",
        event_date=date(2024, 12, 1),
        published_at=datetime(2024, 12, 1, 18, tzinfo=IST),
        effective_from=datetime(2024, 12, 2, tzinfo=IST),
        source_name="synthetic-test",
        source="synthetic-expectations-test",
        data_quality_status="synthetic",
    )
    evidence = AnnouncementSurpriseEvidence(
        announcement_id=announcement.announcement_id,
        company_id="alpha",
        observed_value=120,
        baseline_value=100,
        published_at=announcement.published_at,
        effective_from=announcement.effective_from,
        source_name="synthetic-test",
        is_verified=True,
        data_quality_status="synthetic",
    )
    result = calculate_expectations_gap(
        improving_financials(),
        prices(PRICE_DATES),
        AS_OF,
        prior_as_of=PRIOR_AS_OF,
        announcements=[announcement],
        surprise_evidence=[evidence],
    )
    assert result.signal_status == "valid"
    assert result.announcement_surprise_score > 50
    assert result.coverage_ratio == pytest.approx(1)
    assert result.explanation.announcement_surprise_evidence == [announcement.announcement_id]


def test_future_announcement_evidence_is_excluded():
    future = AnnouncementSurpriseEvidence(
        announcement_id="future-alpha-1",
        company_id="alpha",
        observed_value=999,
        baseline_value=1,
        published_at=AS_OF + timedelta(days=1),
        effective_from=AS_OF + timedelta(days=2),
        source_name="synthetic-test",
        is_verified=True,
        data_quality_status="synthetic",
    )
    result = calculate_expectations_gap(
        improving_financials(),
        prices(PRICE_DATES),
        AS_OF,
        prior_as_of=PRIOR_AS_OF,
        surprise_evidence=[future],
    )
    assert result.announcement_surprise_score is None
    assert result.components["announcement_surprise"].available is False


def test_score_is_deterministic_and_bounded():
    rows = improving_financials()
    first = calculate_expectations_gap(rows, prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF)
    second = calculate_expectations_gap(rows, prices(PRICE_DATES), AS_OF, prior_as_of=PRIOR_AS_OF)
    assert first.model_dump() == second.model_dump()
    assert 0 <= first.expectations_gap_score <= 100
    for component in first.components.values():
        if component.score is not None:
            assert 0 <= component.score <= 100
