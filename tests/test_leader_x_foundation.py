from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from leader_x.models import Company, CorporateAnnouncement, DailyPrice, QuarterlyFinancial
from leader_x.synthetic import generate_companies, generate_dataset
from leader_x.validation import ensure_unique

IST = timezone(timedelta(hours=5, minutes=30))


def test_company_rejects_realistic_exchange_ticker_and_requires_synthetic_shape():
    with pytest.raises(ValidationError):
        Company(
            company_id="tcs",
            ticker="TCS",
            company_name="Not Synthetic",
            sector="Technology",
            listing_date=date(2022, 1, 1),
            source="test",
            data_quality_status="synthetic",
        )


def test_missing_financial_metrics_remain_missing():
    row = QuarterlyFinancial(
        financial_id="epsilon-2024q1",
        company_id="epsilon",
        fiscal_year=2024,
        fiscal_quarter="Q1",
        period_end_date=date(2024, 3, 31),
        reported_at=datetime(2024, 5, 1, 18, tzinfo=IST),
        effective_from=datetime(2024, 5, 2, 0, tzinfo=IST),
        revenue=Decimal("100"),
        ebitda_margin_pct=None,
        operating_cash_flow=None,
        source="test",
        data_quality_status="synthetic",
    )
    assert row.ebitda_margin_pct is None
    assert row.operating_cash_flow is None


def test_duplicate_announcement_ids_are_rejected():
    announcements = [
        CorporateAnnouncement(
            announcement_id="event-1",
            company_id="alpha",
            announcement_type="results",
            headline="Synthetic results update",
            event_date=date(2024, 4, 1),
            published_at=datetime(2024, 4, 1, 18, tzinfo=IST),
            effective_from=datetime(2024, 4, 2, tzinfo=IST),
            source_name="synthetic",
            source="synthetic",
            data_quality_status="synthetic",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate announcement_id"):
        ensure_unique(announcements + announcements, "announcement_id")


def test_timestamps_must_be_timezone_aware_and_ordered():
    with pytest.raises(ValidationError, match="timezone"):
        CorporateAnnouncement(
            announcement_id="event-2",
            company_id="alpha",
            announcement_type="results",
            headline="Synthetic results update",
            event_date=date(2024, 4, 1),
            published_at=datetime(2024, 4, 1, 18),
            effective_from=datetime(2024, 4, 2, 0, tzinfo=IST),
            source_name="synthetic",
            source="synthetic",
            data_quality_status="synthetic",
        )
    with pytest.raises(ValidationError, match="cannot precede"):
        CorporateAnnouncement(
            announcement_id="event-3",
            company_id="alpha",
            announcement_type="results",
            headline="Synthetic results update",
            event_date=date(2024, 4, 1),
            published_at=datetime(2024, 4, 2, 18, tzinfo=IST),
            effective_from=datetime(2024, 4, 2, 17, tzinfo=IST),
            source_name="synthetic",
            source="synthetic",
            data_quality_status="synthetic",
        )


def test_daily_price_ohlc_validation():
    with pytest.raises(ValidationError):
        DailyPrice(
            company_id="alpha",
            trade_date=date(2024, 1, 2),
            open=Decimal("10"),
            high=Decimal("9"),
            low=Decimal("8"),
            close=Decimal("10"),
            volume=100,
            source="test",
            data_quality_status="synthetic",
        )


def test_generated_dataset_is_fully_synthetic_and_has_required_patterns(tmp_path):
    output_dir = tmp_path / "synthetic"
    database_path = tmp_path / "leader_x.duckdb"
    counts = generate_dataset(output_dir, database_path)
    assert counts["companies"] == 6
    assert counts["quarterly_financials"] >= 72
    assert counts["daily_prices"] >= 3000
    assert counts["corporate_announcements"] == 6
    assert all(company.data_quality_status == "synthetic" for company in generate_companies())
    assert all(company.ticker.startswith("SYN_") for company in generate_companies())
    assert database_path.exists()
    assert all(path.suffix == ".parquet" for path in output_dir.iterdir())

    delta = next(company for company in generate_companies() if company.company_id == "delta")
    assert delta.delisting_date == date(2024, 12, 31)
    assert not delta.is_active
