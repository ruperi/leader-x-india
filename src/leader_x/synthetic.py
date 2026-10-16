"""Deterministic, clearly labelled synthetic LEADER-X demonstration data."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import Company, CorporateAnnouncement, DailyPrice, QuarterlyFinancial
from .storage import create_duckdb_from_parquet, write_records_to_parquet

IST = timezone(timedelta(hours=5, minutes=30))
SYNTHETIC_SOURCE = "leader-x-synthetic-generator-v1"

COMPANY_SPECS = [
    ("alpha", "SYN_ALPHA", "Asterline Components", "Industrial Manufacturing", None),
    ("beta", "SYN_BETA", "Bharat Meadow Foods", "Consumer Staples", None),
    ("gamma", "SYN_GAMMA", "Cedar Arc Systems", "Information Technology", None),
    ("delta", "SYN_DELTA", "Delta Loom Mobility", "Automotive", date(2024, 12, 31)),
    ("epsilon", "SYN_EPSILON", "Eastern Vale Healthcare", "Healthcare", None),
    ("zeta", "SYN_ZETA", "Zinnia Grid Services", "Utilities", None),
]


def _published_at(period_end: date, quarter_index: int, delayed: bool = False) -> datetime:
    days_after_period = 40 if delayed else 28
    return datetime.combine(period_end + timedelta(days=days_after_period), time(18, 0), IST)


def generate_companies() -> list[Company]:
    return [
        Company(
            company_id=company_id,
            ticker=ticker,
            company_name=name,
            sector=sector,
            listing_date=date(2022, 1, 3),
            delisting_date=delisting_date,
            is_active=delisting_date is None,
            source=SYNTHETIC_SOURCE,
            data_quality_status="synthetic",
        )
        for company_id, ticker, name, sector, delisting_date in COMPANY_SPECS
    ]


def _quarter_end(year: int, quarter: int) -> date:
    return [date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31)][quarter - 1]


def _quarter_rows(company: Company) -> list[QuarterlyFinancial]:
    rows: list[QuarterlyFinancial] = []
    quarter_count = 12 if company.company_id == "delta" else 16
    start_year, start_quarter = 2022, 1
    for index in range(quarter_count):
        quarter_number = start_quarter + index
        year = start_year + (quarter_number - 1) // 4
        quarter = ((quarter_number - 1) % 4) + 1
        period_end = _quarter_end(year, quarter)
        delayed = company.company_id in {"epsilon", "delta"} and index == 7
        reported_at = _published_at(period_end, index, delayed)
        effective_from = reported_at + timedelta(days=1 if reported_at.hour >= 15 else 0)

        if company.company_id == "alpha":
            revenue = 100 + index * 8
            margin = Decimal("14.0") + Decimal(index) * Decimal("0.35")
            pat_growth = Decimal("8.0") + Decimal(index) * Decimal("1.5")
            cash_flow = 16 + index * 2
        elif company.company_id == "beta":
            revenue = 120 + index * 7
            margin = Decimal("18.0") - Decimal(index) * Decimal("0.25")
            pat_growth = Decimal("12.0") - Decimal(index) * Decimal("0.8")
            cash_flow = None if index in {4, 5, 6} else 18 + index
        elif company.company_id == "gamma":
            revenue = 150 + index * 2
            margin = Decimal("22.0") + (Decimal("0.1") if index % 2 else Decimal("0"))
            pat_growth = Decimal("4.0")
            cash_flow = 24 + index
        elif company.company_id == "delta":
            revenue = 90 + index * 2
            margin = Decimal("12.0") + (Decimal("0.15") if index > 7 else Decimal("0"))
            pat_growth = Decimal("2.0")
            cash_flow = 10 + index if index != 3 else None
        elif company.company_id == "epsilon":
            revenue = 80 + index * 6
            margin = None if index in {2, 3, 8, 9} else Decimal("10.0") + Decimal(index) * Decimal("0.2")
            pat_growth = None if index in {8, 9} else Decimal("6.0") + Decimal(index)
            cash_flow = None
        else:
            revenue = 110 + index * 5
            margin = Decimal("16.0") + Decimal(index) * Decimal("0.2")
            pat_growth = Decimal("9.0") + Decimal(index) * Decimal("0.5")
            cash_flow = 14 + index * 2

        revenue_yoy = Decimal("5.0") + Decimal(index) * Decimal("0.8")
        ebitda = Decimal(revenue) * margin / Decimal("100") if margin is not None else None
        pat = Decimal(revenue) * (Decimal("0.06") + Decimal(index) / Decimal("1000"))
        debt = Decimal("80") - Decimal(index) * Decimal("2") if company.company_id == "alpha" else Decimal("100")
        rows.append(
            QuarterlyFinancial(
                financial_id=f"{company.company_id}-{year}q{quarter}",
                company_id=company.company_id,
                fiscal_year=year,
                fiscal_quarter=f"Q{quarter}",
                period_end_date=period_end,
                reported_at=reported_at,
                effective_from=effective_from,
                revenue=Decimal(revenue),
                revenue_yoy_pct=revenue_yoy,
                ebitda=ebitda,
                ebitda_margin_pct=margin,
                profit_after_tax=pat,
                pat_yoy_pct=pat_growth,
                eps=pat / Decimal("10"),
                operating_cash_flow=Decimal(cash_flow) if cash_flow is not None else None,
                free_cash_flow=Decimal(cash_flow - 3) if cash_flow is not None else None,
                debt=debt,
                net_debt_to_ebitda=debt / ebitda if ebitda else None,
                source=SYNTHETIC_SOURCE,
                data_quality_status="synthetic",
            )
        )
    return rows


def _price_rows(company: Company) -> list[DailyPrice]:
    rows: list[DailyPrice] = []
    start = date(2023, 1, 2)
    end = company.delisting_date or date(2025, 12, 31)
    base = {"alpha": 100, "beta": 130, "gamma": 160, "delta": 90, "epsilon": 75, "zeta": 115}[company.company_id]
    drift = {"alpha": 0.12, "beta": 0.08, "gamma": 0.22, "delta": -0.03, "epsilon": 0.04, "zeta": 0.1}[company.company_id]
    day_index = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            cycle = ((day_index % 11) - 5) * 0.08
            close = max(5.0, base + day_index * drift + cycle)
            open_price = close - 0.25
            high = close + 0.7
            low = max(1.0, open_price - 0.65)
            rows.append(
                DailyPrice(
                    company_id=company.company_id,
                    trade_date=current,
                    open=Decimal(str(round(open_price, 2))),
                    high=Decimal(str(round(high, 2))),
                    low=Decimal(str(round(low, 2))),
                    close=Decimal(str(round(close, 2))),
                    adjusted_close=Decimal(str(round(close, 2))),
                    volume=100_000 + day_index * 17,
                    source=SYNTHETIC_SOURCE,
                    data_quality_status="synthetic",
                )
            )
            day_index += 1
        current += timedelta(days=1)
    return rows


def _announcement_rows(companies: list[Company]) -> list[CorporateAnnouncement]:
    rows: list[CorporateAnnouncement] = []
    for index, company in enumerate(companies):
        event_date = date(2024, 4, 15) + timedelta(days=index * 11)
        published_at = datetime.combine(event_date, time(17, 30), IST)
        effective_from = published_at + timedelta(days=1)
        rows.append(
            CorporateAnnouncement(
                announcement_id=f"announcement-{company.company_id}-01",
                company_id=company.company_id,
                announcement_type="results",
                headline=f"{company.company_name} reports quarterly operating update",
                event_date=event_date,
                published_at=published_at,
                effective_from=effective_from,
                source_name=SYNTHETIC_SOURCE,
                source_url=None,
                is_verified=False,
                source=SYNTHETIC_SOURCE,
                data_quality_status="synthetic",
            )
        )
    return rows


def _records(items: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def generate_dataset(output_dir: Path, database_path: Path) -> dict[str, int]:
    companies = generate_companies()
    financials = [financial for company in companies for financial in _quarter_rows(company)]
    prices = [price for company in companies for price in _price_rows(company)]
    announcements = _announcement_rows(companies)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records_to_parquet(_records(companies), output_dir / "companies.parquet")
    write_records_to_parquet(_records(financials), output_dir / "quarterly_financials.parquet")
    write_records_to_parquet(_records(prices), output_dir / "daily_prices.parquet")
    write_records_to_parquet(_records(announcements), output_dir / "corporate_announcements.parquet")
    create_duckdb_from_parquet(database_path, output_dir)
    return {
        "companies": len(companies),
        "quarterly_financials": len(financials),
        "daily_prices": len(prices),
        "corporate_announcements": len(announcements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--database", type=Path, default=Path("data/leader_x.duckdb"))
    args = parser.parse_args()
    counts = generate_dataset(args.output_dir, args.database)
    print(f"Generated synthetic LEADER-X data: {counts}")


if __name__ == "__main__":
    main()
