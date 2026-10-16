"""Data loading and view-model helpers for the offline Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .expectations import ExpectationsGapResult, calculate_expectations_gap
from .models import Company, CorporateAnnouncement, DailyPrice, QuarterlyFinancial

IST = timezone(timedelta(hours=5, minutes=30))
DEMO_AS_OF = datetime(2024, 12, 31, 23, 59, tzinfo=IST)
DEMO_PRIOR_AS_OF = DEMO_AS_OF - timedelta(days=92)


@dataclass(frozen=True)
class DashboardDataset:
    companies: list[Company]
    financials: list[QuarterlyFinancial]
    prices: list[DailyPrice]
    announcements: list[CorporateAnnouncement]


@dataclass(frozen=True)
class CompanySnapshot:
    company: Company
    result: ExpectationsGapResult


def _read_models(path: Path, model_type: type) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    return [model_type.model_validate(row) for row in pq.read_table(path).to_pylist()]


def load_dataset(data_dir: Path) -> DashboardDataset:
    """Load the existing offline Parquet dataset through the domain models."""
    return DashboardDataset(
        companies=_read_models(data_dir / "companies.parquet", Company),
        financials=_read_models(data_dir / "quarterly_financials.parquet", QuarterlyFinancial),
        prices=_read_models(data_dir / "daily_prices.parquet", DailyPrice),
        announcements=_read_models(
            data_dir / "corporate_announcements.parquet", CorporateAnnouncement
        ),
    )


def build_snapshots(
    dataset: DashboardDataset,
    as_of: datetime = DEMO_AS_OF,
    prior_as_of: datetime = DEMO_PRIOR_AS_OF,
) -> list[CompanySnapshot]:
    """Calculate one Phase 4 snapshot per company without duplicating scoring logic."""
    financials_by_company: dict[str, list[QuarterlyFinancial]] = {}
    prices_by_company: dict[str, list[DailyPrice]] = {}
    announcements_by_company: dict[str, list[CorporateAnnouncement]] = {}
    for row in dataset.financials:
        financials_by_company.setdefault(row.company_id, []).append(row)
    for row in dataset.prices:
        prices_by_company.setdefault(row.company_id, []).append(row)
    for row in dataset.announcements:
        announcements_by_company.setdefault(row.company_id, []).append(row)

    snapshots: list[CompanySnapshot] = []
    for company in dataset.companies:
        company_financials = financials_by_company.get(company.company_id, [])
        company_prices = prices_by_company.get(company.company_id, [])
        company_announcements = announcements_by_company.get(company.company_id, [])
        if not company_financials or not company_prices:
            continue
        result = calculate_expectations_gap(
            company_financials,
            company_prices,
            as_of,
            prior_as_of=prior_as_of,
            announcements=company_announcements,
        )
        snapshots.append(CompanySnapshot(company=company, result=result))
    return snapshots


def rank_snapshots(
    snapshots: list[CompanySnapshot],
    *,
    min_gap_score: float = 0.0,
    statuses: set[str] | None = None,
    data_quality: set[str] | None = None,
) -> list[CompanySnapshot]:
    """Filter and sort snapshots, keeping missing scores at the bottom."""
    filtered = [
        snapshot
        for snapshot in snapshots
        if (snapshot.result.expectations_gap_score or 0) >= min_gap_score
        and (not statuses or snapshot.result.signal_status in statuses)
        and (not data_quality or snapshot.result.data_quality_status in data_quality)
    ]
    return sorted(
        filtered,
        key=lambda snapshot: (
            snapshot.result.expectations_gap_score is not None,
            snapshot.result.expectations_gap_score or -1,
            snapshot.result.fundamental_improvement_score or -1,
        ),
        reverse=True,
    )


def display_number(value: float | None, suffix: str = "", decimals: int = 1) -> str:
    """Format missing numeric values as an em dash rather than zero."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def display_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def discovery_frame(snapshots: list[CompanySnapshot]) -> pd.DataFrame:
    rows = []
    for rank, snapshot in enumerate(snapshots, start=1):
        result = snapshot.result
        rows.append(
            {
                "Rank": rank,
                "Ticker": snapshot.company.ticker,
                "Company": snapshot.company.company_name,
                "Momentum": display_number(result.current_momentum.business_momentum_score),
                "Gap proxy": display_number(result.expectations_gap_score),
                "Confidence": display_percent(result.confidence_score),
                "Coverage": display_percent(result.coverage_ratio),
                "Status": result.signal_status,
                "Data": result.data_quality_status,
                "Why": result.explanation.why,
            }
        )
    return pd.DataFrame(rows)


def price_frame(prices: list[DailyPrice]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Date": row.trade_date, "Close": float(row.adjusted_close or row.close)} for row in prices]
    ).sort_values("Date")


def financial_frame(financials: list[QuarterlyFinancial]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Period": f"{row.fiscal_year} {row.fiscal_quarter}",
                "Period end": row.period_end_date,
                "Revenue": float(row.revenue) if row.revenue is not None else None,
                "Revenue YoY %": float(row.revenue_yoy_pct) if row.revenue_yoy_pct is not None else None,
                "EBITDA margin %": float(row.ebitda_margin_pct)
                if row.ebitda_margin_pct is not None
                else None,
                "PAT YoY %": float(row.pat_yoy_pct) if row.pat_yoy_pct is not None else None,
                "Operating cash flow": float(row.operating_cash_flow)
                if row.operating_cash_flow is not None
                else None,
                "Debt": float(row.debt) if row.debt is not None else None,
                "Effective from": row.effective_from,
            }
            for row in sorted(financials, key=lambda item: item.period_end_date)
        ]
    )


def announcement_frame(announcements: list[CorporateAnnouncement]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Event date": row.event_date,
                "Published": row.published_at,
                "Type": row.announcement_type,
                "Headline": row.headline,
                "Data": row.data_quality_status,
            }
            for row in sorted(announcements, key=lambda item: item.event_date, reverse=True)
        ]
    )


def component_frame(result: ExpectationsGapResult) -> pd.DataFrame:
    rows = []
    for component in result.components.values():
        rows.append(
            {
                "Component": component.name.replace("_", " ").title(),
                "Score": display_number(component.score),
                "Original weight": display_percent(component.original_weight / 100),
                "Used weight": display_percent(component.used_weight / 100),
                "Available": "Yes" if component.available else "No",
                "Explanation": component.explanation,
            }
        )
    return pd.DataFrame(rows)
