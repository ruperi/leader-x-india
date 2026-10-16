"""Validated domain models for the LEADER-X India data foundation."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DataQuality = Literal["synthetic", "real", "partial", "unverified"]


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return value


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str = Field(min_length=1)
    data_quality_status: DataQuality


class Company(DomainModel):
    company_id: str = Field(pattern=r"^[a-z0-9-]+$", min_length=3)
    ticker: str = Field(pattern=r"^SYN_[A-Z]+$", min_length=5)
    company_name: str = Field(min_length=3)
    exchange: Literal["SYNTHETIC"] = "SYNTHETIC"
    sector: str = Field(min_length=2)
    listing_date: date
    delisting_date: date | None = None
    is_active: bool = True

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "Company":
        if self.delisting_date and self.delisting_date < self.listing_date:
            raise ValueError("delisting_date cannot precede listing_date")
        if self.delisting_date and self.is_active:
            raise ValueError("company with delisting_date cannot be active")
        return self


class QuarterlyFinancial(DomainModel):
    financial_id: str = Field(min_length=3)
    company_id: str = Field(min_length=3)
    fiscal_year: int = Field(ge=2000, le=2100)
    fiscal_quarter: Literal["Q1", "Q2", "Q3", "Q4"]
    period_end_date: date
    reported_at: datetime
    effective_from: datetime
    revenue: Decimal | None = Field(default=None, ge=0)
    revenue_yoy_pct: Decimal | None = None
    ebitda: Decimal | None = Field(default=None, ge=0)
    ebitda_margin_pct: Decimal | None = None
    profit_after_tax: Decimal | None = Field(default=None, ge=0)
    pat_yoy_pct: Decimal | None = None
    eps: Decimal | None = None
    operating_cash_flow: Decimal | None = None
    free_cash_flow: Decimal | None = None
    debt: Decimal | None = Field(default=None, ge=0)
    net_debt_to_ebitda: Decimal | None = Field(default=None, ge=0)

    _reported_timezone = field_validator("reported_at", "effective_from")(_require_aware)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "QuarterlyFinancial":
        if self.effective_from < self.reported_at:
            raise ValueError("effective_from cannot precede reported_at")
        return self


class DailyPrice(DomainModel):
    company_id: str = Field(min_length=3)
    trade_date: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    adjusted_close: Decimal | None = Field(default=None, gt=0)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "DailyPrice":
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC values are inconsistent")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")
        return self


class CorporateAnnouncement(DomainModel):
    announcement_id: str = Field(min_length=3)
    company_id: str = Field(min_length=3)
    announcement_type: Literal["results", "order", "guidance", "acquisition", "management"]
    headline: str = Field(min_length=5)
    event_date: date
    published_at: datetime
    effective_from: datetime
    source_name: str = Field(min_length=1)
    source_url: str | None = None
    is_verified: bool = False

    _published_timezone = field_validator("published_at", "effective_from")(_require_aware)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "CorporateAnnouncement":
        if self.effective_from < self.published_at:
            raise ValueError("effective_from cannot precede published_at")
        return self
