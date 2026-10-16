"""LEADER-X India offline research dashboard."""

from pathlib import Path

import streamlit as st

from leader_x.dashboard import (
    DEMO_AS_OF,
    CompanySnapshot,
    DashboardDataset,
    announcement_frame,
    build_snapshots,
    component_frame,
    discovery_frame,
    display_number,
    display_percent,
    financial_frame,
    load_dataset,
    price_frame,
    rank_snapshots,
)

DATA_DIR = Path("data/synthetic")


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { --lx-ink: #f8fafc; --lx-muted: #94a3b8; --lx-bg: #020617;
                --lx-panel: #0f172a; --lx-border: #334155; --lx-green: #22c55e; }
        .stApp { background: linear-gradient(135deg, #020617 0%, #0b1224 55%, #111827 100%); }
        [data-testid="stHeader"] { background: rgba(2, 6, 23, 0.82); }
        [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1e293b; }
        .lx-kicker { color: #22c55e; font-family: monospace; letter-spacing: .12em;
                     text-transform: uppercase; font-size: .72rem; font-weight: 700; }
        .lx-title { color: #f8fafc; font-size: 2.4rem; font-weight: 700; letter-spacing: -.04em;
                    margin: .25rem 0 .35rem; }
        .lx-subtitle { color: #cbd5e1; max-width: 780px; line-height: 1.55; }
        .lx-badge { display: inline-block; background: rgba(34,197,94,.12); color: #86efac;
                    border: 1px solid rgba(34,197,94,.38); border-radius: 999px; padding: .25rem .7rem;
                    font-size: .75rem; font-weight: 700; letter-spacing: .04em; }
        .lx-panel { background: rgba(15,23,42,.78); border: 1px solid #334155; border-radius: 14px;
                    padding: 1rem 1.15rem; margin: .5rem 0 1rem; }
        .lx-muted { color: #94a3b8; font-size: .9rem; }
        .lx-warning { background: rgba(245,158,11,.1); border-left: 3px solid #f59e0b;
                      color: #fde68a; padding: .8rem 1rem; border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.markdown('<div class="lx-kicker">Frontier Buildathon 2026 · Research workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="lx-title">LEADER-X India</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lx-subtitle">Find improving businesses before price recognition catches up — with every signal, timestamp and limitation visible.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="lx-badge">SYNTHETIC DEMONSTRATION · NOT REAL MARKET DATA</span>', unsafe_allow_html=True)


def _sidebar(snapshots: list[CompanySnapshot]) -> str:
    st.sidebar.markdown("## LEADER-X India")
    st.sidebar.caption("Offline research demo")
    st.sidebar.markdown("### Navigate")
    page = st.sidebar.radio("Page", ["Stock Discovery", "Company Research", "Evidence & Risk"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("**As-of date**")
    st.sidebar.code(DEMO_AS_OF.strftime("%d %b %Y %H:%M IST"), language=None)
    st.sidebar.caption("All scores use only information available by this timestamp.")
    st.sidebar.markdown("---")
    st.sidebar.warning("Synthetic data only. This proxy does not establish profitable trades.")
    st.sidebar.caption(f"{len(snapshots)} fictional companies loaded from Parquet")
    return page


def _metric_row(result) -> None:
    cards = st.columns(4)
    cards[0].metric("Business momentum", display_number(result.current_momentum.business_momentum_score))
    cards[1].metric("Expectations gap proxy", display_number(result.expectations_gap_score))
    cards[2].metric("Confidence", display_percent(result.confidence_score))
    cards[3].metric("Coverage", display_percent(result.coverage_ratio))


def _stock_discovery(snapshots: list[CompanySnapshot]) -> None:
    st.subheader("Stock Discovery")
    st.caption("Ranked research candidates — not buy or sell recommendations.")
    controls = st.columns([1.2, 1, 1])
    min_score = controls[0].slider("Minimum gap proxy", 0, 100, 0, 5)
    statuses = sorted({snapshot.result.signal_status for snapshot in snapshots})
    selected_statuses = controls[1].multiselect("Signal status", statuses, default=statuses)
    qualities = sorted({snapshot.result.data_quality_status for snapshot in snapshots})
    selected_quality = controls[2].multiselect("Data quality", qualities, default=qualities)
    ranked = rank_snapshots(
        snapshots,
        min_gap_score=min_score,
        statuses=set(selected_statuses),
        data_quality=set(selected_quality),
    )
    valid = sum(snapshot.result.signal_status == "valid" for snapshot in snapshots)
    cards = st.columns(3)
    cards[0].metric("Companies", len(snapshots))
    cards[1].metric("Valid signals", valid)
    cards[2].metric("Visible candidates", len(ranked))
    st.dataframe(
        discovery_frame(ranked),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Why": st.column_config.TextColumn("Why this rank", width="large"),
            "Rank": st.column_config.NumberColumn("#", width="small"),
        },
    )
    if not ranked:
        st.info("No companies match these filters. Missing scores remain excluded from scored filters.")


def _records_for_company(dataset: DashboardDataset, company_id: str, as_of) -> tuple[list, list, list]:
    financials = [
        row for row in dataset.financials if row.company_id == company_id and row.effective_from <= as_of
    ]
    prices = [row for row in dataset.prices if row.company_id == company_id and row.trade_date <= as_of.date()]
    announcements = [
        row
        for row in dataset.announcements
        if row.company_id == company_id and row.published_at <= as_of and row.effective_from <= as_of
    ]
    return financials, prices, announcements


def _company_research(dataset: DashboardDataset, snapshots: list[CompanySnapshot]) -> None:
    st.subheader("Company Research")
    options = {snapshot.company.ticker: snapshot for snapshot in snapshots}
    selected_ticker = st.selectbox("Select a fictional company", list(options))
    snapshot = options[selected_ticker]
    financials, prices, announcements = _records_for_company(dataset, snapshot.company.company_id, DEMO_AS_OF)
    result = snapshot.result
    st.markdown(
        f"### {snapshot.company.ticker} · {snapshot.company.company_name}\n"
        f"<span class='lx-muted'>{snapshot.company.sector} · {result.data_quality_status.upper()} · as of {DEMO_AS_OF.date()}</span>",
        unsafe_allow_html=True,
    )
    _metric_row(result)
    if result.signal_status != "valid":
        st.warning(f"This company is not a valid ranked signal: {result.signal_status}.")

    st.markdown("#### Signal decomposition")
    st.dataframe(component_frame(result), use_container_width=True, hide_index=True)
    st.markdown("#### Why this company is ranked this way")
    st.markdown(f'<div class="lx-panel">{result.explanation.why}</div>', unsafe_allow_html=True)
    if result.explanation.missing_data:
        st.warning("Missing or unavailable inputs: " + "; ".join(result.explanation.missing_data))
    st.caption(
        f"Momentum comparison: {result.prior_as_of.date()} → {result.as_of.date()} · "
        f"Fundamental improvement: {display_number(result.fundamental_improvement_score)} · "
        f"Price recognition: {display_number(result.price_recognition_score)}"
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Historical price")
        price_data = price_frame(prices)
        if price_data.empty:
            st.info("No price history is available for this company at the selected as-of date.")
        else:
            st.line_chart(price_data.set_index("Date")["Close"], height=270)
    with right:
        st.markdown("#### Quarterly financial trends")
        finance_data = financial_frame(financials)
        if finance_data.empty:
            st.info("No financial history is available at the selected as-of date.")
        else:
            trend_columns = ["Revenue YoY %", "EBITDA margin %", "PAT YoY %"]
            st.line_chart(finance_data.set_index("Period")[trend_columns], height=270)
    st.markdown("#### Cash flow and balance sheet")
    finance_data = financial_frame(financials)
    if not finance_data.empty:
        st.line_chart(finance_data.set_index("Period")[["Operating cash flow", "Debt"]], height=220)

    st.markdown("#### Corporate announcement timeline")
    announcement_data = announcement_frame(announcements)
    if announcement_data.empty:
        st.info("No announcements are available by the selected as-of date.")
    else:
        st.dataframe(announcement_data, use_container_width=True, hide_index=True)


def _evidence_risk(dataset: DashboardDataset, snapshots: list[CompanySnapshot]) -> None:
    st.subheader("Evidence & Risk")
    st.caption("The dashboard makes uncertainty visible instead of hiding it behind a ranking.")
    st.markdown('<div class="lx-warning">Synthetic demonstration only. No record here is a real NSE security or a profitability claim.</div>', unsafe_allow_html=True)
    st.markdown("#### What the proxy does and does not know")
    st.markdown(
        """
        - The Expectations Gap Proxy is **not analyst consensus** and is not a true market-expectations gap.
        - Existing announcements do not include a verified guidance or consensus baseline, so announcement surprise is unavailable by default.
        - Price recognition uses absolute 1-month and 3-month returns because this offline demo has no defensible benchmark.
        - Future financial reports, announcements and prices are excluded from the fixed as-of date.
        - Missing data reduces coverage and confidence; it is never silently converted to zero.
        """
    )
    rows = []
    for snapshot in snapshots:
        result = snapshot.result
        rows.append(
            {
                "Ticker": snapshot.company.ticker,
                "Status": result.signal_status,
                "Confidence": display_percent(result.confidence_score),
                "Coverage": display_percent(result.coverage_ratio),
                "Missing / unavailable inputs": "; ".join(result.explanation.missing_data) or "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### Current evidence boundary")
    st.info(
        "This version is intentionally an offline research demonstration. "
        "It does not connect to live market data, analyst estimates, broker APIs or order execution."
    )


def main() -> None:
    st.set_page_config(page_title="LEADER-X India", page_icon="📈", layout="wide")
    _apply_style()
    _header()
    try:
        dataset = load_dataset(DATA_DIR)
        snapshots = build_snapshots(dataset)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Unable to load the offline synthetic dataset: {exc}")
        st.info("Run: uv run python -m leader_x.synthetic")
        st.stop()
    page = _sidebar(snapshots)
    if page == "Stock Discovery":
        _stock_discovery(snapshots)
    elif page == "Company Research":
        _company_research(dataset, snapshots)
    else:
        _evidence_risk(dataset, snapshots)


if __name__ == "__main__":
    main()
