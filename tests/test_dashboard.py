from pathlib import Path

from leader_x.dashboard import (
    DEMO_AS_OF,
    build_snapshots,
    discovery_frame,
    display_number,
    display_percent,
    load_dataset,
    rank_snapshots,
)
from leader_x.synthetic import generate_dataset


def test_dashboard_loads_the_existing_six_company_synthetic_dataset(tmp_path):
    data_dir = tmp_path / "synthetic"
    generate_dataset(data_dir, tmp_path / "leader_x.duckdb")
    dataset = load_dataset(data_dir)
    snapshots = build_snapshots(dataset)
    assert len(dataset.companies) == 6
    assert len(snapshots) == 6
    assert all(snapshot.result.data_quality_status == "synthetic" for snapshot in snapshots)


def test_dashboard_ranking_is_score_ordered_and_filters_status():
    dataset = load_dataset(Path("data/synthetic"))
    snapshots = build_snapshots(dataset, DEMO_AS_OF)
    ranked = rank_snapshots(snapshots)
    scores = [snapshot.result.expectations_gap_score for snapshot in ranked]
    assert scores == sorted(scores, key=lambda value: value if value is not None else -1, reverse=True)
    valid_only = rank_snapshots(snapshots, statuses={"valid"})
    assert all(snapshot.result.signal_status == "valid" for snapshot in valid_only)


def test_missing_numeric_values_are_displayed_as_unknown_not_zero():
    assert display_number(None) == "—"
    assert display_percent(None) == "—"
    assert display_number(0) == "0.0"


def test_discovery_frame_preserves_synthetic_labels_and_explanations(tmp_path):
    data_dir = tmp_path / "synthetic"
    generate_dataset(data_dir, tmp_path / "leader_x.duckdb")
    snapshots = build_snapshots(load_dataset(data_dir))
    frame = discovery_frame(rank_snapshots(snapshots))
    assert len(frame) == 6
    assert set(frame["Data"]) == {"synthetic"}
    assert frame["Why"].notna().all()
    assert all(ticker.startswith("SYN_") for ticker in frame["Ticker"])
