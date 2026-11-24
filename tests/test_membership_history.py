from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


MEMBERSHIP_PATH = Path(__file__).resolve().parents[1] / "VN30_membership_history.csv"


EXPECTED_TRANSITIONS = {
    "2016-01-25": ({"NT2", "SBT"}, {"CSM", "VSH"}),
    "2016-07-25": ({"BID", "GAS", "HNG", "MWG"}, {"EIB", "HHS", "HVG", "PVT"}),
    "2017-01-23": ({"BMP", "CTD", "DHG"}, {"FLC", "HCM", "PPC"}),
    "2017-07-24": ({"NVL", "ROS", "SAB"}, {"HAG", "HNG", "ITA"}),
    "2018-01-22": ({"PLX", "VJC"}, {"KBC", "PVD"}),
    "2018-07-23": ({"PNJ", "VPB", "VRE"}, {"BID", "BVH", "NT2"}),
    "2019-02-11": ({"EIB", "HDB", "TCB", "VHM"}, {"BMP", "HSG", "KDC", "PLX"}),
    "2019-08-05": ({"BID", "BVH"}, {"CII", "DHG"}),
    "2020-02-03": ({"PLX", "POW"}, {"DPM", "GMD"}),
    "2020-08-03": ({"KDH", "TCH"}, {"BVH", "CTD"}),
    "2021-02-01": ({"BVH", "PDR", "TPB"}, {"EIB", "ROS", "SAB"}),
    "2021-08-02": ({"ACB", "GVR", "SAB"}, {"REE", "SBT", "TCH"}),
    "2022-02-07": (set(), set()),
    "2022-08-01": ({"VIB"}, {"PNJ"}),
    "2023-02-06": ({"BCM"}, {"KDH"}),
    "2023-08-07": ({"SHB", "SSB"}, {"NVL", "PDR"}),
    "2024-02-05": (set(), set()),
    "2024-08-05": (set(), set()),
    "2025-02-03": ({"LPB"}, {"POW"}),
    "2025-08-04": ({"DGC"}, {"BVH"}),
}


def test_committed_vn30_history_has_expected_periods_and_transitions() -> None:
    if not MEMBERSHIP_PATH.exists():
        pytest.skip(
            "point-in-time HOSE membership input has not been restored; "
            "production loading still fails closed when it is absent"
        )
    membership = pd.read_csv(MEMBERSHIP_PATH)
    assert len(membership) == 630
    assert membership["ticker"].nunique() == 63
    assert not membership.duplicated(
        ["effective_from", "effective_to", "ticker"]
    ).any()

    periods = list(
        membership.groupby(["effective_from", "effective_to"], sort=True)
    )
    assert len(periods) == 21
    previous: set[str] | None = None
    observed: dict[str, tuple[set[str], set[str]]] = {}
    for (start, _end), rows in periods:
        current = set(rows["ticker"])
        assert len(current) == 30
        if previous is not None:
            observed[start] = (current - previous, previous - current)
        previous = current

    assert observed == EXPECTED_TRANSITIONS
