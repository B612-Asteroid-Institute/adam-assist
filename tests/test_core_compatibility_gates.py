from __future__ import annotations

import pytest

from migration.scripts.run_core_compatibility_matrix import compare_reports
from migration.scripts.verify_rust_core_line import (
    CRATES_IO_SOURCE,
    REQUIRED_CORE_PACKAGES,
    verify_core_line,
)


def _report(core_version: str, value: float = 1.0) -> dict:
    return {
        "versions": {"adam-core": core_version, "adam-assist": "0.4.1"},
        "assist_wheel_sha256": "abc",
        "epochs_mjd_tdb": [59216.0],
        "input_schema": "orbits-schema",
        "input_schema_sha256": "input-schema-sha",
        "output_schema": "orbits-schema",
        "output_schema_sha256": "output-schema-sha",
        "propagated_values": [[value, 2.0, 3.0, 4.0, 5.0, 6.0]],
    }


def _metadata(version: str) -> dict:
    return {
        "packages": [
            {"name": name, "version": version, "source": CRATES_IO_SOURCE}
            for name in sorted(REQUIRED_CORE_PACKAGES)
        ]
    }


def test_same_wheel_reports_accept_compatible_core_05_outputs() -> None:
    summary = compare_reports([_report("0.5.7"), _report("0.5.8", 1.0 + 1.0e-14)])
    assert summary["core_versions"] == ["0.5.7", "0.5.8"]
    assert summary["assist_wheel_sha256"] == "abc"


def test_same_wheel_reports_reject_different_wheels_and_science() -> None:
    different_wheel = _report("0.5.8")
    different_wheel["assist_wheel_sha256"] = "def"
    with pytest.raises(ValueError, match="identical"):
        compare_reports([_report("0.5.7"), different_wheel])
    with pytest.raises(ValueError, match="propagation mismatch"):
        compare_reports([_report("0.5.7"), _report("0.5.8", 1.1)])
    different_schema = _report("0.5.8")
    different_schema["output_schema_sha256"] = "changed-schema-sha"
    with pytest.raises(ValueError, match="different schemas"):
        compare_reports([_report("0.5.7"), different_schema])


def test_rust_core_line_accepts_one_version_and_rejects_mixed_graph() -> None:
    selected = verify_core_line(_metadata("0.5.7"), "0.5.7")
    assert set(selected) == REQUIRED_CORE_PACKAGES
    mixed = _metadata("0.5.8")
    mixed["packages"][0]["version"] = "0.5.7"
    with pytest.raises(ValueError, match="public Core line"):
        verify_core_line(mixed, "0.5.8")
    duplicate = _metadata("0.5.8")
    duplicate["packages"].append({"name": "adam_core_rs_coords", "version": "0.5.7"})
    with pytest.raises(ValueError, match="public Core line"):
        verify_core_line(duplicate, "0.5.8")


def test_rust_core_line_rejects_non_registry_source() -> None:
    metadata = _metadata("0.5.8")
    metadata["packages"][0]["source"] = None
    with pytest.raises(ValueError, match="public Core line"):
        verify_core_line(metadata, "0.5.8")
