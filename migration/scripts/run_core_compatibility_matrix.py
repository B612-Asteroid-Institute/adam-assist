#!/usr/bin/env python3
"""Install one ASSIST wheel beside multiple Core wheels and compare smokes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(
    command: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, env=env, check=True)


def compare_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two Core compatibility reports are required")
    assist_versions = {report["versions"]["adam-assist"] for report in reports}
    wheel_hashes = {report["assist_wheel_sha256"] for report in reports}
    if len(assist_versions) != 1 or len(wheel_hashes) != 1:
        raise ValueError(
            "compatibility runs did not use one identical adam-assist wheel"
        )
    reference = reports[0]
    reference_values = reference["propagated_values"]
    reference_epochs = reference["epochs_mjd_tdb"]
    reference_schemas = (
        reference["input_schema_sha256"],
        reference["output_schema_sha256"],
    )
    for report in reports[1:]:
        schemas = (
            report["input_schema_sha256"],
            report["output_schema_sha256"],
        )
        if schemas != reference_schemas:
            raise ValueError("Core compatibility runs returned different schemas")
        if report["epochs_mjd_tdb"] != reference_epochs:
            raise ValueError("Core compatibility runs returned different epochs")
        for expected_row, actual_row in zip(
            reference_values, report["propagated_values"], strict=True
        ):
            for expected, actual in zip(expected_row, actual_row, strict=True):
                tolerance = max(1.0e-14, abs(expected) * 1.0e-13)
                if abs(expected - actual) > tolerance:
                    raise ValueError(
                        "Core compatibility propagation mismatch: "
                        f"{actual} vs {expected} (tolerance {tolerance})"
                    )
    return {
        "schema_version": 1,
        "status": "passed",
        "assist_version": next(iter(assist_versions)),
        "assist_wheel_sha256": next(iter(wheel_hashes)),
        "core_versions": [report["versions"]["adam-core"] for report in reports],
    }


def _parse_core_wheel(value: str) -> tuple[str, Path]:
    version, separator, path = value.partition("=")
    if not separator or not version or not path:
        raise argparse.ArgumentTypeError("Core wheel must be VERSION=PATH")
    return version, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--assist-wheel", type=Path, required=True)
    parser.add_argument("--assist-version", required=True)
    parser.add_argument(
        "--core-wheel", action="append", type=_parse_core_wheel, required=True
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    assist_wheel = args.assist_wheel.resolve()
    smoke_script = Path(__file__).with_name("core_compatibility_smoke.py").resolve()
    reports: list[dict[str, Any]] = []

    for core_version, core_wheel_value in args.core_wheel:
        core_wheel = core_wheel_value.resolve()
        runtime = workspace / f"core-{core_version}"
        _run([args.python.resolve(), "-m", "venv", runtime], cwd=workspace)
        python = _venv_python(runtime)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        for name in list(environment):
            if name.startswith(("ADAM_CORE_KERNEL_", "ADAM_CORE_RS_ASSIST_")):
                environment.pop(name)
        environment.update(
            {
                "ADAM_CORE_KERNEL_OFFLINE": "1",
                "ADAM_CORE_KERNEL_CACHE": str(runtime / "kernel-cache"),
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_ONLY_BINARY": ":all:",
                "PYTHONNOUSERSITE": "1",
            }
        )
        _run(
            [python, "-m", "pip", "install", "--upgrade", "pip"],
            cwd=workspace,
            env=environment,
        )
        _run(
            [python, "-m", "pip", "install", core_wheel, assist_wheel],
            cwd=workspace,
            env=environment,
        )
        _run([python, "-m", "pip", "check"], cwd=workspace, env=environment)
        report_path = workspace / f"core-{core_version}.json"
        _run(
            [
                python,
                smoke_script,
                "--expected-core-version",
                core_version,
                "--expected-assist-version",
                args.assist_version,
                "--assist-wheel",
                assist_wheel,
                "--report",
                report_path,
            ],
            cwd=workspace,
            env=environment,
        )
        reports.append(json.loads(report_path.read_text()))

    summary = compare_reports(reports)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
