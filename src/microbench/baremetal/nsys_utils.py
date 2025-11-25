#!/usr/bin/env python3
"""
Small helpers for running nsys and managing trace paths.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple

from soda import utils


def get_trace_dir() -> Path:
    """
    Locate (and create) the traces directory next to the jobs file.
    """
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    trace_dir = jobs_file.parent / "traces"
    utils.ensure_dir(trace_dir)
    return trace_dir


def _nsys_rep_path(trace_output: Path) -> Path:
    """
    Resolve the actual .nsys-rep file path nsys will create.
    """
    if trace_output.suffix == ".nsys-rep":
        return trace_output
    return trace_output.with_suffix(".nsys-rep")


def nsys_profile_to_sqlite(
    trace_output: Path,
    args,
    *,
    timeout=None,
    clean_trace: bool = False,
) -> Tuple[bool, Optional[str], str]:
    """
    Run `nsys profile` followed by `nsys export` to sqlite.

    Returns: (success, sqlite_path|None, message)
    """
    rep_path = _nsys_rep_path(trace_output)
    sqlite_path = rep_path.with_suffix(".sqlite")

    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,osrt",
        "--output",
        str(trace_output),
        "--force-overwrite=true",
    ] + list(args)

    result = subprocess.run(
        nsys_cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        return False, None, result.stderr or result.stdout

    if not rep_path.exists():
        return False, None, f"Trace file not created: {rep_path}"

    utils.remove_file(sqlite_path)

    export_cmd = [
        "nsys",
        "export",
        "--type=sqlite",
        "--output",
        str(sqlite_path),
        "--force-overwrite=true",
        str(rep_path),
    ]

    export_result = subprocess.run(
        export_cmd, capture_output=True, text=True
    )
    if export_result.returncode != 0:
        return False, None, export_result.stderr or export_result.stdout

    if not sqlite_path.exists():
        return False, None, f"SQLite file not created: {sqlite_path}"

    if clean_trace:
        utils.remove_file(rep_path)

    return True, str(sqlite_path), ""
