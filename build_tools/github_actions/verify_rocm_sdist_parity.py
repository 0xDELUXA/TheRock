#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify content-parity between two rocm meta sdists.

The rocm sdist is built independently by the Linux and Windows
multi-arch release jobs and uploaded to the same S3 key. If they
diverge, the last-writer-wins overwrite silently drops one platform's
device extras from the published index. This script extracts both
tarballs and runs a recursive content comparison; on mismatch it
emits both PKG-INFO files plus a directory diff and exits non-zero.

Scope: intended for per-run artifact directories (each containing
exactly one rocm-*.tar.gz). Not meant for published release indexes,
which accumulate multiple versions over time.

Usage:
    verify_rocm_sdist_parity.py
        --linux-tarball-dir <dir>
        --windows-tarball-dir <dir>
        [--workdir <dir>]
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def find_single_sdist(directory: Path, label: str) -> Path:
    """Return the single rocm-*.tar.gz in `directory`, fail-fast otherwise.

    The script is designed for per-run artifact directories where exactly
    one rocm sdist exists. Any other count is a configuration error and
    must surface loudly rather than silently picking one.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{label} tarball directory does not exist: {directory}"
        )
    candidates = sorted(directory.glob("rocm-*.tar.gz"))
    if not candidates:
        listing = sorted(p.name for p in directory.iterdir())
        raise FileNotFoundError(
            f"No rocm-*.tar.gz in {label} dir {directory}. "
            f"Directory contents: {listing}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one rocm-*.tar.gz in {label} dir {directory}, "
            f"found {len(candidates)}: {[p.name for p in candidates]}. "
            f"This script is scoped to per-run artifact directories, not "
            f"multi-version release indexes."
        )
    return candidates[0]


def extract(tarball: Path, dest: Path) -> Path:
    """Extract `tarball` into `dest` and return its sole top-level directory.

    sdists pack a single root directory (e.g. rocm-7.13.0/); any other
    layout means the sdist itself is malformed and is treated as a hard
    error.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level directory in {tarball}, "
            f"found {[c.name for c in children]}"
        )
    return children[0]


def trees_content_equal(left: Path, right: Path) -> bool:
    """Recursive deep content comparison, mirroring 'diff -r' semantics.

    Returns True when every file under `left` has a counterpart under
    `right` with identical bytes, and vice versa. Does not consider tar
    metadata (mtime, uid/gid) since extracted sdists may legitimately
    differ on those without affecting installed behavior.
    """
    cmp = filecmp.dircmp(str(left), str(right))
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatches, errors = filecmp.cmpfiles(
        str(left), str(right), cmp.common_files, shallow=False
    )
    if mismatches or errors:
        return False
    for sub in cmp.common_dirs:
        if not trees_content_equal(left / sub, right / sub):
            return False
    return True


def emit_pkg_info(label: str, root: Path) -> None:
    """Print a labelled PKG-INFO section for the GitHub Actions log."""
    pkg_info = root / "PKG-INFO"
    print(f"::group::{label} PKG-INFO ({pkg_info})")
    if pkg_info.is_file():
        print(pkg_info.read_text())
    else:
        print("(missing)")
    print("::endgroup::")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--linux-tarball-dir",
        type=Path,
        required=True,
        help="Directory containing the Linux build's rocm-*.tar.gz.",
    )
    p.add_argument(
        "--windows-tarball-dir",
        type=Path,
        required=True,
        help="Directory containing the Windows build's rocm-*.tar.gz.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Directory for extracting tarballs. Defaults to "
            "<linux-tarball-dir>/../extracted; created if absent and wiped "
            "if it already exists."
        ),
    )
    args = p.parse_args(argv)

    linux_tarball = find_single_sdist(args.linux_tarball_dir, "Linux")
    windows_tarball = find_single_sdist(args.windows_tarball_dir, "Windows")

    workdir = args.workdir or args.linux_tarball_dir.parent / "extracted"
    if workdir.exists():
        shutil.rmtree(workdir)
    linux_root = extract(linux_tarball, workdir / "linux")
    windows_root = extract(windows_tarball, workdir / "windows")

    if trees_content_equal(linux_root, windows_root):
        print(
            f"rocm sdists are content-identical across Linux and Windows "
            f"({linux_tarball.name})"
        )
        return 0

    print(
        "::error::rocm sdists differ between Linux and Windows multi-arch "
        "release builds"
    )
    emit_pkg_info("Linux", linux_root)
    emit_pkg_info("Windows", windows_root)
    # Capture diff output instead of letting it stream directly to fd 1,
    # so it lands inside the ::group:: block rather than racing with
    # Python's stdout buffer.
    diff_result = subprocess.run(
        ["diff", "-r", str(linux_root), str(windows_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    print("::group::Recursive diff")
    if diff_result.stdout:
        print(diff_result.stdout, end="")
    if diff_result.stderr:
        print(diff_result.stderr, end="", file=sys.stderr)
    print("::endgroup::")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
