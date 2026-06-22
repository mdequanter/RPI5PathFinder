#!/usr/bin/env python3
"""Check that the Raspberry Pi 5 CSI camera can be read.

Runs the standard `rpicam-hello` (libcamera) tooling to verify a CSI camera
(e.g. an IMX219) is detected and can be opened. Optionally captures a still
image to confirm the full pipeline works end to end.

Intended to run *on* the Raspberry Pi 5 (Bookworm, libcamera/rpicam-apps).

Usage:
    python3 check_camera.py              # detect + list cameras
    python3 check_camera.py --capture    # also grab a test still
    python3 check_camera.py --capture -o /tmp/test.jpg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout/stderr as text."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def find_tool() -> str | None:
    """Locate the camera CLI tool (newer `rpicam-*`, older `libcamera-*`)."""
    for tool in ("rpicam-hello", "libcamera-hello"):
        if shutil.which(tool):
            return tool
    return None


def list_cameras(tool: str) -> bool:
    """List available cameras. Returns True if at least one was found."""
    try:
        result = run([tool, "--list-cameras"])
    except subprocess.TimeoutExpired:
        print("ERROR: timed out while listing cameras.", file=sys.stderr)
        return False

    output = (result.stdout or "") + (result.stderr or "")
    print(output.rstrip())

    if "No cameras available" in output:
        print("\n[FAIL] No cameras detected.")
        return False

    if "Available cameras" in output and result.returncode == 0:
        print("\n[OK] Camera detected.")
        return True

    print(f"\n[FAIL] Unexpected result (exit code {result.returncode}).")
    return False


def capture_still(tool: str, output: str, timeout_ms: int = 2000) -> bool:
    """Capture a single still image to verify the capture pipeline."""
    still_tool = tool.replace("-hello", "-still")
    if not shutil.which(still_tool):
        # Fall back to rpicam-hello, which can also save a frame.
        still_tool = tool

    cmd = [still_tool, "-n", "-t", str(timeout_ms), "-o", output]
    try:
        result = run(cmd, timeout=30)
    except subprocess.TimeoutExpired:
        print("ERROR: timed out while capturing image.", file=sys.stderr)
        return False

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())

    if result.returncode == 0:
        print(f"\n[OK] Captured test image -> {output}")
        return True

    print(f"\n[FAIL] Capture failed (exit code {result.returncode}).")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the RPi5 CSI camera can be read."
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Also capture a test still image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="camera_test.jpg",
        help="Output path for the test image (default: camera_test.jpg).",
    )
    args = parser.parse_args()

    tool = find_tool()
    if tool is None:
        print(
            "ERROR: neither 'rpicam-hello' nor 'libcamera-hello' was found.\n"
            "Install with: sudo apt install -y rpicam-apps",
            file=sys.stderr,
        )
        return 2

    print(f"Using camera tool: {tool}\n")

    if not list_cameras(tool):
        return 1

    if args.capture:
        print()
        if not capture_still(tool, args.output):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())