#!/usr/bin/env python3
"""Keep one Astra S OpenNI2 stream alive without blocking wrist cameras.

Run this *instead of* directly running ``astra_s_depth_hub.py`` or
``astra_s_ir_hub.py``.  The OpenNI2 native ``read_frame`` call can wedge
forever on this Astra S/driver.  A thread cannot recover from that state: it
cannot be killed and it can make a cached frame look fresh.  This parent stays
outside the OpenNI2 process, watches the publish file, then terminates and
restarts the child (optionally USB-resets the device) when no real frame is
published.

Only one process can own one Astra S.  ``rgbd`` starts its OpenNI2 colour and
depth streams together and publishes BOTH ``/tmp/vsp_astra_rgb.png`` and
``/tmp/vsp_astra_depth_mm.npy``.  The parent watches both files, so a colour
or depth wedge restarts the isolated process.  ``ir`` is a fallback mode only:
the Astra S structured-light depth stream and IR stream cannot run together.

Examples (headless):
  python astra_s_stream_supervisor.py --mode rgbd
  python astra_s_stream_supervisor.py --mode ir
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from camera_utils import ASTRA_DEPTH_MM_PATH, ASTRA_IR_FRAME_PATH


ROOT = Path(__file__).resolve().parent
STREAMS = {
    "rgbd": (ROOT / "astra_s_depth_hub.py", (ASTRA_DEPTH_MM_PATH, "/tmp/vsp_astra_rgb.png")),
    "ir": (ROOT / "astra_s_ir_hub.py", (ASTRA_IR_FRAME_PATH,)),
}


def _age_s(path: str) -> float:
    try:
        return time.time() - os.path.getmtime(path)
    except FileNotFoundError:
        return float("inf")


def _astra_usb_node() -> str | None:
    """Return Astra's current bus node; the device number changes after reset."""
    try:
        output = subprocess.run(
            ["lsusb", "-d", "2bc5:0402"], text=True, capture_output=True, timeout=3, check=False
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    # Example: "Bus 001 Device 034: ID 2bc5:0402 ..."
    fields = output.split()
    try:
        return f"/dev/bus/usb/{int(fields[1]):03d}/{int(fields[3].rstrip(':')):03d}"
    except (IndexError, ValueError):
        return None


def _existing_owner() -> tuple[str, str] | None:
    """Return (device, pid-list) if another process already owns the Astra."""
    node = _astra_usb_node()
    if node is None:
        return None
    try:
        result = subprocess.run(["lsof", "-t", node], text=True, capture_output=True, timeout=3, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    pids = ", ".join(line for line in result.stdout.splitlines() if line.isdigit())
    return (node, pids) if pids else None


def _stop(child: subprocess.Popen[object]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _usb_reset(device_id: str) -> None:
    """Best-effort reset; the next child start still reports a clear failure."""
    try:
        completed = subprocess.run(["usbreset", device_id], text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[astra_supervisor] USB reset skipped: {exc}", flush=True)
        return
    detail = (completed.stdout + completed.stderr).strip()
    print(f"[astra_supervisor] usbreset exit={completed.returncode}: {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=tuple(STREAMS), default="rgbd")
    parser.add_argument("--stale-seconds", type=float, default=8.0)
    parser.add_argument("--check-seconds", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--usb-device-id", default="2bc5:0402", help="usbreset ID; empty string disables reset")
    parser.add_argument("--no-usb-reset", action="store_true", help="restart child only")
    args = parser.parse_args()
    if args.stale_seconds <= 0 or args.check_seconds <= 0 or args.max_consecutive_failures < 1:
        parser.error("stale/check seconds and max consecutive failures must be positive")

    child_script, publish_paths = STREAMS[args.mode]
    if not child_script.exists():
        parser.error(f"missing child publisher: {child_script}")
    owner = _existing_owner()
    if owner is not None:
        node, pids = owner
        print(
            f"[astra_supervisor] Astra is already busy ({node}, PID {pids}). "
            "Stop the existing astra_s_stream_supervisor.py/other OpenNI2 owner before starting another one.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    child: subprocess.Popen[object] | None = None
    stopping = False
    consecutive_failures = 0

    def request_stop(*_unused: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping:
            # Unlinking old frames prevents a previous publisher's mtime from
            # being mistaken for a successful new child startup.
            for publish_path in publish_paths:
                Path(publish_path).unlink(missing_ok=True)
            env = os.environ.copy()
            env[f"ASTRA_{args.mode.upper()}_HUB_HEADLESS"] = "1"
            child = subprocess.Popen([sys.executable, str(child_script)], env=env)
            print(f"[astra_supervisor] started {args.mode} publisher pid={child.pid}", flush=True)
            started_at = time.monotonic()
            restart_reason = ""

            while not stopping:
                time.sleep(args.check_seconds)
                if child.poll() is not None:
                    restart_reason = f"child exited ({child.returncode})"
                    print(f"[astra_supervisor] {restart_reason}", flush=True)
                    break
                stale = [path for path in publish_paths if _age_s(path) > args.stale_seconds]
                if stale:
                    restart_reason = f"stale publish: {', '.join(stale)}"
                    print(
                        f"[astra_supervisor] {restart_reason}; "
                        "restarting isolated OpenNI2 child",
                        flush=True,
                    )
                    _stop(child)
                    if not args.no_usb_reset and args.usb_device_id:
                        _usb_reset(args.usb_device_id)
                        time.sleep(3)
                    break
            _stop(child)
            child = None
            # An immediate open failure (not a later device stall) should not
            # busy-loop.  Give libusb time to release the interface, then
            # stop with an actionable message after the configured bound.
            if time.monotonic() - started_at < args.stale_seconds:
                consecutive_failures += 1
                if consecutive_failures >= args.max_consecutive_failures:
                    print(
                        f"[astra_supervisor] giving up after {consecutive_failures} immediate failures "
                        f"({restart_reason or 'publisher never became healthy'}). Check device ownership with "
                        "`lsof /dev/bus/usb/<bus>/<device>` after `lsusb -d 2bc5:0402`.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                delay = min(2 ** (consecutive_failures - 1), 8)
                print(f"[astra_supervisor] retrying in {delay}s ({consecutive_failures}/{args.max_consecutive_failures})", flush=True)
                time.sleep(delay)
            else:
                consecutive_failures = 0
    finally:
        if child is not None:
            _stop(child)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
