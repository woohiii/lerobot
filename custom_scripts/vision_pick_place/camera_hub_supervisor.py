#!/usr/bin/env python3
"""Supervise camera_hub.py when a UVC ``VideoCapture.read`` wedges.

``camera_hub.py`` deliberately gives each UVC camera a thread, so a slow
camera does not delay its peers.  That alone cannot recover a native V4L2
read that blocks forever: Python cannot kill that thread.  This parent owns
no camera and watches every published file; when any one goes stale it kills
and recreates the isolated hub process, which reopens all UVC devices by
their product names.  The short restart is preferable to silently using a
frozen wrist image for a grasp.

Run after stopping any directly-run camera_hub.py:
  CAMERA_HUB_HEADLESS=1 python camera_hub_supervisor.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# The external RGB camera is optional on this rig.  The two wrist feeds are
# the required inputs for bimanual grasping, so an absent RGB camera must not
# cause a restart loop that destabilizes otherwise healthy wrists.
REQUIRED_PUBLISH_PATHS = ("/tmp/vsp_wrist.png", "/tmp/vsp_wrist_left.png")


def stale_paths(paths: tuple[str, ...], stale_seconds: float) -> list[str]:
    now = time.time()
    stale: list[str] = []
    for path in paths:
        try:
            if now - os.path.getmtime(path) > stale_seconds:
                stale.append(path)
        except FileNotFoundError:
            stale.append(path)
    return stale


def stop(child: subprocess.Popen[object]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stale-seconds", type=float, default=8.0)
    parser.add_argument("--check-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.stale_seconds <= 0 or args.check_seconds <= 0:
        parser.error("stale/check seconds must be positive")

    stopping = False

    def request_stop(*_unused: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    child: subprocess.Popen[object] | None = None
    try:
        while not stopping:
            # Do not remove files: a running child overwrites them and an old
            # direct hub remains diagnosable until the operator stops it.
            env = os.environ.copy()
            env["CAMERA_HUB_HEADLESS"] = "1"
            child = subprocess.Popen([sys.executable, str(ROOT / "camera_hub.py")], env=env)
            print(f"[camera_hub_supervisor] started pid={child.pid}", flush=True)
            while not stopping:
                time.sleep(args.check_seconds)
                stale = stale_paths(REQUIRED_PUBLISH_PATHS, args.stale_seconds)
                if child.poll() is not None:
                    print(f"[camera_hub_supervisor] child exited ({child.returncode}); restarting", flush=True)
                    break
                if stale:
                    print(f"[camera_hub_supervisor] stale publish: {', '.join(stale)}; restarting hub", flush=True)
                    break
            stop(child)
            child = None
    finally:
        if child is not None:
            stop(child)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
