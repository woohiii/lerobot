#!/usr/bin/env bash
# The Astra S USB handshake is flaky (intermittent silent packet loss - see
# usbmon trace notes in orbbec_camera.py). Retrying inside one Python process
# corrupts the openni ctypes bindings, so retry at the PROCESS level instead:
# each attempt is a fresh `python3` invocation.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/youngchan/miniconda3/envs/lerobot/bin/python3
MAX_ATTEMPTS="${1:-10}"

for i in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== attempt $i/$MAX_ATTEMPTS ==="
    if "$PY" -u red_cube_detect.py --mode snapshot; then
        echo "=== SUCCESS on attempt $i ==="
        exit 0
    fi
    sleep 1
done

echo "=== all $MAX_ATTEMPTS attempts failed ==="
exit 1
