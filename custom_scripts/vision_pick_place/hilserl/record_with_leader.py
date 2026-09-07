"""Thin launcher for `python -m lerobot.rl.gym_manipulator` that patches
`SOLeader` (so101_leader/so100_leader) with the `get_teleop_events()` method
HIL-SERL's processor pipeline requires (see hil_processor.py's
`AddTeleopEventsAsInfoStep` / `_check_teleop_with_events` - it hard-codes
"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop", SOLeader
is NOT in that list even though docs/source/hilserl.mdx's "Setting up the
SO101 leader" section documents `control_mode: "leader"` as supported).
Confirmed live 2026-09-02: running gym_manipulator.py directly with
`--teleop.type=so101_leader` raises
  TypeError: Teleoperator SOLeader must implement get_teleop_events() ...
at pipeline construction (before any hardware motion - safe to hit, just
doesn't work). This patches the gap without touching lerobot core: the
leader arm keeps driving the follower via its own get_action() (motor
positions) as always; this only ADDS the missing event method, read via a
pynput keyboard listener running alongside it - same library and pattern
teleop_keyboard.py's KeyboardEndEffectorTeleop already uses for the same
purpose, just without stealing the arrow keys for movement since the leader
arm (not the keyboard) drives the robot here.

Per the user's 2026-09-02 request, keys are NOT the doc's s/esc/space:
  - Right arrow: success (save episode)
  - Left arrow:  rerecord (redo episode)
  - Esc:         failure (save as failed episode)
  - Space:       toggle intervention on/off (only matters once you get to
                 online actor/learner training with a policy running -
                 during plain `mode: "record"` there is no policy, so
                 intervention defaults to ON and space is unused)

Run exactly like gym_manipulator.py, e.g.:
  uv run python custom_scripts/vision_pick_place/hilserl/record_with_leader.py \
    --config_path custom_scripts/vision_pick_place/hilserl/env_config_record.json
"""

import sys
import threading
from typing import Any

from pynput import keyboard

from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.utils import TeleopEvents


def _patched_init(self, config):
    self.__orig_init__(config)
    self._event_lock = threading.Lock()
    self._pending_success = False
    self._pending_rerecord = False
    self._pending_terminate = False
    self._intervention_active = True  # no policy in "record" mode - human always drives
    self._key_listener = None


def _on_press(self, key):
    with self._event_lock:
        if key == keyboard.Key.right:
            self._pending_success = True
            self._pending_terminate = True
        elif key == keyboard.Key.left:
            self._pending_rerecord = True
            self._pending_terminate = True
        elif key == keyboard.Key.esc:
            self._pending_success = False
            self._pending_terminate = True
        elif key == keyboard.Key.space:
            self._intervention_active = not self._intervention_active
            print(f"[record_with_leader] intervention {'ON' if self._intervention_active else 'OFF'}")


def _patched_connect(self, *args, **kwargs):
    self.__orig_connect__(*args, **kwargs)
    self._key_listener = keyboard.Listener(on_press=lambda k: self._on_press(k))
    self._key_listener.start()
    print("[record_with_leader] keyboard listener ready: -> save, <- redo, esc=fail, space=toggle intervention")


def _patched_disconnect(self, *args, **kwargs):
    if self._key_listener is not None:
        self._key_listener.stop()
        self._key_listener = None
    self.__orig_disconnect__(*args, **kwargs)


def _get_teleop_events(self) -> dict[str, Any]:
    with self._event_lock:
        success = self._pending_success
        rerecord = self._pending_rerecord
        terminate = self._pending_terminate
        self._pending_success = False
        self._pending_rerecord = False
        self._pending_terminate = False
    return {
        TeleopEvents.IS_INTERVENTION: self._intervention_active,
        TeleopEvents.TERMINATE_EPISODE: terminate,
        TeleopEvents.SUCCESS: success,
        TeleopEvents.RERECORD_EPISODE: rerecord,
    }


SOLeader.__orig_init__ = SOLeader.__init__
SOLeader.__orig_connect__ = SOLeader.connect
SOLeader.__orig_disconnect__ = SOLeader.disconnect
SOLeader.__init__ = _patched_init
SOLeader.connect = _patched_connect
SOLeader.disconnect = _patched_disconnect
SOLeader._on_press = _on_press
SOLeader.get_teleop_events = _get_teleop_events

if __name__ == "__main__":
    from lerobot.rl.gym_manipulator import main

    sys.exit(main())
