"""Bimanual entry point: sequences BOTH arms of a bi_so_follower through the
existing single-arm task_state_machine.run(), one after the other.

Strictly sequential, never concurrent - this is the MVP scope (Phase 3 of
the bimanual plan). True cooperative bimanual (e.g. one arm holding an
object while the other manipulates it) is an explicit stretch goal, not
built here. Because of that sequential guarantee, config.apply_side(side)
(a shared-module-global swap) is safe to use as the per-arm calibration
switch - see config.py's apply_side docstring.

Camera devices are NOT opened here, same convention as main.py: camera_hub.py
must already be running and publishing both wrist frames (config.WRIST_FRAME_PATH
for left, its right-arm override for right) before this starts.

Run (ports have no hardcoded default - this hardware has a known port-drift
problem on USB replug):
  uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/orchestrator_bimanual.py \\
      --left-port /dev/so101_follower_left --right-port /dev/so101_follower_right
"""

from __future__ import annotations

import argparse
import sys
import time

import config
import perception
from kinematics import SOArm101
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig


def run_one_side(bi: BiSOFollower, side: str, grasp_strategy: str, policy_path: str | None) -> bool:
    config.apply_side(side)  # must happen before building SOArm101/cap below - both read config.X bare
    arm = SOArm101(robot=bi.left_arm if side == "left" else bi.right_arm)
    arm.connect()  # no-op on the underlying connection (already connected via bi.connect()); still applies _protect_arm_motors()

    cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not cap.isOpened():
        print(f"[orchestrator] {side} 팔: 손목캠 프레임이 없습니다 ({config.WRIST_FRAME_PATH}). camera_hub.py가 이 팔의 카메라를 발행 중인지 확인하세요.")
        return False

    print(f"[orchestrator] === {side} 팔 시작 === joints={arm.get_joint_deg()}")
    time.sleep(1)

    import task_state_machine

    return task_state_machine.run(arm, cap, grasp_strategy=grasp_strategy, policy_path=policy_path)


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True, help="왼쪽 팔 follower 포트 (예: /dev/so101_follower_left)")
    parser.add_argument("--right-port", required=True, help="오른쪽 팔 follower 포트 (예: /dev/so101_follower_right)")
    parser.add_argument("--grasp-strategy", choices=["legacy", "il"], default="legacy",
                         help="양팔 모두에 적용할 파지 전략 (기본: %(default)r)")
    parser.add_argument("--policy-path", default=None, help="grasp-strategy=il일 때 필요한 학습된 ACT 체크포인트 경로")
    args = parser.parse_args()

    if args.grasp_strategy == "il" and not args.policy_path:
        parser.error("--grasp-strategy il requires --policy-path")

    bi_config = BiSOFollowerConfig(
        id="bi_follower",
        left_arm_config=SOFollowerConfig(port=args.left_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
        right_arm_config=SOFollowerConfig(port=args.right_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
    )
    bi = BiSOFollower(bi_config)
    bi.connect(calibrate=False)
    print("[orchestrator] 양팔 연결 성공")

    results: dict[str, bool] = {}
    try:
        for side in ["left", "right"]:
            try:
                results[side] = run_one_side(bi, side, args.grasp_strategy, args.policy_path)
            except KeyboardInterrupt:
                print(f"\n[orchestrator] {side} 팔 진행 중 사용자가 중단했습니다.")
                results[side] = False
                raise
    except KeyboardInterrupt:
        pass
    finally:
        bi.disconnect()

    print("\n=== 결과 요약 ===")
    for side in ["left", "right"]:
        status = "성공" if results.get(side) else "실패/미실행"
        print(f"  {side}: {status}")

    return all(results.get(side, False) for side in ["left", "right"])


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
