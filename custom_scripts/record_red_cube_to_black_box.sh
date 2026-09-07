#!/usr/bin/env bash
# Record 30 SO-101 demonstrations with Astra S RGB + metric depth and wrist RGB.
set -euo pipefail

CAMERAS='{astra: {type: orbbec, width: 640, height: 480, depth_width: 320, depth_height: 240, fps: 30, use_rgb: true, use_depth: true, preview: true}, wrist: {type: opencv, index_or_path: /dev/video5, width: 640, height: 480, fps: 30, backend: 200, warmup_s: 5, preview: true, preview_name: "Wrist camera (recording)"}}'

exec uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader \
  --dataset.repo_id=youngchan/so101_red_cube_to_black_box \
  --dataset.root=data/so101_red_cube_to_black_box_30eps_v9 \
  --dataset.single_task="Pick up the red cube and place it inside the black box." \
  --dataset.fps=30 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=30 \
  --dataset.video=true \
  --dataset.push_to_hub=false
