#!/usr/bin/env bash
set -euo pipefail
cd ~/lerobot

env -u PYTHONPATH .venv/bin/python custom_scripts/lerobot_record_home_reset.py \
  --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower \
  --teleop.type=so101_leader  --teleop.port=/dev/so101_leader  --teleop.id=leader \
  --robot.cameras='{ cam1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=youngchan/so101_glue_pickup \
  --dataset.single_task="Pick up the glue container and lift it." \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=6 \
  --dataset.push_to_hub=false \
  --display_data=true
