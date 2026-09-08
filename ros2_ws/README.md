# Astra S ROS2 드라이버 (준비 단계)

`ros_astra_camera` (upstream, `ros2-development` 브랜치) 를 submodule로 가져와
Astra S용 파라미터/launch로 감싼 것. **기존 `custom_scripts/vision_pick_place/`의
openni2/파일 기반 파이프라인은 그대로 유지**하며, 이 워크스페이스는 별도로 존재.

> ⚠️ `ros2-development` 브랜치는 upstream README에 "now is not stable, please
> do not use it"라고 명시돼 있음. 현재 시점에 존재하는 유일한 공식 ROS2 브랜치라
> 이걸 씀. 빌드/토픽 발행이 실제로 되는지는 아래 검증 절차로 직접 확인할 것.

## 구성

- `src/ros_astra_camera/` — git submodule, upstream 그대로 (건드리지 않음)
- `src/astra_s_bringup/` — 이 저장소에서 새로 만든 launch 패키지
  - `params/astra_s_params.yaml`: Astra S 전용 값 (컬러는 UVC가 아니라
    OpenNI2 스트림 사용 — `orbbec_color_camera.py`에서 이미 확인된 사실)
  - `launch/astra_s.launch.py`: 위 파라미터로 `astra_camera` 컴포넌트 실행

## 빌드 전 준비

1. Orbbec 벤더 OpenNI2 드라이버(`liborbbec.so`)가 필요함. 이 저장소
   `custom_scripts/vision_pick_place/openni2_redist/`에 이미 있는 것을 그대로
   심볼릭 링크로 연결 (한 번만):
   ```bash
   mkdir -p src/ros_astra_camera/astra_camera/openni2_redist
   ln -s "$(git rev-parse --show-toplevel)/custom_scripts/vision_pick_place/openni2_redist" \
       src/ros_astra_camera/astra_camera/openni2_redist/x64
   ```
2. udev 규칙 (일반 사용자 권한으로 USB 접근):
   ```bash
   sudo cp src/ros_astra_camera/astra_camera/scripts/56-orbbec-usb.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
3. 의존 패키지 설치:
   ```bash
   sudo apt install -y libuvc-dev libgoogle-glog-dev
   rosdep install --from-paths src --ignore-src -r -y
   ```

## 빌드 & 실행

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to astra_s_bringup
source install/setup.bash
ros2 launch astra_s_bringup astra_s.launch.py
```

## 검증

다른 터미널에서 (카메라 연결된 상태):

```bash
source install/setup.bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_raw
```

두 토픽이 정상적인 Hz로 뜨면 드라이버가 살아있는 것. 안 뜨면 `ros2 launch`
콘솔 로그에서 OpenNI2 디바이스 오픈 실패 여부부터 확인.

## 스코프 밖 (다음 단계)

`task_trash_to_bin/perception.py`, `calibrate_camera.py` 등 소비자 코드는
아직 이 ROS2 토픽을 구독하지 않음 — 이번 작업은 드라이버 노드 + 토픽 발행까지.
