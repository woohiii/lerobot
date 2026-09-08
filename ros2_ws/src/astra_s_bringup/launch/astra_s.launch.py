# ros_astra_camera(ros2-development)의 astra_mini.launch.py를 그대로
# 따르되, astra_s_params.yaml을 사용하도록만 바꾼 launch 파일.
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python import get_package_share_directory
import yaml


def generate_launch_description():
    params_file = get_package_share_directory("astra_s_bringup") + "/params/astra_s_params.yaml"
    with open(params_file, "r") as file:
        config_params = yaml.safe_load(file)
    container = ComposableNodeContainer(
        name="astra_camera_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="astra_camera",
                plugin="astra_camera::OBCameraNodeFactory",
                name="camera",
                namespace="camera",
                parameters=[config_params],
            ),
            ComposableNode(
                package="astra_camera",
                plugin="astra_camera::PointCloudXyzrgbNode",
                namespace="camera",
                name="point_cloud_xyzrgb",
            ),
        ],
        output="screen",
    )
    return LaunchDescription([container])
