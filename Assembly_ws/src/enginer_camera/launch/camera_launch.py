from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = Path(get_package_share_directory("enginer_camera")) / "config" / "camera.yaml"
    return LaunchDescription([
        Node(
            package="enginer_camera",
            executable="camera_node",
            name="camera_node",
            output="screen",
            parameters=[str(config_path)],
        ),
    ])
